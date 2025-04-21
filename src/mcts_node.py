# src/mcts_node.py v1.4
"""
Представление узла дерева MCTS для OFC Pineapple.
Содержит состояние игры, статистику посещений/наград и логику MCTS (выбор, расширение, симуляция).
Использует GameState.advance_state() для симуляции.
"""

import math
import time
import random
import multiprocessing
import traceback
import sys
from typing import Optional, Any, List, Tuple, Set, Dict
from collections import Counter

# Импорты из src пакета
from src.game_state import GameState
from src.fantasyland_solver import FantasylandSolver
from src.card import card_to_str, Card as CardUtils # Используем алиас для утилит
from src.scoring import (
    RANK_CLASS_QUADS, RANK_CLASS_TRIPS, get_hand_rank_safe,
    check_board_foul, get_row_royalty, RANK_CLASS_PAIR,
    RANK_CLASS_HIGH_CARD
)

# Функция-воркер для параллельного роллаута
def run_parallel_rollout(node_state_dict: dict) -> Tuple[float, Set[Any]]:
    """ Выполняет один роллаут из заданного состояния в отдельном процессе. """
    try:
        # Пересоздаем зависимости внутри воркера
        from src.game_state import GameState as WorkerGameState
        from src.mcts_node import MCTSNode as WorkerMCTSNode # Импортируем здесь для доступа к политикам

        game_state = WorkerGameState.from_dict(node_state_dict)
        # Если состояние уже терминальное, сразу возвращаем счет
        if game_state.is_round_over():
            score_p0 = game_state.get_terminal_score()
            return float(score_p0), set()

        # Создаем временный узел для вызова rollout (или просто вызываем статическую/внешнюю функцию rollout)
        # Передаем game_state напрямую в функцию симуляции, чтобы избежать создания лишнего узла
        reward, sim_actions = WorkerMCTSNode.static_rollout_simulation(game_state, perspective_player=0)
        return reward, sim_actions
    except Exception as e:
        print(f"Error in parallel rollout worker: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 0.0, set()


class MCTSNode:
    """ Узел в дереве поиска Монте-Карло (MCTS). """
    # (Атрибуты и __init__ без изменений)
    def __init__(self, game_state: GameState, parent: Optional['MCTSNode'] = None, action: Optional[Any] = None):
        self.game_state: GameState = game_state
        self.parent: Optional['MCTSNode'] = parent
        self.action: Optional[Any] = action
        self.children: Dict[Any, 'MCTSNode'] = {}
        self.untried_actions: Optional[List[Any]] = None
        self.visits: int = 0
        self.total_reward: float = 0.0 # Всегда с точки зрения P0
        self.rave_visits: Dict[Any, int] = {}
        self.rave_total_reward: Dict[Any, float] = {} # Всегда с точки зрения P0

    # (Методы _get_player_to_move, expand, is_terminal без изменений)
    def _get_player_to_move(self) -> int:
        return self.game_state.get_player_to_move()

    def expand(self) -> Optional['MCTSNode']:
        player_to_move = self._get_player_to_move()
        if player_to_move == -1: return None
        if self.untried_actions is None:
            self.untried_actions = self.game_state.get_legal_actions_for_player(player_to_move)
            random.shuffle(self.untried_actions)
            for act in self.untried_actions:
                 try: hash(act);
                 except TypeError: continue
                 if act not in self.rave_visits: self.rave_visits[act] = 0; self.rave_total_reward[act] = 0.0
        if not self.untried_actions: return None
        action = self.untried_actions.pop()
        next_state: Optional[GameState] = None; original_state_repr = self.game_state.get_state_representation()
        try:
            if isinstance(action, tuple) and action[0] == "FANTASYLAND_INPUT":
                 # Не расширяем узлы Фантазии здесь, они обрабатываются в choose_action
                 print("Debug: Skipping expansion of FANTASYLAND_INPUT action.", file=sys.stderr)
                 return self.expand() if self.untried_actions else None # Пытаемся взять следующее действие
            else:
                 # Применяем обычное действие
                 next_state = self.game_state.apply_action(player_to_move, action)

            # Проверяем, изменилось ли состояние
            if next_state is self.game_state or next_state.get_state_representation() == original_state_repr:
                print(f"Error: apply_action returned the same state for P{player_to_move}.", file=sys.stderr)
                print(f"Action: {action}", file=sys.stderr)
                # Пытаемся взять следующее действие, если есть
                return self.expand() if self.untried_actions else None
        except Exception as e:
            print(f"Error applying action during expand for P{player_to_move}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            # Пытаемся взять следующее действие, если есть
            return self.expand() if self.untried_actions else None

        # Создаем дочерний узел
        child_node = MCTSNode(next_state, parent=self, action=action)
        # Добавляем в словарь children, только если действие хешируемое
        try:
            hash(action)
            self.children[action] = child_node
        except TypeError:
             print(f"Warning: Action {action} is not hashable, child node created but not added to children dict.", file=sys.stderr)
             # Узел создан, но не будет выбран через UCT, если действие не хешируемое.
             # Это может быть проблемой, если такие действия возможны.
             # В OFC стандартные действия должны быть хешируемыми (кортежи).
        return child_node


    def is_terminal(self) -> bool:
        return self.game_state.is_round_over()

    @staticmethod
    def static_rollout_simulation(initial_game_state: GameState, perspective_player: int = 0) -> Tuple[float, Set[Any]]:
        """
        Статический метод для выполнения симуляции (rollout) из заданного состояния.
        Может быть вызван из параллельного воркера.
        """
        current_rollout_state = initial_game_state.copy()
        simulation_actions_set: Set[Any] = set()
        MAX_ROLLOUT_STEPS = 50 # Ограничение на случай зацикливания
        steps = 0

        while not current_rollout_state.is_round_over() and steps < MAX_ROLLOUT_STEPS:
            steps += 1
            player_to_act_rollout = current_rollout_state.get_player_to_move()

            if player_to_act_rollout != -1:
                # --- Ход игрока ---
                action: Optional[Any] = None
                next_rollout_state: Optional[GameState] = None
                is_fl_turn = current_rollout_state.is_fantasyland_round and current_rollout_state.fantasyland_status[player_to_act_rollout]

                if is_fl_turn:
                    hand = current_rollout_state.fantasyland_hands[player_to_act_rollout]
                    if hand:
                        # Используем статическую версию эвристики
                        placement, discarded = MCTSNode._static_heuristic_fantasyland_placement(hand)
                        try:
                            if placement and discarded is not None:
                                next_rollout_state = current_rollout_state.apply_fantasyland_placement(player_to_act_rollout, placement, discarded)
                            else:
                                next_rollout_state = current_rollout_state.apply_fantasyland_foul(player_to_act_rollout, hand)
                        except Exception as e_fl:
                             print(f"Error applying FL action in static rollout P{player_to_act_rollout}: {e_fl}", file=sys.stderr)
                             next_rollout_state = current_rollout_state.apply_fantasyland_foul(player_to_act_rollout, hand if hand else [])
                    else:
                        print(f"Warning: FL Player {player_to_act_rollout} has no hand in static rollout. Applying foul.", file=sys.stderr)
                        next_rollout_state = current_rollout_state.apply_fantasyland_foul(player_to_act_rollout, [])

                else: # Обычный ход
                    hand = current_rollout_state.current_hands.get(player_to_act_rollout)
                    if hand:
                        possible_moves = current_rollout_state.get_legal_actions_for_player(player_to_act_rollout)
                        if possible_moves:
                            # Используем статическую версию политики
                            action = MCTSNode._static_heuristic_rollout_policy(current_rollout_state, player_to_act_rollout, possible_moves)
                            if action:
                                # Добавляем действие в набор, только если оно хешируемое
                                try:
                                     hash(action)
                                     simulation_actions_set.add(action)
                                except TypeError: pass # Игнорируем нехешируемые

                                try:
                                    next_rollout_state_candidate = current_rollout_state.apply_action(player_to_act_rollout, action)
                                    if next_rollout_state_candidate is current_rollout_state or next_rollout_state_candidate == current_rollout_state:
                                         raise RuntimeError("apply_action returned same state")
                                    next_rollout_state = next_rollout_state_candidate
                                except Exception as e_apply:
                                     print(f"Error applying action in static rollout P{player_to_act_rollout}: {e_apply}", file=sys.stderr)
                                     board = current_rollout_state.boards[player_to_act_rollout]; board.is_foul = True; current_rollout_state._player_finished_round[player_to_act_rollout] = True
                                     if current_rollout_state.current_hands.get(player_to_act_rollout): current_rollout_state.private_discard[player_to_act_rollout].extend(current_rollout_state.current_hands[player_to_act_rollout]); current_rollout_state.current_hands[player_to_act_rollout] = None
                                     next_rollout_state = current_rollout_state
                            else:
                                 print(f"Warning: Static rollout policy returned None for P{player_to_act_rollout}. Applying foul.", file=sys.stderr)
                                 board = current_rollout_state.boards[player_to_act_rollout]; board.is_foul = True; current_rollout_state._player_finished_round[player_to_act_rollout] = True
                                 if current_rollout_state.current_hands.get(player_to_act_rollout): current_rollout_state.private_discard[player_to_act_rollout].extend(current_rollout_state.current_hands[player_to_act_rollout]); current_rollout_state.current_hands[player_to_act_rollout] = None
                                 next_rollout_state = current_rollout_state
                        else:
                             print(f"Warning: No legal actions found for P{player_to_act_rollout} in static rollout. Applying foul.", file=sys.stderr)
                             board = current_rollout_state.boards[player_to_act_rollout]; board.is_foul = True; current_rollout_state._player_finished_round[player_to_act_rollout] = True
                             if current_rollout_state.current_hands.get(player_to_act_rollout): current_rollout_state.private_discard[player_to_act_rollout].extend(current_rollout_state.current_hands[player_to_act_rollout]); current_rollout_state.current_hands[player_to_act_rollout] = None
                             next_rollout_state = current_rollout_state
                    else:
                         print(f"Warning: Player {player_to_act_rollout} has no hand in static rollout. Applying foul.", file=sys.stderr)
                         board = current_rollout_state.boards[player_to_act_rollout]; board.is_foul = True; current_rollout_state._player_finished_round[player_to_act_rollout] = True
                         next_rollout_state = current_rollout_state

                # --- Обновление состояния ПОСЛЕ хода ---
                if next_rollout_state is not None:
                    current_rollout_state = next_rollout_state
                else:
                    print(f"Error: Next state became None after action for P{player_to_act_rollout} in static rollout. Stopping.", file=sys.stderr)
                    break

                # --- Продвигаем состояние (передача хода/улицы, раздача) ПОСЛЕ хода ---
                if not current_rollout_state.is_round_over():
                    try:
                        advanced_state = current_rollout_state.advance_state()
                        # Обновляем состояние только если оно изменилось
                        if advanced_state != current_rollout_state:
                            current_rollout_state = advanced_state
                    except Exception as e_adv:
                        print(f"Error advancing state after P{player_to_act_rollout}'s action in static rollout: {e_adv}", file=sys.stderr)
                        break

            else: # player_to_act_rollout == -1 (никто не может ходить)
                try:
                    advanced_state = current_rollout_state.advance_state()
                    if advanced_state == current_rollout_state:
                        break # Раунд застрял или закончился
                    current_rollout_state = advanced_state
                except Exception as e_adv:
                    print(f"Error advancing state (no player) in static rollout: {e_adv}", file=sys.stderr)
                    break

        # --- Конец симуляции ---
        if steps >= MAX_ROLLOUT_STEPS:
            print(f"Warning: Static rollout reached MAX_ROLLOUT_STEPS ({MAX_ROLLOUT_STEPS}).", file=sys.stderr)

        final_score_p0 = 0.0
        if current_rollout_state.is_round_over():
            try:
                final_score_p0 = current_rollout_state.get_terminal_score()
            except Exception as e_score:
                 print(f"Error getting terminal score in static rollout: {e_score}", file=sys.stderr)
                 final_score_p0 = 0.0 # Считаем 0 при ошибке
        else:
            print(f"Warning: Static rollout ended prematurely (Steps: {steps}, Round Over: {current_rollout_state.is_round_over()}). Returning score 0.", file=sys.stderr)

        reward = float(final_score_p0) if perspective_player == 0 else float(-final_score_p0)
        return reward, simulation_actions_set

    def rollout(self, perspective_player: int = 0) -> Tuple[float, Set[Any]]:
        """ Проводит симуляцию (rollout) до конца раунда (используя статический метод). """
        # Этот метод теперь просто вызывает статическую версию
        return MCTSNode.static_rollout_simulation(self.game_state, perspective_player)

    @staticmethod
    def _static_random_rollout_policy(actions: List[Any]) -> Optional[Any]:
        """ Статическая случайная политика для роллаутов. """
        return random.choice(actions) if actions else None

    @staticmethod
    def _static_heuristic_rollout_policy(state: GameState, player_idx: int, actions: List[Any]) -> Optional[Any]:
        """ Статическая эвристическая политика для роллаутов. """
        if not actions: return None
        # Используем случайную политику по умолчанию
        return MCTSNode._static_random_rollout_policy(actions)

    @staticmethod
    def _static_heuristic_fantasyland_placement(hand: List[int]) -> Tuple[Optional[Dict[str, List[int]]], Optional[List[int]]]:
        """ Статическая эвристика для размещения Фантазии в роллаутах. """
        try:
            n_cards = len(hand)
            n_place = 13
            if not (14 <= n_cards <= 17): return None, None

            n_discard = n_cards - n_place
            # Сбрасываем самые младшие карты
            sorted_hand = sorted(hand, key=lambda c: CardUtils.get_rank_int(c))
            discarded = sorted_hand[:n_discard]
            remaining = sorted_hand[n_discard:]

            if len(remaining) != 13: return None, discarded

            # Простое безопасное размещение
            sorted_rem = sorted(remaining, key=lambda c: CardUtils.get_rank_int(c), reverse=True)
            placement = {
                'bottom': sorted_rem[0:5],
                'middle': sorted_rem[5:10],
                'top': sorted_rem[10:13]
            }
            # Проверяем на фол
            if check_board_foul(placement['top'], placement['middle'], placement['bottom']):
                 placement_swapped = {
                      'bottom': placement['middle'],
                      'middle': placement['bottom'],
                      'top': placement['top']
                 }
                 if check_board_foul(placement_swapped['top'], placement_swapped['middle'], placement_swapped['bottom']):
                      return None, discarded
                 else:
                      return placement_swapped, discarded
            else:
                return placement, discarded
        except Exception as e:
            print(f"Error in static heuristic FL placement: {e}", file=sys.stderr)
            default_discard = hand[13:] if len(hand) > 13 else []
            return None, default_discard

    # (Методы get_q_value, get_rave_q_value, uct_select_child, __repr__ без изменений)
    def get_q_value(self, perspective_player: int) -> float:
        """Возвращает Q-значение узла с точки зрения указанного игрока."""
        if self.visits == 0:
            return 0.0
        raw_q = self.total_reward / self.visits
        player_who_acted = self.parent._get_player_to_move() if self.parent else -1

        if player_who_acted == perspective_player:
            return raw_q
        elif player_who_acted != -1:
            return -raw_q
        else:
            return raw_q if perspective_player == 0 else -raw_q

    def get_rave_q_value(self, action: Any, perspective_player: int) -> float:
        """Возвращает RAVE Q-значение для действия с точки зрения указанного игрока."""
        # Проверяем хешируемость действия перед доступом к словарю
        try: hash(action)
        except TypeError: return 0.0 # Нехешируемое действие не имеет RAVE

        rave_visits = self.rave_visits.get(action, 0)
        if rave_visits == 0:
            return 0.0
        rave_reward = self.rave_total_reward.get(action, 0.0)
        raw_rave_q = rave_reward / rave_visits
        player_to_move_in_current_node = self._get_player_to_move()

        if player_to_move_in_current_node == -1:
            return 0.0
        return raw_rave_q if player_to_move_in_current_node == perspective_player else -raw_rave_q

    def uct_select_child(self, exploration_constant: float, rave_k: float) -> Optional['MCTSNode']:
        """Выбирает дочерний узел с использованием формулы UCT (с RAVE)."""
        best_score = -float('inf')
        best_child = None
        current_player_perspective = self._get_player_to_move()

        if current_player_perspective == -1:
            return None

        parent_visits_log = math.log(self.visits + 1e-6)
        children_items = list(self.children.items())
        if not children_items:
             return None

        random.shuffle(children_items)

        for action, child in children_items:
            child_visits = child.visits
            rave_visits = 0
            score = -float('inf')
            is_hashable = True
            try: hash(action)
            except TypeError: is_hashable = False

            if is_hashable:
                rave_visits = self.rave_visits.get(action, 0)

            if child_visits == 0:
                if rave_visits > 0 and rave_k > 0 and is_hashable:
                    rave_q = self.get_rave_q_value(action, current_player_perspective)
                    explore_rave = exploration_constant * math.sqrt(parent_visits_log / (rave_visits + 1e-6))
                    score = rave_q + explore_rave
                else:
                    score = float('inf')
            else:
                q_child = child.get_q_value(current_player_perspective)
                exploit_term = q_child
                explore_term = exploration_constant * math.sqrt(parent_visits_log / child_visits)
                ucb1_score = exploit_term + explore_term

                if rave_visits > 0 and rave_k > 0 and is_hashable:
                    rave_q = self.get_rave_q_value(action, current_player_perspective)
                    beta = math.sqrt(rave_k / (3 * self.visits + rave_k)) if self.visits > 0 else 1.0
                    score = (1.0 - beta) * ucb1_score + beta * rave_q
                else:
                    score = ucb1_score

            if score > best_score:
                best_score = score
                best_child = child
            elif score == best_score and score != float('inf') and score != -float('inf'):
                 if random.choice([True, False]):
                      best_child = child

        if best_child is None and children_items:
             best_child = random.choice([child for _, child in children_items])

        return best_child

    def __repr__(self):
        """Строковое представление узла для отладки."""
        player_idx = self._get_player_to_move()
        player = f'P{player_idx}' if player_idx != -1 else 'T' # T for Terminal
        q_val_p0 = self.get_q_value(0) # Q-value с точки зрения P0
        action_repr = "Root"
        if self.action:
            try:
                action_str = str(self.action)
                action_repr = (action_str[:25] + '...') if len(action_str) > 28 else action_str
            except Exception:
                action_repr = "???"

        return (
            f"[{player} Act:{action_repr} V={self.visits} Q0={q_val_p0:.2f} "
            f"N_Child={len(self.children)} U_Act={len(self.untried_actions or [])}]"
        )
