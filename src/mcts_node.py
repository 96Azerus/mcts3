# src/mcts_node.py v1.3
"""
Представление узла дерева MCTS для OFC Pineapple.
Содержит состояние игры, статистику посещений/наград и логику MCTS (выбор, расширение, симуляция).
Использует GameState.advance_state() для симуляции.
"""

import math
import random
import traceback
import sys
from typing import Optional, Dict, Any, List, Tuple, Set
from collections import Counter

# Импорты из src пакета
from src.game_state import GameState
from src.card import Card as CardUtils, card_to_str, RANK_MAP, STR_RANKS, INVALID_CARD
from src.scoring import (
    RANK_CLASS_QUADS, RANK_CLASS_TRIPS, get_hand_rank_safe,
    check_board_foul, get_row_royalty, RANK_CLASS_PAIR,
    RANK_CLASS_HIGH_CARD
)
from src.fantasyland_solver import FantasylandSolver
from itertools import combinations


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
            if isinstance(action, tuple) and action[0] == "FANTASYLAND_INPUT": return None
            else: next_state = self.game_state.apply_action(player_to_move, action)
            if next_state is self.game_state or next_state.get_state_representation() == original_state_repr:
                print(f"Error: apply_action same state P{player_to_move}.", file=sys.stderr); print(f"Action: {action}", file=sys.stderr); return self.expand() if self.untried_actions else None
        except Exception as e: print(f"Error applying action expand P{player_to_move}: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr); return self.expand() if self.untried_actions else None
        child_node = MCTSNode(next_state, parent=self, action=action); self.children[action] = child_node; return child_node

    def is_terminal(self) -> bool:
        return self.game_state.is_round_over()

    def rollout(self, perspective_player: int = 0) -> Tuple[float, Set[Any]]:
        """ Проводит симуляцию (rollout) до конца раунда. """
        current_rollout_state = self.game_state.copy()
        simulation_actions_set: Set[Any] = set()
        MAX_ROLLOUT_STEPS = 50
        steps = 0
        while not current_rollout_state.is_round_over() and steps < MAX_ROLLOUT_STEPS:
            steps += 1
            player_to_act_rollout = current_rollout_state.get_player_to_move()
            if player_to_act_rollout != -1:
                action: Optional[Any] = None; next_rollout_state: Optional[GameState] = None
                is_fl_turn = current_rollout_state.is_fantasyland_round and current_rollout_state.fantasyland_status[player_to_act_rollout]
                if is_fl_turn:
                    hand = current_rollout_state.fantasyland_hands[player_to_act_rollout]
                    if hand:
                        placement, discarded = self._heuristic_fantasyland_placement(hand)
                        try: # --- ИЗМЕНЕНИЕ: Добавляем try-except вокруг apply ---
                            if placement and discarded is not None: next_rollout_state = current_rollout_state.apply_fantasyland_placement(player_to_act_rollout, placement, discarded)
                            else: next_rollout_state = current_rollout_state.apply_fantasyland_foul(player_to_act_rollout, hand)
                        except Exception as e_fl:
                             print(f"Error applying FL action in rollout P{player_to_act_rollout}: {e_fl}", file=sys.stderr)
                             # Считаем фолом при любой ошибке
                             next_rollout_state = current_rollout_state.apply_fantasyland_foul(player_to_act_rollout, hand if hand else [])
                else:
                    hand = current_rollout_state.current_hands.get(player_to_act_rollout)
                    if hand:
                        possible_moves = current_rollout_state.get_legal_actions_for_player(player_to_act_rollout)
                        if possible_moves:
                            action = self._heuristic_rollout_policy(current_rollout_state, player_to_act_rollout, possible_moves) # Используем эвристику
                            if action:
                                simulation_actions_set.add(action)
                                try: # --- ИЗМЕНЕНИЕ: Добавляем try-except вокруг apply ---
                                    next_rollout_state_candidate = current_rollout_state.apply_action(player_to_act_rollout, action)
                                    if next_rollout_state_candidate is current_rollout_state or next_rollout_state_candidate == current_rollout_state:
                                         raise RuntimeError("apply_action returned same state") # Генерируем ошибку
                                    next_rollout_state = next_rollout_state_candidate
                                except Exception as e_apply:
                                     print(f"Error applying action in rollout P{player_to_act_rollout}: {e_apply}", file=sys.stderr)
                                     # Считаем фолом при ошибке
                                     board = current_rollout_state.boards[player_to_act_rollout]; board.is_foul = True; current_rollout_state._player_finished_round[player_to_act_rollout] = True
                                     if current_rollout_state.current_hands.get(player_to_act_rollout): current_rollout_state.private_discard[player_to_act_rollout].extend(current_rollout_state.current_hands[player_to_act_rollout]); current_rollout_state.current_hands[player_to_act_rollout] = None
                                     next_rollout_state = current_rollout_state # Остаемся в измененном состоянии (с фолом)
                            else: # Фол, если эвристика не выбрала ход
                                 board = current_rollout_state.boards[player_to_act_rollout]; board.is_foul = True; current_rollout_state._player_finished_round[player_to_act_rollout] = True
                                 if current_rollout_state.current_hands.get(player_to_act_rollout): current_rollout_state.private_discard[player_to_act_rollout].extend(current_rollout_state.current_hands[player_to_act_rollout]); current_rollout_state.current_hands[player_to_act_rollout] = None
                                 next_rollout_state = current_rollout_state
                        else: # Фол, если нет легальных ходов
                             board = current_rollout_state.boards[player_to_act_rollout]; board.is_foul = True; current_rollout_state._player_finished_round[player_to_act_rollout] = True
                             if current_rollout_state.current_hands.get(player_to_act_rollout): current_rollout_state.private_discard[player_to_act_rollout].extend(current_rollout_state.current_hands[player_to_act_rollout]); current_rollout_state.current_hands[player_to_act_rollout] = None
                             next_rollout_state = current_rollout_state
                if next_rollout_state is not None and next_rollout_state != current_rollout_state: current_rollout_state = next_rollout_state
                elif next_rollout_state is None and is_fl_turn: print(f"Warning: FL action P{player_to_act_rollout} resulted in None state.", file=sys.stderr); break
                if not current_rollout_state.is_round_over():
                    try: # --- ИЗМЕНЕНИЕ: Добавляем try-except вокруг advance ---
                         advanced_state = current_rollout_state.advance_state()
                         if advanced_state != current_rollout_state: current_rollout_state = advanced_state
                    except Exception as e_adv:
                         print(f"Error advancing state in rollout: {e_adv}", file=sys.stderr)
                         break # Прерываем симуляцию при ошибке advance
            else:
                try: # --- ИЗМЕНЕНИЕ: Добавляем try-except вокруг advance ---
                     advanced_state = current_rollout_state.advance_state()
                     if advanced_state == current_rollout_state: break
                     current_rollout_state = advanced_state
                except Exception as e_adv:
                     print(f"Error advancing state (no player) in rollout: {e_adv}", file=sys.stderr)
                     break
        if steps >= MAX_ROLLOUT_STEPS: pass
        final_score_p0 = current_rollout_state.get_terminal_score()
        reward = float(final_score_p0) if perspective_player == 0 else float(-final_score_p0)
        return reward, simulation_actions_set

    def _random_rollout_policy(self, actions: List[Any]) -> Optional[Any]:
        """Простая случайная политика для роллаутов."""
        return random.choice(actions) if actions else None

    def _heuristic_rollout_policy(self, state: GameState, player_idx: int, actions: List[Any]) -> Optional[Any]:
        """Политика для роллаутов."""
        if not actions: return None

        # --- ИЗМЕНЕНИЕ: Используем случайную политику для всех улиц ---
        # В будущем здесь можно реализовать более сложную эвристику,
        # но для отладки MCTS и получения ненулевых Q-значений
        # случайная политика часто является хорошей отправной точкой.
        return self._random_rollout_policy(actions)
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

        # Старая эвристика (закомментирована):
        # if state.street == 1:
        #     return random.choice(actions)
        # else:
        #     hand = state.current_hands.get(player_idx)
        #     if not hand or len(hand) != 3:
        #         return random.choice(actions)
        #     best_action = None
        #     best_score = -float('inf')
        #     actions_sample = random.sample(actions, min(len(actions), 20))
        #     for action in actions_sample:
        #         p1, p2, d_int = action
        #         score = 0
        #         try: score -= CardUtils.get_rank_int(d_int) * 0.1
        #         except Exception: pass
        #         def bonus(card_int, row_name):
        #             try:
        #                 rank_val = CardUtils.get_rank_int(card_int)
        #                 if row_name == 'bottom': return rank_val
        #                 elif row_name == 'top': return -rank_val
        #                 else: return 0
        #             except Exception: return 0
        #         score += bonus(p1[0], p1[1])
        #         score += bonus(p2[0], p2[1])
        #         score += random.uniform(-0.1, 0.1)
        #         if score > best_score:
        #             best_score = score
        #             best_action = action
        #     return best_action if best_action else random.choice(actions)

    def _heuristic_fantasyland_placement(self, hand: List[int]) -> Tuple[Optional[Dict[str, List[int]]], Optional[List[int]]]:
        """Простая эвристика для размещения Фантазии в роллаутах."""
        try:
            n_cards = len(hand)
            n_place = 13
            if not (14 <= n_cards <= 17): return None, None # Неверный размер руки

            n_discard = n_cards - n_place
            # Сбрасываем самые младшие карты
            sorted_hand = sorted(hand, key=lambda c: CardUtils.get_rank_int(c))
            discarded = sorted_hand[:n_discard]
            remaining = sorted_hand[n_discard:]

            if len(remaining) != 13: return None, discarded # Ошибка

            # Простое безопасное размещение оставшихся 13 карт
            sorted_rem = sorted(remaining, key=lambda c: CardUtils.get_rank_int(c), reverse=True)
            placement = {
                'bottom': sorted_rem[0:5],
                'middle': sorted_rem[5:10],
                'top': sorted_rem[10:13]
            }
            # Проверяем на фол
            if check_board_foul(placement['top'], placement['middle'], placement['bottom']):
                 # Если фол, пытаемся поменять middle и bottom
                 placement_swapped = {
                      'bottom': placement['middle'],
                      'middle': placement['bottom'],
                      'top': placement['top']
                 }
                 if check_board_foul(placement_swapped['top'], placement_swapped['middle'], placement_swapped['bottom']):
                      return None, discarded # Фол даже после обмена
                 else:
                      return placement_swapped, discarded
            else:
                return placement, discarded
        except Exception as e:
            print(f"Error in heuristic FL placement during rollout: {e}", file=sys.stderr)
            # Возвращаем фол (None для placement) со стандартным сбросом
            default_discard = hand[13:] if len(hand) > 13 else []
            return None, default_discard

    def get_q_value(self, perspective_player: int) -> float:
        """Возвращает Q-значение узла с точки зрения указанного игрока."""
        if self.visits == 0:
            return 0.0
        raw_q = self.total_reward / self.visits
        # Определяем, чей ход привел к этому состоянию (кто ходил в родительском узле)
        player_who_acted = self.parent._get_player_to_move() if self.parent else -1

        # --- ИСПРАВЛЕНИЕ СИНТАКСИСА ---
        if player_who_acted == perspective_player:
            # Если ход делал игрок, с чьей точки зрения мы смотрим, Q-значение прямое
            return raw_q
        elif player_who_acted != -1:
            # Если ход делал другой игрок, инвертируем Q-значение
            return -raw_q
        else:
            # Если player_who_acted == -1 (например, корневой узел или терминальный),
            # возвращаем значение с точки зрения P0 по умолчанию
            return raw_q if perspective_player == 0 else -raw_q
        # --- КОНЕЦ ИСПРАВЛЕНИЯ СИНТАКСИСА ---

    def get_rave_q_value(self, action: Any, perspective_player: int) -> float:
        """Возвращает RAVE Q-значение для действия с точки зрения указанного игрока."""
        rave_visits = self.rave_visits.get(action, 0)
        if rave_visits == 0:
            return 0.0
        rave_reward = self.rave_total_reward.get(action, 0.0)
        raw_rave_q = rave_reward / rave_visits
        # Определяем, чей ход был бы после выполнения этого действия
        # (т.е. чей ход в дочернем узле, если бы он был создан)
        # Это немного сложнее, так как дочерний узел может не существовать.
        # Проще использовать точку зрения игрока, который выбирает действие (т.е. чей ход в ТЕКУЩЕМ узле)
        player_to_move_in_current_node = self._get_player_to_move()

        if player_to_move_in_current_node == -1:
            # Если в текущем узле никто не ходит (терминальный), RAVE не имеет смысла
            return 0.0
        # Возвращаем RAVE Q с точки зрения игрока, который выбирает действие
        return raw_rave_q if player_to_move_in_current_node == perspective_player else -raw_rave_q

    def uct_select_child(self, exploration_constant: float, rave_k: float) -> Optional['MCTSNode']:
        """Выбирает дочерний узел с использованием формулы UCT (с RAVE)."""
        best_score = -float('inf')
        best_child = None
        current_player_perspective = self._get_player_to_move()

        if current_player_perspective == -1:
            return None # Не можем выбирать, если никто не ходит

        # Логарифм посещений родителя (добавляем epsilon для избежания log(0))
        parent_visits_log = math.log(self.visits + 1e-6)
        children_items = list(self.children.items())
        if not children_items:
             return None # Нет дочерних узлов для выбора

        # Перемешиваем для случайного выбора при равных очках
        random.shuffle(children_items)

        for action, child in children_items:
            child_visits = child.visits
            rave_visits = 0
            score = -float('inf') # Начальное значение

            # Проверяем хешируемость действия для RAVE
            is_hashable = True
            try:
                hash(action)
            except TypeError:
                is_hashable = False

            if is_hashable:
                rave_visits = self.rave_visits.get(action, 0)

            if child_visits == 0:
                # Если узел не посещался, используем RAVE (если есть) или даем высокий приоритет
                if rave_visits > 0 and rave_k > 0 and is_hashable:
                    rave_q = self.get_rave_q_value(action, current_player_perspective)
                    # Используем только RAVE Q и exploration для неисследованных узлов
                    explore_rave = exploration_constant * math.sqrt(parent_visits_log / (rave_visits + 1e-6))
                    score = rave_q + explore_rave
                else:
                    # Даем максимальный приоритет неисследованным узлам без RAVE
                    score = float('inf')
            else:
                # Стандартный UCB1
                q_child = child.get_q_value(current_player_perspective)
                exploit_term = q_child
                explore_term = exploration_constant * math.sqrt(parent_visits_log / child_visits)
                ucb1_score = exploit_term + explore_term

                # Комбинируем с RAVE, если он доступен
                if rave_visits > 0 and rave_k > 0 and is_hashable:
                    rave_q = self.get_rave_q_value(action, current_player_perspective)
                    # Формула AMAF/RAVE UCT
                    beta = math.sqrt(rave_k / (3 * self.visits + rave_k)) if self.visits > 0 else 1.0
                    score = (1.0 - beta) * ucb1_score + beta * rave_q
                else:
                    # Используем только UCB1, если RAVE недоступен
                    score = ucb1_score

            # Обновляем лучший узел
            if score > best_score:
                best_score = score
                best_child = child
            # Случайный выбор при равенстве (кроме inf)
            elif score == best_score and score != float('inf') and score != -float('inf'):
                 if random.choice([True, False]):
                      best_child = child

        # Если по какой-то причине лучший узел не выбран (например, все inf), выбираем случайно
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
                # Пытаемся получить короткое строковое представление действия
                action_str = str(self.action)
                action_repr = (action_str[:25] + '...') if len(action_str) > 28 else action_str
            except Exception:
                action_repr = "???" # Если не удалось преобразовать действие в строку

        return (
            f"[{player} Act:{action_repr} V={self.visits} Q0={q_val_p0:.2f} "
            f"N_Child={len(self.children)} U_Act={len(self.untried_actions or [])}]"
        )
