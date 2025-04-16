# src/mcts_node.py
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
                 except TypeError: continue # Пропускаем нехешируемые
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

    # (Метод rollout с изменениями в вызове _heuristic_rollout_policy)
    def rollout(self, perspective_player: int = 0) -> Tuple[float, Set[Any]]:
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
                        if placement and discarded is not None: next_rollout_state = current_rollout_state.apply_fantasyland_placement(player_to_act_rollout, placement, discarded)
                        else: next_rollout_state = current_rollout_state.apply_fantasyland_foul(player_to_act_rollout, hand)
                else:
                    hand = current_rollout_state.current_hands.get(player_to_act_rollout)
                    if hand:
                        possible_moves = current_rollout_state.get_legal_actions_for_player(player_to_act_rollout)
                        if possible_moves:
                            # --- ИЗМЕНЕНИЕ: Используем простую случайную политику ---
                            action = self._random_rollout_policy(possible_moves)
                            # action = self._heuristic_rollout_policy(current_rollout_state, player_to_act_rollout, possible_moves) # Старая строка
                            if action:
                                simulation_actions_set.add(action)
                                next_rollout_state_candidate = current_rollout_state.apply_action(player_to_act_rollout, action)
                                if next_rollout_state_candidate is current_rollout_state or next_rollout_state_candidate == current_rollout_state:
                                     print(f"Warning: Rollout apply_action same state P{player_to_act_rollout}.", file=sys.stderr)
                                     board = current_rollout_state.boards[player_to_act_rollout]; board.is_foul = True; current_rollout_state._player_finished_round[player_to_act_rollout] = True
                                     if current_rollout_state.current_hands.get(player_to_act_rollout): current_rollout_state.private_discard[player_to_act_rollout].extend(current_rollout_state.current_hands[player_to_act_rollout]); current_rollout_state.current_hands[player_to_act_rollout] = None
                                     next_rollout_state = current_rollout_state
                                else: next_rollout_state = next_rollout_state_candidate
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
                    advanced_state = current_rollout_state.advance_state()
                    if advanced_state != current_rollout_state: current_rollout_state = advanced_state
            else:
                advanced_state = current_rollout_state.advance_state()
                if advanced_state == current_rollout_state: break
                current_rollout_state = advanced_state
        if steps >= MAX_ROLLOUT_STEPS: pass # print("Warning: Rollout reached max steps.", file=sys.stderr)
        final_score_p0 = current_rollout_state.get_terminal_score()
        reward = float(final_score_p0) if perspective_player == 0 else float(-final_score_p0)
        return reward, simulation_actions_set

    # --- ИЗМЕНЕНИЕ: Добавлена простая случайная политика ---
    def _random_rollout_policy(self, actions: List[Any]) -> Optional[Any]:
        """Простая политика: выбирает случайное легальное действие."""
        if not actions:
            return None
        return random.choice(actions)

    # (Метод _heuristic_rollout_policy остается, но не используется в rollout по умолчанию)
    def _heuristic_rollout_policy(self, state: GameState, player_idx: int, actions: List[Any]) -> Optional[Any]:
        """Эвристическая политика для выбора хода в симуляции (старая версия)."""
        if not actions: return None
        if state.street == 1: return random.choice(actions)
        else:
            hand = state.current_hands.get(player_idx)
            if not hand or len(hand) != 3: return random.choice(actions)
            best_action = None; best_score = -float('inf')
            actions_sample = random.sample(actions, min(len(actions), 20))
            for action in actions_sample:
                place1, place2, discarded_int = action; score = 0
                try: discard_rank_idx = CardUtils.get_rank_int(discarded_int); score -= discard_rank_idx * 0.1
                except Exception: pass
                def simple_placement_bonus(card_int, row_name):
                     try: rank_idx = CardUtils.get_rank_int(card_int); return rank_idx if row_name == 'bottom' else -rank_idx if row_name == 'top' else 0
                     except: return 0
                score += simple_placement_bonus(place1[0], place1[1]); score += simple_placement_bonus(place2[0], place2[1])
                score += random.uniform(-0.1, 0.1)
                if score > best_score: best_score = score; best_action = action
            return best_action if best_action else random.choice(actions)

    # (Метод _heuristic_fantasyland_placement без изменений)
    def _heuristic_fantasyland_placement(self, hand: List[int]) -> Tuple[Optional[Dict[str, List[int]]], Optional[List[int]]]:
        try:
            n_cards = len(hand); n_place = 13
            if not (14 <= n_cards <= 17): return None, None
            n_discard = n_cards - n_place
            sorted_hand = sorted(hand, key=lambda c: CardUtils.get_rank_int(c))
            discarded_list = sorted_hand[:n_discard]; remaining = sorted_hand[n_discard:]
            if len(remaining) != 13: return None, discarded_list
            sorted_remaining = sorted(remaining, key=lambda c: CardUtils.get_rank_int(c), reverse=True)
            placement = {'bottom': sorted_remaining[0:5], 'middle': sorted_remaining[5:10], 'top': sorted_remaining[10:13]}
            if check_board_foul(placement['top'], placement['middle'], placement['bottom']): return None, discarded_list
            else: return placement, discarded_list
        except Exception as e:
            print(f"Error in heuristic FL placement: {e}", file=sys.stderr)
            default_discard = hand[13:] if len(hand) > 13 else []
            return None, default_discard

    # (Методы get_q_value, get_rave_q_value, uct_select_child, __repr__ без изменений)
    def get_q_value(self, perspective_player: int) -> float:
        if self.visits == 0: return 0.0
        raw_q = self.total_reward / self.visits
        if self.parent: player_who_acted = self.parent._get_player_to_move()
        else: player_who_acted = -1
        if player_who_acted == perspective_player: return raw_q
        elif player_who_acted != -1: return -raw_q
        else: return raw_q if perspective_player == 0 else -raw_q

    def get_rave_q_value(self, action: Any, perspective_player: int) -> float:
        rave_visits = self.rave_visits.get(action, 0)
        if rave_visits == 0: return 0.0
        rave_reward = self.rave_total_reward.get(action, 0.0); raw_rave_q = rave_reward / rave_visits
        player_to_move = self._get_player_to_move()
        if player_to_move == -1: return 0.0
        if player_to_move == perspective_player: return raw_rave_q
        else: return -raw_rave_q

    def uct_select_child(self, exploration_constant: float, rave_k: float) -> Optional['MCTSNode']:
        best_score = -float('inf'); best_child = None
        current_player_perspective = self._get_player_to_move()
        if current_player_perspective == -1: return None
        parent_visits_log = math.log(self.visits + 1e-6)
        children_items = list(self.children.items())
        if not children_items: return None
        for action, child in children_items:
            child_visits = child.visits; rave_visits = 0
            try: hash(action); rave_visits = self.rave_visits.get(action, 0) # Получаем RAVE только для хешируемых
            except TypeError: pass
            score = -float('inf')
            if child_visits == 0:
                if rave_visits > 0 and rave_k > 0:
                    rave_q = self.get_rave_q_value(action, current_player_perspective)
                    explore_rave = exploration_constant * math.sqrt(parent_visits_log / (rave_visits + 1e-6)); score = rave_q + explore_rave
                else: score = float('inf')
            else:
                q_child = child.get_q_value(current_player_perspective); exploit_term = q_child
                explore_term = exploration_constant * math.sqrt(parent_visits_log / child_visits); ucb1_score = exploit_term + explore_term
                if rave_visits > 0 and rave_k > 0:
                    rave_q = self.get_rave_q_value(action, current_player_perspective)
                    beta = math.sqrt(rave_k / (3 * self.visits + rave_k)) if self.visits > 0 else 1.0; score = (1.0 - beta) * ucb1_score + beta * rave_q
                else: score = ucb1_score
            if score > best_score: best_score = score; best_child = child
            elif score == best_score and score != float('inf') and score != -float('inf'):
                 if random.choice([True, False]): best_child = child
        if best_child is None and children_items: best_child = random.choice([child for _, child in children_items])
        return best_child

    def __repr__(self):
        player_idx = self._get_player_to_move(); player = f'P{player_idx}' if player_idx != -1 else 'T'
        q_val_p0 = self.get_q_value(0); action_repr = "Root"
        if self.action:
             try: action_str = str(self.action); action_repr = (action_str[:25] + '...') if len(action_str) > 28 else action_str
             except: action_repr = "???"
        return (f"[{player} Act:{action_repr} V={self.visits} Q0={q_val_p0:.2f} "
                f"N_Child={len(self.children)} U_Act={len(self.untried_actions or [])}]")
