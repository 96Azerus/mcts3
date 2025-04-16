# src/mcts_agent.py
"""
Реализация MCTS-агента для игры OFC Pineapple.
Поддерживает RAVE, параллелизацию и Progressive Fantasyland.
"""

import math
import time
import random
import multiprocessing
import traceback
import sys
from typing import Optional, Any, List, Tuple, Set, Dict

# Импорты из src пакета
from src.mcts_node import MCTSNode
from src.game_state import GameState
from src.fantasyland_solver import FantasylandSolver
from src.card import card_to_str, Card as CardUtils # Используем алиас для утилит

# Функция-воркер для параллельного роллаута
def run_parallel_rollout(node_state_dict: dict) -> Tuple[float, Set[Any]]:
    """ Выполняет один роллаут из заданного состояния в отдельном процессе. """
    try:
        from src.game_state import GameState as WorkerGameState
        game_state = WorkerGameState.from_dict(node_state_dict)
        if game_state.is_round_over():
            score_p0 = game_state.get_terminal_score()
            return float(score_p0), set()
        from src.mcts_node import MCTSNode as WorkerMCTSNode
        temp_node = WorkerMCTSNode(game_state)
        reward, sim_actions = temp_node.rollout(perspective_player=0)
        return reward, sim_actions
    except Exception as e:
        print(f"Error in parallel rollout worker: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 0.0, set()


class MCTSAgent:
    """ Агент MCTS для OFC Pineapple. """
    DEFAULT_EXPLORATION: float = 1.414
    DEFAULT_RAVE_K: float = 500
    DEFAULT_TIME_LIMIT_MS: int = 5000
    DEFAULT_NUM_WORKERS: int = max(1, multiprocessing.cpu_count() - 1 if multiprocessing.cpu_count() > 1 else 1)
    DEFAULT_ROLLOUTS_PER_LEAF: int = 4

    def __init__(self,
                 exploration: Optional[float] = None,
                 rave_k: Optional[float] = None,
                 time_limit_ms: Optional[int] = None,
                 num_workers: Optional[int] = None,
                 rollouts_per_leaf: Optional[int] = None):
        """ Инициализирует MCTS-агента. """
        self.exploration: float = exploration if exploration is not None else self.DEFAULT_EXPLORATION
        self.rave_k: float = rave_k if rave_k is not None else self.DEFAULT_RAVE_K
        time_limit_val: int = time_limit_ms if time_limit_ms is not None else self.DEFAULT_TIME_LIMIT_MS
        self.time_limit: float = time_limit_val / 1000.0
        max_cpus = multiprocessing.cpu_count()
        requested_workers: int = num_workers if num_workers is not None else self.DEFAULT_NUM_WORKERS
        self.num_workers: int = max(1, min(requested_workers, max_cpus))
        self.rollouts_per_leaf: int = rollouts_per_leaf if rollouts_per_leaf is not None else self.DEFAULT_ROLLOUTS_PER_LEAF
        if self.num_workers == 1 and self.rollouts_per_leaf > 1:
            print(f"Warning: num_workers=1, reducing rollouts_per_leaf from {self.rollouts_per_leaf} to 1.")
            self.rollouts_per_leaf = 1
        self.fantasyland_solver = FantasylandSolver()
        print(f"MCTS Agent initialized: TimeLimit={self.time_limit:.2f}s, Exploration={self.exploration}, "
              f"RaveK={self.rave_k}, Workers={self.num_workers}, RolloutsPerLeaf={self.rollouts_per_leaf}")
        try:
            current_method = multiprocessing.get_start_method(allow_none=True)
            if current_method != 'spawn':
                multiprocessing.set_start_method('spawn', force=True)
                print(f"Multiprocessing start method set to 'spawn'.")
        except Exception as e:
            print(f"Warning: Could not set multiprocessing start method to 'spawn': {e}. Using default ({multiprocessing.get_start_method()}).")
            sys.stdout.flush()


    def choose_action(self, game_state: GameState) -> Optional[Any]:
        """ Выбирает лучшее действие с помощью MCTS. """
        start_time_total = time.time()
        print(f"\n--- AI Agent: Choosing action (Street {game_state.street}) ---")
        sys.stdout.flush()

        player_to_act = game_state.get_player_to_move() # Используем метод GameState

        if player_to_act == -1:
            print("AI Agent Error: No player can act. Returning None.")
            sys.stdout.flush()
            return None

        print(f"Player to act: {player_to_act}")
        sys.stdout.flush()

        # --- Обработка Fantasyland ---
        if game_state.is_fantasyland_round and game_state.fantasyland_status[player_to_act]:
            hand = game_state.fantasyland_hands[player_to_act]
            if hand:
                print(f"Player {player_to_act} is in Fantasyland. Solving...")
                sys.stdout.flush(); start_fl_time = time.time()
                placement, discarded = self.fantasyland_solver.solve(hand)
                solve_time = time.time() - start_fl_time; print(f"Fantasyland solved in {solve_time:.3f}s"); sys.stdout.flush()
                if placement and discarded is not None: return ("FANTASYLAND_PLACEMENT", placement, discarded)
                else: print("Warning: Fantasyland solver failed. Returning foul action."); sys.stdout.flush(); return ("FANTASYLAND_FOUL", hand)
            else: print(f"AI Agent Error: FL player {player_to_act} has no hand."); sys.stdout.flush(); return None

        # --- Обычный ход (MCTS) ---
        print(f"Starting MCTS for player {player_to_act}...")
        sys.stdout.flush()
        initial_actions = game_state.get_legal_actions_for_player(player_to_act)
        print(f"Found {len(initial_actions)} legal actions initially.")
        sys.stdout.flush()

        if not initial_actions: print(f"AI Agent Warning: No legal actions found for player {player_to_act}."); sys.stdout.flush(); return None
        if len(initial_actions) == 1: print("Only one legal action found."); sys.stdout.flush(); return initial_actions[0]

        root_node = MCTSNode(game_state)
        root_node.untried_actions = list(initial_actions)
        random.shuffle(root_node.untried_actions)
        # Инициализация RAVE статистики
        for act in root_node.untried_actions:
             # --- ИСПРАВЛЕНО: Проверка хешируемости перед добавлением в словарь ---
             try:
                  hash(act) # Проверяем, можно ли хешировать действие
                  if act not in root_node.rave_visits:
                       root_node.rave_visits[act] = 0
                       root_node.rave_total_reward[act] = 0.0
             except TypeError:
                  print(f"Warning: Action {self._format_action(act)} is not hashable and cannot be used in RAVE.", file=sys.stderr)


        start_mcts_time = time.time(); num_simulations = 0
        try:
            with multiprocessing.Pool(processes=self.num_workers) as pool:
                while time.time() - start_mcts_time < self.time_limit:
                    path, leaf_node = self._select(root_node)
                    if leaf_node is None: continue
                    node_to_rollout_from = leaf_node; expanded_node = None
                    if not leaf_node.is_terminal():
                        if leaf_node.untried_actions:
                            expanded_node = leaf_node.expand()
                            if expanded_node: node_to_rollout_from = expanded_node; path.append(expanded_node)
                    results = []; simulation_actions_aggregated: Set[Any] = set()
                    if not node_to_rollout_from.is_terminal():
                        try: node_state_dict = node_to_rollout_from.game_state.to_dict()
                        except Exception as e: print(f"Error serializing state: {e}", file=sys.stderr); sys.stderr.flush(); continue
                        async_results = [pool.apply_async(run_parallel_rollout, (node_state_dict,)) for _ in range(self.rollouts_per_leaf)]
                        for res in async_results:
                            try:
                                timeout_get = max(0.1, self.time_limit * 0.1)
                                reward, sim_actions = res.get(timeout=timeout_get)
                                results.append(reward); simulation_actions_aggregated.update(sim_actions); num_simulations += 1
                            except multiprocessing.TimeoutError: print("Warning: Rollout worker timed out.", file=sys.stderr); sys.stderr.flush()
                            except Exception as e: print(f"Warning: Error getting result from worker: {e}", file=sys.stderr); sys.stderr.flush()
                    else: reward = node_to_rollout_from.game_state.get_terminal_score(); results.append(reward); num_simulations += 1
                    if results:
                        total_reward = sum(results); num_rollouts = len(results)
                        if expanded_node and expanded_node.action: simulation_actions_aggregated.add(expanded_node.action)
                        self._backpropagate_parallel(path, total_reward, num_rollouts, simulation_actions_aggregated)
        except Exception as e:
            print(f"Error during MCTS parallel execution: {e}", file=sys.stderr); traceback.print_exc(file=sys.stderr); sys.stderr.flush()
            print("Choosing random action due to MCTS error."); return random.choice(initial_actions) if initial_actions else None

        elapsed_time = time.time() - start_mcts_time
        sims_per_sec = (num_simulations / elapsed_time) if elapsed_time > 0 else 0
        print(f"MCTS finished: Ran {num_simulations} simulations in {elapsed_time:.3f}s ({sims_per_sec:.1f} sims/s).")
        sys.stdout.flush()

        best_action = self._select_best_action(root_node, initial_actions)
        total_time = time.time() - start_time_total
        print(f"--- AI Agent: Action chosen in {total_time:.3f}s ---")
        sys.stdout.flush()
        return best_action

    # (Методы _select, _backpropagate_parallel, _select_best_action, _format_action без изменений)
    def _select(self, node: MCTSNode) -> Tuple[List[MCTSNode], Optional[MCTSNode]]:
        path = [node]; current_node = node
        while True:
            if current_node.is_terminal(): return path, current_node
            player_to_move = current_node._get_player_to_move()
            if player_to_move == -1: return path, current_node
            if current_node.untried_actions is None:
                current_node.untried_actions = current_node.game_state.get_legal_actions_for_player(player_to_move)
                random.shuffle(current_node.untried_actions)
                for act in current_node.untried_actions:
                     try: # Проверка хешируемости
                          hash(act)
                          if act not in current_node.rave_visits: current_node.rave_visits[act] = 0; current_node.rave_total_reward[act] = 0.0
                     except TypeError: pass # Игнорируем нехешируемые для RAVE
            if current_node.untried_actions: return path, current_node
            if not current_node.children: return path, current_node
            selected_child = current_node.uct_select_child(self.exploration, self.rave_k)
            if selected_child is None:
                if current_node.children:
                     try: selected_child = random.choice(list(current_node.children.values()))
                     except IndexError: return path, current_node
                else: return path, current_node
            current_node = selected_child; path.append(current_node)

    def _backpropagate_parallel(self, path: List[MCTSNode], total_reward: float, num_rollouts: int, simulation_actions: Set[Any]):
        if num_rollouts == 0: return
        for node in reversed(path):
            node.visits += num_rollouts; node.total_reward += total_reward
            player_to_move_from_node = node._get_player_to_move()
            if player_to_move_from_node != -1:
                possible_actions = set(node.children.keys())
                if node.untried_actions: possible_actions.update(node.untried_actions)
                relevant_sim_actions = simulation_actions.intersection(possible_actions)
                for action in relevant_sim_actions:
                     try: # Проверка хешируемости
                          hash(action)
                          if action not in node.rave_visits: node.rave_visits[action] = 0; node.rave_total_reward[action] = 0.0
                          node.rave_visits[action] += num_rollouts; node.rave_total_reward[action] += total_reward
                     except TypeError: pass # Игнорируем нехешируемые для RAVE

    def _select_best_action(self, root_node: MCTSNode, initial_actions: List[Any]) -> Optional[Any]:
        if not root_node.children: print("Warning: No children found. Choosing random."); sys.stdout.flush(); return random.choice(initial_actions) if initial_actions else None
        best_action = None; max_visits = -1; items = list(root_node.children.items()); random.shuffle(items)
        print(f"--- Evaluating {len(items)} child nodes ---")
        for action, child_node in items:
             q_val_p0 = child_node.get_q_value(0); rave_q_p0 = 0.0
             try: # Проверка хешируемости для RAVE
                  hash(action)
                  if root_node.rave_visits.get(action, 0) > 0: rave_q_p0 = root_node.get_rave_q_value(action, 0)
             except TypeError: pass
             print(f"  Action: {self._format_action(action):<40} Visits: {child_node.visits:<6} Q(P0): {q_val_p0:<8.3f} RAVE_Q(P0): {rave_q_p0:<8.3f} RAVE_V: {root_node.rave_visits.get(action, 0):<5}")
             if child_node.visits > max_visits: max_visits = child_node.visits; best_action = action
        if best_action is None: print("Warning: Could not determine best action. Choosing first shuffled."); sys.stdout.flush(); best_action = items[0][0] if items else (random.choice(initial_actions) if initial_actions else None)
        if best_action: print(f"Selected action (max visits = {max_visits}): {self._format_action(best_action)}")
        else: print("Error: Failed to select any action.")
        sys.stdout.flush(); return best_action

    def _format_action(self, action: Any) -> str:
        if action is None: return "None"
        try:
            if isinstance(action, tuple) and len(action) == 3 and isinstance(action[0], tuple) and len(action[0]) == 3 and isinstance(action[0][0], int) and isinstance(action[1], tuple) and len(action[1]) == 3 and isinstance(action[1][0], int) and isinstance(action[2], int):
                p1, p2, d = action; return f"PINEAPPLE: {card_to_str(p1[0])}@{p1[1]}{p1[2]}, {card_to_str(p2[0])}@{p2[1]}{p2[2]}; DISC={card_to_str(d)}"
            # --- ИСПРАВЛЕНО: Проверяем tuple для улицы 1 ---
            elif isinstance(action, tuple) and len(action) == 2 and isinstance(action[0], tuple) and action[0] and isinstance(action[0][0], tuple) and len(action[0][0]) == 3 and isinstance(action[0][0][0], int):
                 placements_str = ", ".join([f"{card_to_str(c)}@{r}{i}" for c, r, i in action[0]]); return f"STREET 1: Place [{placements_str}]"
            # --- ИСПРАВЛЕНО: Проверяем tuple для ФЛ входа ---
            elif isinstance(action, tuple) and action[0] == "FANTASYLAND_INPUT" and isinstance(action[1], tuple):
                 return f"FANTASYLAND_INPUT ({len(action[1])} cards)"
            elif isinstance(action, tuple) and action[0] == "FANTASYLAND_PLACEMENT": discard_count = len(action[2]) if isinstance(action[2], list) else '?'; return f"FANTASYLAND_PLACE (Discard {discard_count})"
            elif isinstance(action, tuple) and action[0] == "FANTASYLAND_FOUL": discard_count = len(action[1]) if isinstance(action[1], list) else '?'; return f"FANTASYLAND_FOUL (Discard {discard_count})"
            elif isinstance(action, tuple): formatted = [repr(i) if isinstance(i, (str, int, float, bool, type(None))) else "[...]" if isinstance(i, list) else "{...}" if isinstance(i, dict) else self._format_action(i) for i in action]; return f"Tuple Action: ({', '.join(formatted)})"
            else: return str(action)
        except Exception: return "ErrorFormattingAction"
