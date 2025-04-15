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

# --- ИСПРАВЛЕНО: Импорты из src пакета ---
from src.mcts_node import MCTSNode
from src.game_state import GameState
from src.fantasyland_solver import FantasylandSolver
from src.card import card_to_str, Card as CardUtils # Используем алиас для утилит

# --- Функция-воркер для параллельного роллаута ---
# Должна быть на верхнем уровне модуля для корректной работы multiprocessing
def run_parallel_rollout(node_state_dict: dict) -> Tuple[float, Set[Any]]:
    """
    Выполняет один роллаут из заданного состояния в отдельном процессе.

    Args:
        node_state_dict: Словарь, представляющий состояние GameState (результат state.to_dict()).

    Returns:
        Кортеж (награда_P0, набор_действий_симуляции).
    """
    try:
        # --- ИСПРАВЛЕНО: Импорт GameState внутри функции ---
        # Это необходимо, так как воркеры могут не наследовать импорты главного процесса
        from src.game_state import GameState as WorkerGameState
        game_state = WorkerGameState.from_dict(node_state_dict)

        # Если состояние уже терминальное, возвращаем счет
        if game_state.is_round_over():
            score_p0 = game_state.get_terminal_score()
            return float(score_p0), set()

        # Создаем временный узел для выполнения роллаута
        # --- ИСПРАВЛЕНО: Импорт MCTSNode внутри функции ---
        from src.mcts_node import MCTSNode as WorkerMCTSNode
        temp_node = WorkerMCTSNode(game_state)
        # Выполняем роллаут с точки зрения игрока 0
        reward, sim_actions = temp_node.rollout(perspective_player=0)
        return reward, sim_actions
    except Exception as e:
        # Логгируем ошибку в воркере
        print(f"Error in parallel rollout worker: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Возвращаем нейтральный результат при ошибке
        return 0.0, set()


class MCTSAgent:
    """
    Агент, использующий поиск по дереву Монте-Карло (MCTS) с RAVE и параллелизацией
    для выбора хода в OFC Pineapple.
    """
    # Значения по умолчанию для параметров MCTS
    DEFAULT_EXPLORATION: float = 1.414 # Константа C в формуле UCT
    DEFAULT_RAVE_K: float = 500      # Параметр для балансировки RAVE и UCB1
    DEFAULT_TIME_LIMIT_MS: int = 5000 # Время на ход в миллисекундах
    # Количество воркеров по умолчанию (макс. CPU - 1, но не менее 1)
    DEFAULT_NUM_WORKERS: int = max(1, multiprocessing.cpu_count() - 1 if multiprocessing.cpu_count() > 1 else 1)
    # Количество роллаутов, запускаемых из одного листового узла за итерацию
    DEFAULT_ROLLOUTS_PER_LEAF: int = 4

    def __init__(self,
                 exploration: Optional[float] = None,
                 rave_k: Optional[float] = None,
                 time_limit_ms: Optional[int] = None,
                 num_workers: Optional[int] = None,
                 rollouts_per_leaf: Optional[int] = None):
        """
        Инициализирует MCTS-агента.

        Args:
            exploration: Константа исследования UCT (C).
            rave_k: Параметр RAVE.
            time_limit_ms: Лимит времени на ход в миллисекундах.
            num_workers: Количество параллельных процессов для роллаутов.
            rollouts_per_leaf: Количество симуляций на листовой узел.
        """
        self.exploration: float = exploration if exploration is not None else self.DEFAULT_EXPLORATION
        self.rave_k: float = rave_k if rave_k is not None else self.DEFAULT_RAVE_K
        time_limit_val: int = time_limit_ms if time_limit_ms is not None else self.DEFAULT_TIME_LIMIT_MS
        self.time_limit: float = time_limit_val / 1000.0 # Конвертируем в секунды

        # Настройка количества воркеров
        max_cpus = multiprocessing.cpu_count()
        requested_workers: int = num_workers if num_workers is not None else self.DEFAULT_NUM_WORKERS
        # Ограничиваем количество воркеров доступными CPU и делаем минимум 1
        self.num_workers: int = max(1, min(requested_workers, max_cpus))

        self.rollouts_per_leaf: int = rollouts_per_leaf if rollouts_per_leaf is not None else self.DEFAULT_ROLLOUTS_PER_LEAF
        # Если воркер один, нет смысла делать много роллаутов на лист
        if self.num_workers == 1 and self.rollouts_per_leaf > 1:
            print(f"Warning: num_workers=1, reducing rollouts_per_leaf from {self.rollouts_per_leaf} to 1.")
            self.rollouts_per_leaf = 1

        # Инициализируем солвер для Фантазии
        self.fantasyland_solver = FantasylandSolver()

        print(f"MCTS Agent initialized: TimeLimit={self.time_limit:.2f}s, Exploration={self.exploration}, "
              f"RaveK={self.rave_k}, Workers={self.num_workers}, RolloutsPerLeaf={self.rollouts_per_leaf}")

        # Устанавливаем метод старта процессов 'spawn' для лучшей совместимости
        try:
            current_method = multiprocessing.get_start_method(allow_none=True)
            # Принудительно устанавливаем 'spawn', если еще не установлен
            if current_method != 'spawn':
                multiprocessing.set_start_method('spawn', force=True)
                print(f"Multiprocessing start method set to 'spawn'.")
        except Exception as e:
            print(f"Warning: Could not set multiprocessing start method to 'spawn': {e}. "
                  f"Using default ({multiprocessing.get_start_method()}). Potential issues may arise.")
            sys.stdout.flush()


    def choose_action(self, game_state: GameState) -> Optional[Any]:
        """
        Выбирает лучшее действие для текущего состояния игры с помощью MCTS.

        Args:
            game_state: Текущее состояние GameState.

        Returns:
            Выбранное действие или None, если действие выбрать не удалось.
        """
        start_time_total = time.time()
        print(f"\n--- AI Agent: Choosing action (Street {game_state.street}) ---")
        sys.stdout.flush()

        # Определяем, какой игрок должен ходить
        # Используем внутренний метод MCTSNode для консистентности
        # TODO: Перенести get_player_to_move в GameState
        temp_node_for_player = MCTSNode(game_state)
        player_to_act = temp_node_for_player._get_player_to_move()

        if player_to_act == -1:
            print("AI Agent Error: No player can act in the current state. Returning None.")
            sys.stdout.flush()
            return None

        print(f"Player to act: {player_to_act}")
        sys.stdout.flush()

        # --- Обработка хода в Fantasyland ---
        if game_state.is_fantasyland_round and game_state.fantasyland_status[player_to_act]:
            hand = game_state.fantasyland_hands[player_to_act]
            if hand:
                print(f"Player {player_to_act} is in Fantasyland. Solving...")
                sys.stdout.flush()
                start_fl_time = time.time()
                # Вызываем солвер Фантазии
                placement, discarded = self.fantasyland_solver.solve(hand)
                solve_time = time.time() - start_fl_time
                print(f"Fantasyland solved in {solve_time:.3f}s")
                sys.stdout.flush()

                if placement and discarded is not None:
                    # Возвращаем действие для применения размещения
                    return ("FANTASYLAND_PLACEMENT", placement, discarded)
                else:
                    # Если солвер не справился, возвращаем действие для фола
                    print("Warning: Fantasyland solver failed. Returning foul action.")
                    sys.stdout.flush()
                    return ("FANTASYLAND_FOUL", hand) # Передаем руку для сброса
            else:
                # Этого не должно происходить, если player_to_act определен верно
                print(f"AI Agent Error: Fantasyland player {player_to_act} has no hand. Returning None.")
                sys.stdout.flush()
                return None

        # --- Обычный ход (MCTS) ---
        print(f"Starting MCTS for player {player_to_act}...")
        sys.stdout.flush()

        # Получаем легальные действия
        initial_actions = game_state.get_legal_actions_for_player(player_to_act)
        print(f"Found {len(initial_actions)} legal actions initially.")
        sys.stdout.flush()

        if not initial_actions:
            print(f"AI Agent Warning: No legal actions found for player {player_to_act}. Returning None.")
            sys.stdout.flush()
            return None
        if len(initial_actions) == 1:
            print("Only one legal action found. Returning immediately.")
            sys.stdout.flush()
            return initial_actions[0]

        # Создаем корневой узел MCTS
        root_node = MCTSNode(game_state)
        # Заполняем неиспробованные действия и инициализируем RAVE
        root_node.untried_actions = list(initial_actions)
        random.shuffle(root_node.untried_actions)
        for act in root_node.untried_actions:
            if act not in root_node.rave_visits:
                root_node.rave_visits[act] = 0
                root_node.rave_total_reward[act] = 0.0

        start_mcts_time = time.time()
        num_simulations = 0

        # Основной цикл MCTS
        try:
            # Создаем пул воркеров
            # Используем 'with', чтобы пул гарантированно закрылся
            with multiprocessing.Pool(processes=self.num_workers) as pool:
                while time.time() - start_mcts_time < self.time_limit:
                    # 1. Выбор (Select) + Расширение (Expand)
                    path, leaf_node = self._select(root_node)
                    if leaf_node is None: continue # Не удалось выбрать узел

                    node_to_rollout_from = leaf_node
                    expanded_node = None

                    # Расширяем узел, если он не терминальный и есть неиспробованные действия
                    if not leaf_node.is_terminal():
                        if leaf_node.untried_actions:
                            expanded_node = leaf_node.expand()
                            if expanded_node:
                                node_to_rollout_from = expanded_node
                                path.append(expanded_node) # Добавляем новый узел в путь

                    # 2. Симуляция (Rollout) - Параллельно
                    results = []
                    simulation_actions_aggregated: Set[Any] = set()

                    if not node_to_rollout_from.is_terminal():
                        try:
                            # Сериализуем состояние для передачи воркерам
                            node_state_dict = node_to_rollout_from.game_state.to_dict()
                        except Exception as e:
                            print(f"Error serializing state for parallel rollout: {e}", file=sys.stderr)
                            sys.stderr.flush()
                            continue # Пропускаем итерацию, если не можем сериализовать

                        # Запускаем N роллаутов параллельно
                        async_results = [pool.apply_async(run_parallel_rollout, (node_state_dict,))
                                         for _ in range(self.rollouts_per_leaf)]

                        # Собираем результаты
                        for res in async_results:
                            try:
                                # Устанавливаем таймаут для получения результата
                                timeout_get = max(0.1, self.time_limit * 0.1)
                                reward, sim_actions = res.get(timeout=timeout_get)
                                results.append(reward)
                                simulation_actions_aggregated.update(sim_actions)
                                num_simulations += 1
                            except multiprocessing.TimeoutError:
                                print("Warning: Rollout worker timed out.", file=sys.stderr)
                                sys.stderr.flush()
                            except Exception as e:
                                print(f"Warning: Error getting result from worker: {e}", file=sys.stderr)
                                sys.stderr.flush()
                    else:
                        # Если узел терминальный, получаем счет напрямую
                        reward = node_to_rollout_from.game_state.get_terminal_score()
                        results.append(reward)
                        num_simulations += 1

                    # 3. Обратное распространение (Backpropagate)
                    if results:
                        total_reward_from_batch = sum(results)
                        num_rollouts_in_batch = len(results)
                        # Добавляем действие, которое привело к расширенному узлу (если он был)
                        if expanded_node and expanded_node.action:
                            simulation_actions_aggregated.add(expanded_node.action)
                        # Обновляем статистику узлов в пути
                        self._backpropagate_parallel(path, total_reward_from_batch, num_rollouts_in_batch, simulation_actions_aggregated)

        except Exception as e:
            print(f"Error during MCTS parallel execution: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            print("Choosing random action due to MCTS error.")
            # Возвращаем случайное действие при серьезной ошибке MCTS
            return random.choice(initial_actions) if initial_actions else None
        finally:
             # Убедимся, что пул закрыт, даже если была ошибка (хотя 'with' должен это делать)
             # pool.close()
             # pool.join()
             pass

        elapsed_time = time.time() - start_mcts_time
        sims_per_sec = (num_simulations / elapsed_time) if elapsed_time > 0 else 0
        print(f"MCTS finished: Ran {num_simulations} simulations in {elapsed_time:.3f}s ({sims_per_sec:.1f} sims/s).")
        sys.stdout.flush()

        # 4. Выбор лучшего действия из корневого узла
        best_action = self._select_best_action(root_node, initial_actions)

        total_time = time.time() - start_time_total
        print(f"--- AI Agent: Action chosen in {total_time:.3f}s ---")
        sys.stdout.flush()
        return best_action


    def _select(self, node: MCTSNode) -> Tuple[List[MCTSNode], Optional[MCTSNode]]:
        """
        Фаза выбора (Selection) MCTS: спускается по дереву, выбирая лучшие дочерние узлы
        по UCT+RAVE, пока не достигнет листового узла (не полностью расширенного или терминального).

        Args:
            node: Корневой узел для начала выбора.

        Returns:
            Кортеж (путь_узлов, листовой_узел).
            Листовой узел может быть None, если выбор не удался.
        """
        path = [node]
        current_node = node
        while True:
            # Если узел терминальный, выбор завершен
            if current_node.is_terminal():
                return path, current_node

            # Определяем, кто ходит из этого узла
            player_to_move = current_node._get_player_to_move()
            # Если никто не может ходить (например, ожидание раздачи), останавливаемся здесь
            if player_to_move == -1:
                 return path, current_node

            # Инициализируем неиспробованные действия, если нужно
            if current_node.untried_actions is None:
                current_node.untried_actions = current_node.game_state.get_legal_actions_for_player(player_to_move)
                random.shuffle(current_node.untried_actions)
                # Инициализация RAVE для новых действий
                for act in current_node.untried_actions:
                    if act not in current_node.rave_visits:
                        current_node.rave_visits[act] = 0
                        current_node.rave_total_reward[act] = 0.0

            # Если есть неиспробованные действия, этот узел - листовой для расширения
            if current_node.untried_actions:
                return path, current_node

            # Если нет неиспробованных действий и нет дочерних узлов (листовой узел без возможности расширения)
            if not current_node.children:
                 # Это может произойти, если get_legal_actions вернул пустой список, но узел не терминальный
                 # print(f"Warning: Leaf node reached with no children and no untried actions (Player {player_to_move}). State: {current_node.game_state}")
                 return path, current_node # Возвращаем текущий узел

            # Если все действия испробованы, выбираем лучший дочерний узел по UCT+RAVE
            selected_child = current_node.uct_select_child(self.exploration, self.rave_k)

            # Если не удалось выбрать дочерний узел (маловероятно, но возможно)
            if selected_child is None:
                print(f"Warning: uct_select_child returned None for node {current_node}. Parent had {len(current_node.children)} children.")
                # Попробуем выбрать случайного ребенка, если они есть
                if current_node.children:
                     try: selected_child = random.choice(list(current_node.children.values()))
                     except IndexError: return path, current_node # Не удалось выбрать
                else: return path, current_node # Нет детей

            # Переходим к выбранному дочернему узлу
            current_node = selected_child
            path.append(current_node)


    def _backpropagate_parallel(self, path: List[MCTSNode], total_reward: float, num_rollouts: int, simulation_actions: Set[Any]):
        """
        Фаза обратного распространения (Backpropagation) MCTS.
        Обновляет статистику (visits, total_reward) и RAVE-статистику
        для всех узлов в пройденном пути.

        Args:
            path: Список узлов от корня до узла, из которого проводились симуляции.
            total_reward: Суммарная награда (с точки зрения P0) за все симуляции из батча.
            num_rollouts: Количество симуляций в батче.
            simulation_actions: Множество всех действий, совершенных во всех симуляциях батча.
        """
        if num_rollouts == 0: return

        # Обновляем статистику для каждого узла в пути, начиная с конца
        for node in reversed(path):
            node.visits += num_rollouts
            # total_reward всегда с точки зрения P0
            node.total_reward += total_reward

            # Обновление RAVE статистики
            # Определяем, кто должен был ходить из этого узла
            player_to_move_from_node = node._get_player_to_move()
            if player_to_move_from_node != -1:
                # Находим действия, возможные из этого узла, которые были совершены в симуляциях
                possible_actions_from_node = set(node.children.keys())
                if node.untried_actions:
                    possible_actions_from_node.update(node.untried_actions)

                # Находим пересечение возможных действий и действий из симуляций
                relevant_sim_actions = simulation_actions.intersection(possible_actions_from_node)

                # Обновляем RAVE для релевантных действий
                for action in relevant_sim_actions:
                    if action not in node.rave_visits: # Инициализация, если действие новое
                        node.rave_visits[action] = 0
                        node.rave_total_reward[action] = 0.0
                    node.rave_visits[action] += num_rollouts
                    # rave_total_reward тоже храним с точки зрения P0
                    node.rave_total_reward[action] += total_reward


    def _select_best_action(self, root_node: MCTSNode, initial_actions: List[Any]) -> Optional[Any]:
        """Выбирает лучшее действие из дочерних узлов корневого узла."""
        if not root_node.children:
            print("Warning: No children found in root node after MCTS. Choosing random initial action.")
            sys.stdout.flush()
            return random.choice(initial_actions) if initial_actions else None

        best_action = None
        max_visits = -1

        # Перемешиваем дочерние узлы для случайного выбора при равных посещениях
        items = list(root_node.children.items())
        random.shuffle(items)

        print(f"--- Evaluating {len(items)} child nodes ---")
        # Выбираем действие, которое привело к узлу с наибольшим количеством посещений (Robust Child)
        for action, child_node in items:
             # Выводим статистику для отладки
             q_val_p0 = child_node.get_q_value(0) # Q с точки зрения P0
             rave_q_p0 = root_node.get_rave_q_value(action, 0) if root_node.rave_visits.get(action, 0) > 0 else 0.0
             print(f"  Action: {self._format_action(action):<40} Visits: {child_node.visits:<6} "
                   f"Q(P0): {q_val_p0:<8.3f} RAVE_Q(P0): {rave_q_p0:<8.3f} "
                   f"RAVE_V: {root_node.rave_visits.get(action, 0):<5}")

             if child_node.visits > max_visits:
                max_visits = child_node.visits
                best_action = action

        if best_action is None:
            # Если по какой-то причине не удалось выбрать (например, все посещения = 0)
            print("Warning: Could not determine best action based on visits. Choosing first shuffled.")
            sys.stdout.flush()
            best_action = items[0][0] if items else (random.choice(initial_actions) if initial_actions else None)

        if best_action:
            print(f"Selected action (max visits = {max_visits}): {self._format_action(best_action)}")
        else:
            print("Error: Failed to select any action.")
        sys.stdout.flush()
        return best_action

    def _format_action(self, action: Any) -> str:
        """Вспомогательная функция для форматирования действия в читаемую строку."""
        if action is None: return "None"
        try:
            # Формат для Pineapple (улицы 2-5)
            if isinstance(action, tuple) and len(action) == 3 and \
               isinstance(action[0], tuple) and len(action[0]) == 3 and isinstance(action[0][0], int) and \
               isinstance(action[1], tuple) and len(action[1]) == 3 and isinstance(action[1][0], int) and \
               isinstance(action[2], int):
                p1, p2, d = action
                return f"PINEAPPLE: {card_to_str(p1[0])}@{p1[1]}{p1[2]}, {card_to_str(p2[0])}@{p2[1]}{p2[2]}; DISC={card_to_str(d)}"
            # Формат для улицы 1
            elif isinstance(action, tuple) and len(action) == 2 and isinstance(action[0], list) and \
                 action[0] and isinstance(action[0][0], tuple) and len(action[0][0]) == 3 and isinstance(action[0][0][0], int):
                 placements_str = ", ".join([f"{card_to_str(c)}@{r}{i}" for c, r, i in action[0]])
                 return f"STREET 1: Place [{placements_str}]"
            # Формат для Фантазии (вход)
            elif isinstance(action, tuple) and action[0] == "FANTASYLAND_INPUT" and isinstance(action[1], list):
                 return f"FANTASYLAND_INPUT ({len(action[1])} cards)"
            # Формат для Фантазии (размещение)
            elif isinstance(action, tuple) and action[0] == "FANTASYLAND_PLACEMENT":
                 discard_count = len(action[2]) if isinstance(action[2], list) else '?'
                 return f"FANTASYLAND_PLACE (Discard {discard_count})"
            # Формат для Фантазии (фол)
            elif isinstance(action, tuple) and action[0] == "FANTASYLAND_FOUL":
                 discard_count = len(action[1]) if isinstance(action[1], list) else '?'
                 return f"FANTASYLAND_FOUL (Discard {discard_count})"
            # Общий случай для других кортежей
            elif isinstance(action, tuple):
                formatted_items = []
                for item in action:
                    if isinstance(item, (str, int, float, bool, type(None))): formatted_items.append(repr(item))
                    elif isinstance(item, list): formatted_items.append("[...]")
                    elif isinstance(item, dict): formatted_items.append("{...}")
                    else: formatted_items.append(self._format_action(item)) # Рекурсивный вызов для вложенных действий
                return f"Tuple Action: ({', '.join(formatted_items)})"
            # Для всего остального просто строка
            else:
                return str(action)
        except Exception as e:
            # print(f"Error formatting action {action}: {e}")
            return "ErrorFormattingAction"
