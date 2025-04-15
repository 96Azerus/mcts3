# main.py
"""
Пример использования модулей игры OFC Pineapple для запуска
простого сценария в консоли (например, для отладки).
Использует GameState.advance_state() для управления ходом игры.
"""
import time
import random
import os
import sys
import traceback
from typing import List, Optional, Tuple, Any

# Импорты из src пакета
try:
    from src.card import card_from_str, card_to_str, Card as CardUtils, INVALID_CARD
    from src.game_state import GameState
    from src.mcts_agent import MCTSAgent
    from src.fantasyland_solver import FantasylandSolver
    from src.scoring import check_board_foul
    from src.board import PlayerBoard
except ImportError as e:
    print(f"Ошибка импорта в main.py: {e}", file=sys.stderr)
    print("Убедитесь, что вы запускаете скрипт из корневой директории проекта,", file=sys.stderr)
    print("или что директория src доступна в PYTHONPATH.", file=sys.stderr)
    sys.exit(1)

# --- Функции ввода пользователя (остаются как пример) ---
# (Функции get_human_action_street1, get_human_action_pineapple, get_human_fantasyland_placement
#  остаются без изменений, но не используются в run_simple_scenario)
def get_human_action_street1(hand: List[int], board: 'PlayerBoard') -> Tuple[List[Tuple[int, str, int]], List[int]]:
    # ... (код функции) ...
    print("\nВаша рука (Улица 1):", ", ".join(card_to_str(c) for c in hand))
    print("Ваша доска:")
    print(board)
    placements = []
    available_slots = board.get_available_slots()
    if len(available_slots) >= 5:
         for i in range(5):
              card = hand[i]; row, idx = available_slots[i]; placements.append((card, row, idx))
    print(f"[Пример] Авто-размещение улицы 1: {placements}")
    return placements, []

def get_human_action_pineapple(hand: List[int], board: 'PlayerBoard') -> Optional[Tuple[Tuple[int, str, int], Tuple[int, str, int], int]]:
    # ... (код функции) ...
    print("\nВаша рука (Улицы 2-5):", ", ".join(f"{i+1}:{card_to_str(c)}" for i, c in enumerate(hand)))
    print("Ваша доска:")
    print(board)
    available_slots = board.get_available_slots()
    if len(available_slots) >= 2 and len(hand) == 3:
         discard_card = hand[2]; card1, card2 = hand[0], hand[1]; slot1, slot2 = available_slots[0], available_slots[1]
         action = ((card1, slot1[0], slot1[1]), (card2, slot2[0], slot2[1]), discard_card)
         print(f"[Пример] Авто-ход Pineapple: {action}")
         return action
    return None

def get_human_fantasyland_placement(hand: List[int]) -> Tuple[Optional[Dict[str, List[int]]], Optional[List[int]]]:
     # ... (код функции) ...
     print("\n--- FANTASYLAND ---"); print("Ваша рука:", ", ".join(card_to_str(c) for c in hand))
     solver = FantasylandSolver(); placement, discarded = solver.solve(hand)
     if placement: print("[Пример] Авто-размещение Фантазии (солвер)."); return placement, discarded
     else: print("[Пример] Не удалось разместить Фантазию (фол)."); return None, hand[13:] if len(hand)>13 else []


# --- Упрощенный основной сценарий ---
def run_simple_scenario():
    """Запускает простой сценарий игры AI vs AI для демонстрации."""
    print("--- Запуск упрощенного сценария (AI vs AI) ---")

    try:
        mcts_time_limit = int(os.environ.get('MCTS_TIME_LIMIT_MS', 1000)); mcts_workers = int(os.environ.get('NUM_WORKERS', 1))
        ai_agent = MCTSAgent(time_limit_ms=mcts_time_limit, num_workers=mcts_workers)
    except Exception as e: print(f"Ошибка инициализации AI: {e}"); return

    dealer_idx = random.choice([0, 1])
    game_state = GameState(dealer_idx=dealer_idx)
    game_state.start_new_round(dealer_idx)

    print(f"Начало раунда (Дилер: {dealer_idx})")

    max_steps = 50 # Увеличим лимит шагов
    for step in range(max_steps):
        print(f"\n--- Шаг {step+1} ---")
        print(f"Улица: {game_state.street}, Дилер: {game_state.dealer_idx}")
        print("Доска P0:\n", game_state.boards[0])
        print("Доска P1:\n", game_state.boards[1])
        if game_state.current_hands.get(0): print("Рука P0:", [card_to_str(c) for c in game_state.current_hands[0]])
        if game_state.current_hands.get(1): print("Рука P1:", [card_to_str(c) for c in game_state.current_hands[1]])
        if game_state.fantasyland_hands[0]: print("ФЛ Рука P0:", len(game_state.fantasyland_hands[0]), "карт")
        if game_state.fantasyland_hands[1]: print("ФЛ Рука P1:", len(game_state.fantasyland_hands[1]), "карт")

        if game_state.is_round_over():
            print("\n--- Раунд завершен ---")
            break

        # --- РЕФАКТОРИНГ: Используем get_player_to_move и advance_state ---
        player_to_act = game_state.get_player_to_move()
        print(f"Ожидаемый ход: Игрок {player_to_act if player_to_act != -1 else 'Ожидание'}")

        if player_to_act != -1:
            print(f"Ход Игрока {player_to_act}...")
            action = ai_agent.choose_action(game_state)

            if action is None:
                print(f"Ошибка: AI Игрок {player_to_act} не смог выбрать ход. Применяем фол.")
                # Применяем фол вручную для простоты сценария
                hand_to_foul = game_state.get_player_hand(player_to_act)
                if game_state.is_fantasyland_round and game_state.fantasyland_status[player_to_act]:
                     if hand_to_foul: game_state = game_state.apply_fantasyland_foul(player_to_act, hand_to_foul)
                     else: game_state._player_finished_round[player_to_act] = True # Если руки нет, просто завершаем
                else:
                     board = game_state.boards[player_to_act]; board.is_foul = True
                     game_state._player_finished_round[player_to_act] = True
                     if hand_to_foul: game_state.private_discard[player_to_act].extend(hand_to_foul); game_state.current_hands[player_to_act] = None
                print(f"Игрок {player_to_act} сфолил.")
            else:
                # Применяем ход AI
                is_fl_placement = isinstance(action, tuple) and action[0] == "FANTASYLAND_PLACEMENT"
                is_fl_foul = isinstance(action, tuple) and action[0] == "FANTASYLAND_FOUL"
                action_str = ai_agent._format_action(action) # Получаем строку до применения

                if is_fl_placement:
                    _, placement, discarded = action
                    game_state = game_state.apply_fantasyland_placement(player_to_act, placement, discarded)
                    print(f"Игрок {player_to_act} разместил Фантазию.")
                elif is_fl_foul:
                    _, hand_to_discard = action
                    game_state = game_state.apply_fantasyland_foul(player_to_act, hand_to_discard)
                    print(f"Игрок {player_to_act} сфолил в Фантазии.")
                else:
                    game_state = game_state.apply_action(player_to_act, action)
                    print(f"Игрок {player_to_act} применил действие: {action_str}")

            # Продвигаем состояние ПОСЛЕ хода
            game_state = game_state.advance_state()

        else: # player_to_act == -1
             print("Никто не может ходить. Продвигаем состояние для раздачи...")
             advanced_state = game_state.advance_state()
             if advanced_state == game_state:
                  print("Состояние не изменилось после advance_state. Возможно, конец раунда или нет карт.")
                  break # Выходим из цикла, если продвижение не помогло
             game_state = advanced_state

        time.sleep(0.1) # Небольшая пауза для читаемости вывода

    # Вывод финального состояния
    print("\n--- Финальные доски ---")
    print("Игрок 0:")
    print(game_state.boards[0])
    print("Игрок 1:")
    print(game_state.boards[1])
    if game_state.is_round_over():
        score_diff = game_state.get_terminal_score()
        print(f"Счет за раунд (P0 vs P1): {score_diff}")
    else:
        print(f"\n--- Раунд не завершен после {max_steps} шагов ---")

if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            if sys.stdout.encoding != 'utf-8': sys.stdout.reconfigure(encoding='utf-8')
            if sys.stderr.encoding != 'utf-8': sys.stderr.reconfigure(encoding='utf-8')
        except Exception as e: print(f"Warning: Could not set console encoding: {e}")
    run_simple_scenario()
