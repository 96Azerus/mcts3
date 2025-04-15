# src/fantasyland_solver.py
"""
Эвристический солвер для размещения 13 из N (14-17) карт в Progressive Fantasyland.
Приоритеты: 1. Не фол. 2. Удержание ФЛ (Re-Fantasy). 3. Максимизация роялти.
"""
import random
import time
import sys
from typing import List, Tuple, Dict, Optional
from itertools import combinations, permutations
from collections import Counter

# --- ИСПРАВЛЕНО: Импорты из src пакета ---
from src.card import Card as CardUtils, card_to_str, RANK_MAP, STR_RANKS, INVALID_CARD
from src.scoring import (
    check_fantasyland_stay, get_row_royalty, check_board_foul,
    get_hand_rank_safe, RANK_CLASS_QUADS, RANK_CLASS_TRIPS,
    RANK_CLASS_HIGH_CARD
)

# Импортируем PlayerBoard только для аннотации типов
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.board import PlayerBoard


class FantasylandSolver:
    """
    Решает задачу размещения руки Fantasyland (14-17 карт).
    Использует эвристики для поиска хорошего, но не обязательно оптимального решения.
    """
    # Максимальное количество комбинаций сброса для перебора (для производительности)
    MAX_DISCARD_COMBINATIONS = 100 # Увеличено для большего охвата
    # Максимальное количество размещений для оценки на каждую комбинацию сброса
    MAX_PLACEMENTS_PER_DISCARD = 50 # Можно настроить

    def solve(self, hand: List[int]) -> Tuple[Optional[Dict[str, List[int]]], Optional[List[int]]]:
        """
        Находит эвристически лучшее размещение 13 карт из N (14-17) карт руки Fantasyland.

        Args:
            hand (List[int]): Список из N (14-17) карт (int) в руке Фантазии.

        Returns:
            Tuple[Optional[Dict[str, List[int]]], Optional[List[int]]]:
            Кортеж:
            - Словарь с размещением {'top': [...], 'middle': [...], 'bottom': [...]} (карты int)
              или None, если валидное размещение не найдено.
            - Список сброшенных карт (int) или None.
        """
        n_cards = len(hand)
        n_place = 13
        if not (14 <= n_cards <= 17):
            print(f"Error in FL Solver: Invalid hand size {n_cards}. Expected 14-17.")
            # Возвращаем None, None как сигнал ошибки
            return None, None

        # --- ИСПРАВЛЕНО: Расчет n_discard на основе фактического размера руки ---
        n_discard = n_cards - n_place

        if n_discard < 1: # Должно быть от 1 до 4 карт для сброса
             print(f"Error in FL Solver: Calculated discard count {n_discard} is invalid for hand size {n_cards}.")
             return None, None

        # Проверка валидности карт в руке
        if any(not isinstance(c, int) or c == INVALID_CARD or c < 0 for c in hand):
             print(f"Error in FL Solver: Invalid cards found in hand: {[card_to_str(c) for c in hand]}")
             return None, None
        if len(hand) != len(set(hand)):
             print(f"Error in FL Solver: Duplicate cards found in hand: {[card_to_str(c) for c in hand]}")
             return None, None


        best_overall_placement: Optional[Dict[str, List[int]]] = None
        best_overall_discarded: Optional[List[int]] = None
        # Оценка: -1 = фол, 0 = не удерживает ФЛ, 1 = удерживает ФЛ
        best_overall_score: int = -2 # Начальное значение хуже фола
        best_overall_royalty: int = -1

        start_time = time.time()

        # 1. Генерируем комбинации карт для сброса
        try:
            discard_combinations_iter = combinations(hand, n_discard)
            # Преобразуем итератор в список (может быть затратно по памяти для больших N)
            # Ограничиваем количество комбинаций для анализа
            discard_combinations_list = list(itertools.islice(discard_combinations_iter, self.MAX_DISCARD_COMBINATIONS * 2)) # Берем с запасом
            if len(discard_combinations_list) > self.MAX_DISCARD_COMBINATIONS:
                 # Добавляем "умный" сброс (самые младшие карты)
                 try:
                      sorted_hand = sorted(hand, key=lambda c: CardUtils.get_rank_int(c))
                      smart_discards = [tuple(sorted_hand[:n_discard])]
                 except Exception as e_sort:
                      print(f"Warning: Error sorting hand for smart discard in FL solver: {e_sort}")
                      smart_discards = []
                 # Выбираем случайные комбинации + умный сброс
                 num_random_needed = max(0, self.MAX_DISCARD_COMBINATIONS - len(smart_discards))
                 if num_random_needed > 0 and len(discard_combinations_list) > num_random_needed:
                      random_discards = random.sample(discard_combinations_list, num_random_needed)
                      combinations_to_check = smart_discards + random_discards
                 else:
                      combinations_to_check = smart_discards[:self.MAX_DISCARD_COMBINATIONS]
            else:
                 combinations_to_check = discard_combinations_list

        except Exception as e_comb:
            print(f"Error generating discard combinations: {e_comb}")
            return None, None # Не можем продолжить без комбинаций

        if not combinations_to_check:
             print("Error: No discard combinations generated.")
             return None, None

        # 2. Перебираем варианты сброса
        for discarded_tuple in combinations_to_check:
            discarded_list = list(discarded_tuple)
            # Формируем набор из 13 карт для размещения
            remaining_cards = [c for c in hand if c not in discarded_list]
            if len(remaining_cards) != 13:
                # print(f"Debug: Skipping discard combo due to wrong remaining count {len(remaining_cards)}")
                continue # Пропускаем, если что-то пошло не так

            current_best_placement_for_discard = None
            current_best_score_for_discard = -1 # -1 фол, 0 не ФЛ, 1 ФЛ
            current_max_royalty_for_discard = -1

            # 3. Генерируем и оцениваем эвристические размещения для текущих 13 карт
            placements_to_evaluate = self._generate_heuristic_placements(remaining_cards)

            for placement in placements_to_evaluate:
                if not placement: continue # Пропускаем None

                # Оцениваем размещение (фол, удержание ФЛ, роялти)
                score, royalty = self._evaluate_placement(placement)

                # Выбираем лучшее размещение для ДАННОГО набора 13 карт
                # Приоритет: не фол -> удержание ФЛ -> макс роялти
                if score > current_best_score_for_discard or \
                   (score == current_best_score_for_discard and royalty > current_max_royalty_for_discard):
                    current_best_score_for_discard = score
                    current_max_royalty_for_discard = royalty
                    current_best_placement_for_discard = placement

            # 4. Обновляем лучшее ОБЩЕЕ размещение (среди всех вариантов сброса)
            if current_best_placement_for_discard:
                if current_best_score_for_discard > best_overall_score or \
                   (current_best_score_for_discard == best_overall_score and current_max_royalty_for_discard > best_overall_royalty):
                    best_overall_score = current_best_score_for_discard
                    best_overall_royalty = current_max_royalty_for_discard
                    best_overall_placement = current_best_placement_for_discard
                    best_overall_discarded = discarded_list

        solve_duration = time.time() - start_time
        # print(f"FL Solver: Checked {len(combinations_to_check)} discard combinations in {solve_duration:.3f}s")

        # 5. Обработка случая, если не найдено ни одного валидного размещения
        if best_overall_placement is None:
            print("FL Solver Warning: No valid non-foul placement found for any discard combination.")
            # Пытаемся сделать простое безопасное размещение для первого варианта сброса
            if combinations_to_check:
                first_discard = list(combinations_to_check[0])
                first_remaining = [c for c in hand if c not in first_discard]
                if len(first_remaining) == 13:
                    fallback_placement = self._generate_simple_safe_placement(first_remaining)
                    if fallback_placement:
                        print("FL Solver: Falling back to simple non-foul placement.")
                        return fallback_placement, first_discard
            # Если и это не помогло, возвращаем None, None (сигнал фола для агента)
            print("FL Solver Error: Could not generate any valid placement.")
            return None, None

        return best_overall_placement, best_overall_discarded

    def _generate_heuristic_placements(self, cards: List[int]) -> List[Optional[Dict[str, List[int]]]]:
        """Генерирует несколько кандидатов на размещение 13 карт."""
        if len(cards) != 13: return []

        placements = []
        # Эвристика 1: Пытаемся собрать Каре+ на боттоме
        placements.append(self._try_build_strong_bottom(cards))
        # Эвристика 2: Пытаемся собрать Сет на топе
        placements.append(self._try_build_set_top(cards))
        # Эвристика 3: Пытаемся максимизировать роялти (общее размещение)
        placements.append(self._try_maximize_royalty_heuristic(cards))
        # Эвристика 4: Простое безопасное размещение (на всякий случай)
        placements.append(self._generate_simple_safe_placement(cards))

        # Убираем None и дубликаты (хотя дубликаты маловероятны)
        unique_placements = []
        seen_placements = set()
        for p in placements:
             if p:
                  # Создаем неизменяемое представление для проверки дубликатов
                  p_tuple = tuple(tuple(sorted(p[row])) for row in ['top', 'middle', 'bottom'])
                  if p_tuple not in seen_placements:
                       unique_placements.append(p)
                       seen_placements.add(p_tuple)

        # Ограничиваем количество для оценки
        return random.sample(unique_placements, min(len(unique_placements), self.MAX_PLACEMENTS_PER_DISCARD))


    def _evaluate_placement(self, placement: Dict[str, List[int]]) -> Tuple[int, int]:
        """
        Оценивает готовое размещение 13 карт.

        Args:
            placement: Словарь с размещением {'top': [...], 'middle': [...], 'bottom': [...]}.

        Returns:
            Tuple[int, int]: Кортеж (score, total_royalty).
                             score: -1 (фол), 0 (не удерживает ФЛ), 1 (удерживает ФЛ).
                             total_royalty: Сумма роялти (или -1 при фоле).
        """
        # Проверка корректности структуры placement
        if not placement or \
           len(placement.get('top', [])) != 3 or \
           len(placement.get('middle', [])) != 5 or \
           len(placement.get('bottom', [])) != 5:
             print("Warning: Invalid placement structure in _evaluate_placement.")
             return -1, -1 # Некорректное размещение считаем фолом

        top, middle, bottom = placement['top'], placement['middle'], placement['bottom']

        # Проверяем на фол
        if check_board_foul(top, middle, bottom):
            return -1, -1 # Фол

        # Проверяем удержание ФЛ
        stays_in_fl = check_fantasyland_stay(top, middle, bottom)

        # Считаем роялти
        try:
            total_royalty = (get_row_royalty(top, 'top') +
                             get_row_royalty(middle, 'middle') +
                             get_row_royalty(bottom, 'bottom'))
        except Exception as e_royalty:
             print(f"Error calculating royalty during FL placement evaluation: {e_royalty}")
             total_royalty = 0 # Считаем 0 роялти при ошибке

        score = 1 if stays_in_fl else 0
        return score, total_royalty

    def _find_best_hand(self, cards: List[int], n: int) -> Optional[List[int]]:
        """Находит лучшую n-карточную комбинацию (по рангу) из списка карт."""
        if len(cards) < n: return None

        best_hand_combo: Optional[tuple[int, ...]] = None
        # Инициализируем худшим возможным рангом + запас
        best_rank: int = RANK_CLASS_HIGH_CARD + 1000

        # Перебираем комбинации n карт
        for combo in combinations(cards, n):
            combo_list = list(combo)
            # Используем get_hand_rank_safe, т.к. комбинация всегда будет полной (n карт)
            rank = get_hand_rank_safe(combo_list)
            # Меньший ранг лучше
            if rank < best_rank:
                best_rank = rank
                best_hand_combo = combo # Сохраняем кортеж

        # Возвращаем список или None
        return list(best_hand_combo) if best_hand_combo is not None else None

    def _generate_simple_safe_placement(self, cards: List[int]) -> Optional[Dict[str, List[int]]]:
         """Создает простое размещение, сортируя карты и раскладывая по рядам, проверяя на фол."""
         if len(cards) != 13: return None
         try:
              # Сортируем карты от старшей к младшей
              sorted_cards = sorted(cards, key=lambda c: CardUtils.get_rank_int(c), reverse=True)
              # Простое размещение
              placement = {
                   'bottom': sorted_cards[0:5],
                   'middle': sorted_cards[5:10],
                   'top': sorted_cards[10:13]
              }
              # Проверяем на фол
              if not check_board_foul(placement['top'], placement['middle'], placement['bottom']):
                   return placement
              else:
                   # Если фол, пытаемся поменять местами средний и нижний ряд (частая причина фола)
                   placement_swapped = {
                        'bottom': sorted_cards[5:10],
                        'middle': sorted_cards[0:5],
                        'top': sorted_cards[10:13]
                   }
                   if not check_board_foul(placement_swapped['top'], placement_swapped['middle'], placement_swapped['bottom']):
                        return placement_swapped
                   else:
                        # Если и так фол, возвращаем None
                        return None
         except Exception as e:
              print(f"Error creating simple safe placement: {e}")
              return None

    # --- Эвристики для генерации кандидатов ---
    # (Эти методы пытаются построить сильные комбинации в нужных рядах)

    def _try_build_strong_bottom(self, cards: List[int]) -> Optional[Dict[str, List[int]]]:
        """Пытается собрать Каре+ на боттоме, остальное эвристически."""
        if len(cards) != 13: return None

        best_placement_candidate = None
        max_royalty = -1 # Отслеживаем роялти для выбора лучшего варианта

        # Ограниченный перебор комбинаций для боттома
        bottom_combos_iter = combinations(cards, 5)
        bottom_combos_sample = list(itertools.islice(bottom_combos_iter, 200)) # Ограничение

        for bottom_list in bottom_combos_sample:
            rank_b = get_hand_rank_safe(bottom_list)

            # Проверяем, является ли боттом Каре или лучше
            if rank_b <= RANK_CLASS_QUADS: # Каре = 166
                remaining8 = [c for c in cards if c not in bottom_list]
                if len(remaining8) != 8: continue

                # Находим лучшую среднюю руку из оставшихся 8
                middle_list = self._find_best_hand(remaining8, 5)
                if middle_list:
                    top_list = [c for c in remaining8 if c not in middle_list]
                    if len(top_list) == 3:
                        # Проверяем на фол
                        rank_m = get_hand_rank_safe(middle_list)
                        rank_t = get_hand_rank_safe(top_list)
                        if not (rank_b <= rank_m <= rank_t): continue # Фол

                        placement = {'top': top_list, 'middle': middle_list, 'bottom': list(bottom_list)}
                        _, royalty = self._evaluate_placement(placement)

                        if royalty > max_royalty:
                            max_royalty = royalty
                            best_placement_candidate = placement
        return best_placement_candidate

    def _try_build_set_top(self, cards: List[int]) -> Optional[Dict[str, List[int]]]:
        """Пытается собрать Сет на топе, остальное эвристически."""
        if len(cards) != 13: return None

        best_placement_candidate = None
        max_royalty = -1

        # Ищем возможные сеты
        try:
            rank_counts = Counter(CardUtils.get_rank_int(c) for c in cards)
            possible_set_ranks = [rank for rank, count in rank_counts.items() if count >= 3]
        except Exception as e:
            print(f"Warning: Error counting ranks for set top in FL solver: {e}")
            possible_set_ranks = []

        for set_rank_index in possible_set_ranks:
            # Собираем карты для сета
            set_cards = [c for c in cards if CardUtils.get_rank_int(c) == set_rank_index][:3]
            if len(set_cards) != 3: continue

            remaining10 = [c for c in cards if c not in set_cards]
            if len(remaining10) != 10: continue

            # Находим лучшую нижнюю руку из оставшихся 10
            bottom_list = self._find_best_hand(remaining10, 5)
            if bottom_list:
                 middle_list = [c for c in remaining10 if c not in bottom_list]
                 if len(middle_list) == 5:
                     # Проверяем на фол
                     rank_b = get_hand_rank_safe(bottom_list)
                     rank_m = get_hand_rank_safe(middle_list)
                     rank_t = get_hand_rank_safe(set_cards)
                     if not (rank_b <= rank_m <= rank_t): continue # Фол

                     placement = {'top': set_cards, 'middle': middle_list, 'bottom': bottom_list}
                     _, royalty = self._evaluate_placement(placement)

                     if royalty > max_royalty:
                         max_royalty = royalty
                         best_placement_candidate = placement
        return best_placement_candidate

    def _try_maximize_royalty_heuristic(self, cards: List[int]) -> Optional[Dict[str, List[int]]]:
        """Простая эвристика: размещаем лучшие возможные руки на боттом/мидл/топ без фола."""
        if len(cards) != 13: return None

        best_placement_candidate = None
        max_royalty = -1

        # Ограниченный перебор комбинаций для боттома
        bottom_combos_iter = combinations(cards, 5)
        bottom_combos_sample = list(itertools.islice(bottom_combos_iter, 150)) # Ограничение

        for bottom_list in bottom_combos_sample:
            remaining8 = [c for c in cards if c not in bottom_list]
            if len(remaining8) != 8: continue

            # Находим лучшую среднюю руку
            middle_list = self._find_best_hand(remaining8, 5)
            if middle_list:
                 top_list = [c for c in remaining8 if c not in middle_list]
                 if len(top_list) == 3:
                     # Проверяем на фол
                     rank_b = get_hand_rank_safe(list(bottom_list))
                     rank_m = get_hand_rank_safe(middle_list)
                     rank_t = get_hand_rank_safe(top_list)
                     if not (rank_b <= rank_m <= rank_t): continue # Фол

                     placement = {'top': top_list, 'middle': middle_list, 'bottom': list(bottom_list)}
                     _, royalty = self._evaluate_placement(placement)

                     if royalty > max_royalty:
                         max_royalty = royalty
                         best_placement_candidate = placement

        # Если перебор не дал результата или роялти низкие, возвращаем простое безопасное
        if not best_placement_candidate or max_royalty < 5: # Порог можно настроить
             simple_safe = self._generate_simple_safe_placement(cards)
             if simple_safe:
                  _, simple_royalty = self._evaluate_placement(simple_safe)
                  if simple_royalty > max_royalty:
                       return simple_safe

        return best_placement_candidate
