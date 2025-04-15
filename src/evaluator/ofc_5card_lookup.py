# src/evaluator/ofc_5card_lookup.py
# -*- coding: utf-8 -*-
"""
Генерация таблиц поиска для 5-карточного эвалуатора OFC.
Использует вариант алгоритма Cactus Kev с простыми числами.
"""

import itertools
from typing import Dict, List, Generator

# --- ИСПРАВЛЕНО: Импортируем CardUtils из src пакета ---
try:
    from src.card import Card as CardUtils, INT_RANKS, PRIMES
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import from src.card in ofc_5card_lookup.py: {e}")
    # Заглушка, чтобы модуль хотя бы загрузился, но работать не будет
    class CardUtils:
        PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
        INT_RANKS = range(13)
        @staticmethod
        def prime_product_from_rankbits(rankbits): return 1
        @staticmethod
        def prime_product_from_hand(card_ints): return 1
    INT_RANKS = range(13)
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
    # В реальной ситуации здесь лучше выбросить исключение, чтобы остановить запуск
    # raise ImportError("Failed to import Card utilities, cannot initialize 5-card lookup table.") from e


class LookupTable:
    """
    Создает и хранит таблицы поиска для быстрой оценки 5-карточных рук.

    Таблицы:
    - flush_lookup: Отображает произведение простых чисел для рангов флеша
                    в ранг руки [1 (Royal Flush) .. 1599].
    - unsuited_lookup: Отображает произведение простых чисел для неулучшенных рук
                       (не флеш) в ранг руки [11 (Four Aces) .. 7462 (7-5-4-3-2)].

    Ранги: Меньшее значение соответствует более сильной руке.
    """
    # Константы, определяющие границы рангов для каждого класса рук
    MAX_STRAIGHT_FLUSH: int = 10
    MAX_FOUR_OF_A_KIND: int = 166  # 10 + 156
    MAX_FULL_HOUSE: int = 322      # 166 + 156
    MAX_FLUSH: int = 1599          # 322 + 1277
    MAX_STRAIGHT: int = 1609       # 1599 + 10
    MAX_THREE_OF_A_KIND: int = 2467 # 1609 + 858
    MAX_TWO_PAIR: int = 3325       # 2467 + 858
    MAX_PAIR: int = 6185           # 3325 + 2860
    MAX_HIGH_CARD: int = 7462      # 6185 + 1277

    # Отображение максимального ранга класса в номер класса (1-9)
    MAX_TO_RANK_CLASS: Dict[int, int] = {
        MAX_STRAIGHT_FLUSH: 1,
        MAX_FOUR_OF_A_KIND: 2,
        MAX_FULL_HOUSE: 3,
        MAX_FLUSH: 4,
        MAX_STRAIGHT: 5,
        MAX_THREE_OF_A_KIND: 6,
        MAX_TWO_PAIR: 7,
        MAX_PAIR: 8,
        MAX_HIGH_CARD: 9
    }

    # Отображение номера класса в строку
    RANK_CLASS_TO_STRING: Dict[int, str] = {
        1: "Straight Flush",
        2: "Four of a Kind",
        3: "Full House",
        4: "Flush",
        5: "Straight",
        6: "Three of a Kind",
        7: "Two Pair",
        8: "Pair",
        9: "High Card"
    }

    def __init__(self):
        """
        Инициализирует и вычисляет таблицы поиска.
        """
        self.flush_lookup: Dict[int, int] = {}
        self.unsuited_lookup: Dict[int, int] = {}

        # Заполняем таблицы
        print("Initializing 5-card lookup tables...")
        self._calculate_flushes()
        self._calculate_multiples()
        print(f"Lookup tables initialized. Flush: {len(self.flush_lookup)}, Unsuited: {len(self.unsuited_lookup)}")

    def _calculate_flushes(self):
        """
        Вычисляет ранги для стрит-флешей и обычных флешей.
        Использует битовые маски рангов.
        """
        # Битовые маски для стрит-флешей (от AKQJT до A5432)
        straight_flushes_rank_bits: List[int] = [
            0b1111100000000, # A, K, Q, J, T (Royal)
            0b0111110000000, # K, Q, J, T, 9
            0b0011111000000, # Q, J, T, 9, 8
            0b0001111100000, # J, T, 9, 8, 7
            0b0000111110000, # T, 9, 8, 7, 6
            0b0000011111000, # 9, 8, 7, 6, 5
            0b0000001111100, # 8, 7, 6, 5, 4
            0b0000000111110, # 7, 6, 5, 4, 3
            0b0000000011111, # 6, 5, 4, 3, 2
            0b1000000001111, # A, 5, 4, 3, 2 (Wheel)
        ]

        # Генерация всех возможных 5-карточных комбинаций рангов (битовые маски)
        all_flush_rank_bits: List[int] = []
        # Начинаем с 0b11111 (комбинация 65432)
        gen = self._get_lexographically_next_bit_sequence(0b11111)
        # Всего C(13, 5) = 1287 комбинаций
        for _ in range(1287):
            try:
                rank_bits = next(gen)
                all_flush_rank_bits.append(rank_bits)
            except StopIteration:
                print("Warning: StopIteration in bit sequence generator (should not happen).")
                break

        # Отделяем обычные флеши от стрит-флешей
        straight_flush_set = set(straight_flushes_rank_bits)
        normal_flush_rank_bits = [
            rb for rb in all_flush_rank_bits if rb not in straight_flush_set
        ]

        # Сортируем маски по убыванию (сильные комбинации сначала)
        straight_flushes_rank_bits.sort(reverse=True) # Уже отсортированы правильно
        normal_flush_rank_bits.sort(reverse=True)

        # Заполняем flush_lookup
        # Стрит-флеши (ранги 1-10)
        rank = 1
        for sf_bits in straight_flushes_rank_bits:
            prime_product = CardUtils.prime_product_from_rankbits(sf_bits)
            self.flush_lookup[prime_product] = rank
            rank += 1

        # Обычные флеши (ранги 323 - 1599)
        rank = LookupTable.MAX_FULL_HOUSE + 1
        for f_bits in normal_flush_rank_bits:
            prime_product = CardUtils.prime_product_from_rankbits(f_bits)
            self.flush_lookup[prime_product] = rank
            rank += 1

        # Используем эти же битовые маски для заполнения части unsuited_lookup
        self._calculate_straights_and_highcards(straight_flushes_rank_bits, normal_flush_rank_bits)

    def _calculate_straights_and_highcards(self, straights_rank_bits: List[int], highcards_rank_bits: List[int]):
        """
        Вычисляет ранги для стритов и старших карт (не флеш).
        Использует битовые маски рангов, полученные при расчете флешей.
        """
        # Стриты (ранги 1600 - 1609)
        rank = LookupTable.MAX_FLUSH + 1
        for s_bits in straights_rank_bits:
            prime_product = CardUtils.prime_product_from_rankbits(s_bits)
            self.unsuited_lookup[prime_product] = rank
            rank += 1

        # Старшие карты (ранги 6186 - 7462)
        rank = LookupTable.MAX_PAIR + 1
        # highcards_rank_bits уже отсортированы по убыванию
        for h_bits in highcards_rank_bits:
            prime_product = CardUtils.prime_product_from_rankbits(h_bits)
            self.unsuited_lookup[prime_product] = rank
            rank += 1

    def _calculate_multiples(self):
        """
        Вычисляет ранги для Каре, Фулл-хаусов, Сетов, Двух пар и Пар.
        Использует произведение простых чисел рангов карт.
        """
        # Ранги в обратном порядке (A, K, Q, ..., 2) -> (12, 11, ..., 0)
        backwards_ranks = range(len(INT_RANKS) - 1, -1, -1)

        # 1) Каре (ранги 11 - 166)
        rank = LookupTable.MAX_STRAIGHT_FLUSH + 1
        for quad_rank_idx in backwards_ranks: # Ранг каре
            # Ранг кикера
            kickers_indices = list(backwards_ranks)
            kickers_indices.remove(quad_rank_idx)
            for kicker_idx in kickers_indices:
                product = PRIMES[quad_rank_idx]**4 * PRIMES[kicker_idx]
                self.unsuited_lookup[product] = rank
                rank += 1

        # 2) Фулл-хаус (ранги 167 - 322)
        rank = LookupTable.MAX_FOUR_OF_A_KIND + 1
        for trip_rank_idx in backwards_ranks: # Ранг сета
            # Ранг пары
            pair_rank_indices = list(backwards_ranks)
            pair_rank_indices.remove(trip_rank_idx)
            for pair_rank_idx in pair_rank_indices:
                product = PRIMES[trip_rank_idx]**3 * PRIMES[pair_rank_idx]**2
                self.unsuited_lookup[product] = rank
                rank += 1

        # 3) Сет (ранги 1610 - 2467)
        rank = LookupTable.MAX_STRAIGHT + 1
        for trip_rank_idx in backwards_ranks: # Ранг сета
            # Ранги двух кикеров
            kickers_indices = list(backwards_ranks)
            kickers_indices.remove(trip_rank_idx)
            # Комбинации кикеров (k1 > k2)
            kicker_combos = itertools.combinations(kickers_indices, 2)
            for k1_idx, k2_idx in kicker_combos:
                product = PRIMES[trip_rank_idx]**3 * PRIMES[k1_idx] * PRIMES[k2_idx]
                self.unsuited_lookup[product] = rank
                rank += 1

        # 4) Две пары (ранги 2468 - 3325)
        rank = LookupTable.MAX_THREE_OF_A_KIND + 1
        # Комбинации рангов для двух пар (p1 > p2)
        two_pair_combos = itertools.combinations(backwards_ranks, 2)
        for p1_idx, p2_idx in two_pair_combos:
            # Ранг кикера
            kickers_indices = list(backwards_ranks)
            kickers_indices.remove(p1_idx)
            kickers_indices.remove(p2_idx)
            for kicker_idx in kickers_indices:
                product = PRIMES[p1_idx]**2 * PRIMES[p2_idx]**2 * PRIMES[kicker_idx]
                self.unsuited_lookup[product] = rank
                rank += 1

        # 5) Пара (ранги 3326 - 6185)
        rank = LookupTable.MAX_TWO_PAIR + 1
        for pair_rank_idx in backwards_ranks: # Ранг пары
            # Ранги трех кикеров
            kickers_indices = list(backwards_ranks)
            kickers_indices.remove(pair_rank_idx)
            # Комбинации кикеров (k1 > k2 > k3)
            kicker_combos = itertools.combinations(kickers_indices, 3)
            for k1_idx, k2_idx, k3_idx in kicker_combos:
                product = PRIMES[pair_rank_idx]**2 * PRIMES[k1_idx] * PRIMES[k2_idx] * PRIMES[k3_idx]
                self.unsuited_lookup[product] = rank
                rank += 1

    def _get_lexographically_next_bit_sequence(self, bits: int) -> Generator[int, None, None]:
        """
        Генератор, возвращающий следующую лексикографическую перестановку битов.
        Использует трюк с битовыми операциями.
        Источник: http://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
        """
        next_val = bits
        while True:
            # t вычисляет маску для переноса бита влево
            t = (next_val | (next_val - 1)) + 1
            # next_val получает следующую перестановку
            # Проверка деления на ноль (если next_val & -next_val равно 0)
            next_val_lsb = next_val & -next_val
            if next_val_lsb == 0: break # Предотвращаем деление на ноль
            next_val = t | ((((t & -t) // next_val_lsb) >> 1) - 1)
            # Проверка на максимальное значение для 13 бит (C(13,5))
            # 0b1111100000000 = 7936 + 2048 + 1024 + 512 + 256 = 11776? Нет.
            # C(13,5) - последняя комбинация: 1111100000000 (AKQJT)
            # Первая комбинация: 0000000011111 (65432)
            # Если вернулись к началу или перешли его, останавливаемся
            if next_val == 0 or next_val > 0b1111100000000: # Ограничиваем 13 битами
                 break
            yield next_val

# Этот блок выполняется только при запуске файла напрямую (для отладки/генерации)
if __name__ == '__main__':
    print("Generating 5-card lookup table...")
    lookup = LookupTable()
    print("Table generated.")
    # Можно добавить проверку некоторых значений
    # Пример: Каре тузов с королем
    # AAAA K -> Ranks: 12, 12, 12, 12, 11
    # Primes: 41^4 * 37 = 2825761 * 37 = 104553157
    prime_4a_k = CardUtils.PRIMES[12]**4 * CardUtils.PRIMES[11]
    print(f"Rank for AAAA K (prime {prime_4a_k}): {lookup.unsuited_lookup.get(prime_4a_k)}") # Должно быть 11

    # Пример: Роял флеш
    # AKQJT -> Bits: 0b1111100000000
    # Primes: 41 * 37 * 31 * 29 * 23 = 31338851
    prime_rf = CardUtils.prime_product_from_rankbits(0b1111100000000)
    print(f"Rank for Royal Flush (prime {prime_rf}): {lookup.flush_lookup.get(prime_rf)}") # Должно быть 1

    # Пример: Худшая рука 75432
    # Bits: 0b0000010111100 ? Нет, биты для флеша
    # Primes: 17 * 7 * 5 * 3 * 2 = 42 * 25 = 1050 * 2 = 210? Нет, 17*7*5*3*2 = 119 * 15 * 2 = 119 * 30 = 3570
    prime_75432 = CardUtils.PRIMES[5] * CardUtils.PRIMES[3] * CardUtils.PRIMES[2] * CardUtils.PRIMES[1] * CardUtils.PRIMES[0] # 7 5 4 3 2
    print(f"Rank for 75432 unsuited (prime {prime_75432}): {lookup.unsuited_lookup.get(prime_75432)}") # Должно быть 7462
