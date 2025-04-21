# src/evaluator/ofc_5card_lookup.py v1.1
# -*- coding: utf-8 -*-
"""
Генерация таблиц поиска для 5-карточного эвалуатора OFC.
Использует вариант алгоритма Cactus Kev с простыми числами.
"""

import itertools
import traceback
import sys
from typing import Dict, List, Generator

# Импорты из src пакета
try:
    from src.card import Card as CardUtils, INT_RANKS, PRIMES
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import from src.card in ofc_5card_lookup.py: {e}")
    # Заглушка
    class CardUtils:
        PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
        INT_RANKS = range(13)
        @staticmethod
        def prime_product_from_rankbits(rankbits): return 1
        @staticmethod
        def prime_product_from_hand(card_ints): return 1
    INT_RANKS = range(13)
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
    raise ImportError("Failed to import Card utilities, cannot initialize 5-card lookup table.") from e


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
    # Константы рангов (без изменений)
    MAX_STRAIGHT_FLUSH: int = 10
    MAX_FOUR_OF_A_KIND: int = 166
    MAX_FULL_HOUSE: int = 322
    MAX_FLUSH: int = 1599
    MAX_STRAIGHT: int = 1609
    MAX_THREE_OF_A_KIND: int = 2467
    MAX_TWO_PAIR: int = 3325
    MAX_PAIR: int = 6185
    MAX_HIGH_CARD: int = 7462

    MAX_TO_RANK_CLASS: Dict[int, int] = {
        MAX_STRAIGHT_FLUSH: 1, MAX_FOUR_OF_A_KIND: 2, MAX_FULL_HOUSE: 3,
        MAX_FLUSH: 4, MAX_STRAIGHT: 5, MAX_THREE_OF_A_KIND: 6,
        MAX_TWO_PAIR: 7, MAX_PAIR: 8, MAX_HIGH_CARD: 9
    }
    RANK_CLASS_TO_STRING: Dict[int, str] = {
        1: "Straight Flush", 2: "Four of a Kind", 3: "Full House", 4: "Flush",
        5: "Straight", 6: "Three of a Kind", 7: "Two Pair", 8: "Pair", 9: "High Card"
    }

    def __init__(self):
        """ Инициализирует и вычисляет таблицы поиска. """
        self.flush_lookup: Dict[int, int] = {}
        self.unsuited_lookup: Dict[int, int] = {}
        print("Initializing 5-card lookup tables...")
        self._calculate_flushes()
        self._calculate_multiples()
        print(f"Lookup tables initialized. Flush: {len(self.flush_lookup)}, Unsuited: {len(self.unsuited_lookup)}")

    def _calculate_flushes(self):
        """ Вычисляет ранги для стрит-флешей и обычных флешей. """
        straight_flushes_rank_bits: List[int] = [
            0b1111100000000, 0b0111110000000, 0b0011111000000, 0b0001111100000,
            0b0000111110000, 0b0000011111000, 0b0000001111100, 0b0000000111110,
            0b0000000011111, 0b1000000001111,
        ]
        all_flush_rank_bits: List[int] = []
        start_bits = (1 << 5) - 1
        all_flush_rank_bits.append(start_bits)
        gen = self._get_lexographically_next_bit_sequence(start_bits)
        try:
            while True:
                rank_bits = next(gen)
                all_flush_rank_bits.append(rank_bits)
        except StopIteration: pass
        except Exception as e:
            print(f"Error during bit sequence generation: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

        expected_combinations = 1287 # C(13, 5)
        if len(all_flush_rank_bits) != expected_combinations:
            print(f"CRITICAL WARNING: Generated {len(all_flush_rank_bits)} flush rank combinations, expected {expected_combinations}. Lookup table might be incomplete!", file=sys.stderr)
            # raise RuntimeError("Failed to generate the correct number of flush combinations.") # Можно раскомментировать

        straight_flush_set = set(straight_flushes_rank_bits)
        normal_flush_rank_bits = sorted([rb for rb in all_flush_rank_bits if rb not in straight_flush_set], reverse=True)
        straight_flushes_rank_bits.sort(reverse=True) # Убедимся, что они отсортированы

        rank = 1
        for sf_bits in straight_flushes_rank_bits:
            prime_product = CardUtils.prime_product_from_rankbits(sf_bits)
            self.flush_lookup[prime_product] = rank
            rank += 1

        rank = LookupTable.MAX_FULL_HOUSE + 1
        for f_bits in normal_flush_rank_bits:
            prime_product = CardUtils.prime_product_from_rankbits(f_bits)
            self.flush_lookup[prime_product] = rank
            rank += 1

        self._calculate_straights_and_highcards(straight_flushes_rank_bits, normal_flush_rank_bits)

    # (Методы _calculate_straights_and_highcards, _calculate_multiples, _get_lexographically_next_bit_sequence без изменений)
    def _calculate_straights_and_highcards(self, straights_rank_bits: List[int], highcards_rank_bits: List[int]):
        """ Вычисляет ранги для стритов и старших карт (не флеш). """
        rank = LookupTable.MAX_FLUSH + 1
        for s_bits in straights_rank_bits:
            prime_product = CardUtils.prime_product_from_rankbits(s_bits)
            self.unsuited_lookup[prime_product] = rank
            rank += 1
        rank = LookupTable.MAX_PAIR + 1
        for h_bits in highcards_rank_bits:
            prime_product = CardUtils.prime_product_from_rankbits(h_bits)
            self.unsuited_lookup[prime_product] = rank
            rank += 1

    def _calculate_multiples(self):
        """ Вычисляет ранги для Каре, Фулл-хаусов, Сетов, Двух пар и Пар. """
        backwards_ranks = range(len(INT_RANKS) - 1, -1, -1)
        # 1) Каре
        rank = LookupTable.MAX_STRAIGHT_FLUSH + 1
        for quad_rank_idx in backwards_ranks:
            kickers_indices = list(backwards_ranks); kickers_indices.remove(quad_rank_idx)
            for kicker_idx in kickers_indices:
                product = PRIMES[quad_rank_idx]**4 * PRIMES[kicker_idx]
                self.unsuited_lookup[product] = rank; rank += 1
        # 2) Фулл-хаус
        rank = LookupTable.MAX_FOUR_OF_A_KIND + 1
        for trip_rank_idx in backwards_ranks:
            pair_rank_indices = list(backwards_ranks); pair_rank_indices.remove(trip_rank_idx)
            for pair_rank_idx in pair_rank_indices:
                product = PRIMES[trip_rank_idx]**3 * PRIMES[pair_rank_idx]**2
                self.unsuited_lookup[product] = rank; rank += 1
        # 3) Сет
        rank = LookupTable.MAX_STRAIGHT + 1
        for trip_rank_idx in backwards_ranks:
            kickers_indices = list(backwards_ranks); kickers_indices.remove(trip_rank_idx)
            kicker_combos = itertools.combinations(kickers_indices, 2)
            for k1_idx, k2_idx in kicker_combos:
                product = PRIMES[trip_rank_idx]**3 * PRIMES[k1_idx] * PRIMES[k2_idx]
                self.unsuited_lookup[product] = rank; rank += 1
        # 4) Две пары
        rank = LookupTable.MAX_THREE_OF_A_KIND + 1
        two_pair_combos = itertools.combinations(backwards_ranks, 2)
        for p1_idx, p2_idx in two_pair_combos:
            kickers_indices = list(backwards_ranks); kickers_indices.remove(p1_idx); kickers_indices.remove(p2_idx)
            for kicker_idx in kickers_indices:
                product = PRIMES[p1_idx]**2 * PRIMES[p2_idx]**2 * PRIMES[kicker_idx]
                self.unsuited_lookup[product] = rank; rank += 1
        # 5) Пара
        rank = LookupTable.MAX_TWO_PAIR + 1
        for pair_rank_idx in backwards_ranks:
            kickers_indices = list(backwards_ranks); kickers_indices.remove(pair_rank_idx)
            kicker_combos = itertools.combinations(kickers_indices, 3)
            for k1_idx, k2_idx, k3_idx in kicker_combos:
                product = PRIMES[pair_rank_idx]**2 * PRIMES[k1_idx] * PRIMES[k2_idx] * PRIMES[k3_idx]
                self.unsuited_lookup[product] = rank; rank += 1

    def _get_lexographically_next_bit_sequence(self, bits: int) -> Generator[int, None, None]:
        """ Генератор, возвращающий следующую лексикографическую перестановку битов. """
        next_val = bits
        while True:
            t = (next_val | (next_val - 1)) + 1
            next_val_lsb = next_val & -next_val
            if next_val_lsb == 0: break
            next_val = t | ((((t & -t) // next_val_lsb) >> 1) - 1)
            # Ограничиваем 13 битами (максимальная маска для 5 из 13)
            if next_val == 0 or next_val >= (1 << 13): break
            yield next_val

# (Блок if __name__ == '__main__' без изменений)
if __name__ == '__main__':
    print("Generating 5-card lookup table...")
    lookup = LookupTable()
    print("Table generated.")
    prime_4a_k = CardUtils.PRIMES[12]**4 * CardUtils.PRIMES[11]
    print(f"Rank for AAAA K (prime {prime_4a_k}): {lookup.unsuited_lookup.get(prime_4a_k)}")
    prime_rf = CardUtils.prime_product_from_rankbits(0b1111100000000)
    print(f"Rank for Royal Flush (prime {prime_rf}): {lookup.flush_lookup.get(prime_rf)}")
    prime_75432 = CardUtils.PRIMES[5] * CardUtils.PRIMES[3] * CardUtils.PRIMES[2] * CardUtils.PRIMES[1] * CardUtils.PRIMES[0]
    print(f"Rank for 75432 unsuited (prime {prime_75432}): {lookup.unsuited_lookup.get(prime_75432)}")
