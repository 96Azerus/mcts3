# src/evaluator/ofc_5card_evaluator.py
# -*- coding: utf-8 -*-
"""
Класс для оценки 5-карточных рук OFC (средний и нижний боксы).
Использует предрасчитанные таблицы поиска из ofc_5card_lookup.py.
"""

import itertools
from typing import List, Optional

# --- ИСПРАВЛЕНО: Импортируем CardUtils из src пакета ---
try:
    # Используем алиас для удобства
    from src.card import Card as CardUtils, INVALID_CARD, card_to_str
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import from src.card in ofc_5card_evaluator.py: {e}")
    # Заглушка
    class CardUtils:
        @staticmethod
        def get_prime(card_int): return 1
        @staticmethod
        def prime_product_from_hand(card_ints): return 1
        @staticmethod
        def prime_product_from_rankbits(rankbits): return 1
    INVALID_CARD = -1
    def card_to_str(c): return "??"
    # raise ImportError("Failed to import Card utilities, cannot initialize 5-card evaluator.") from e

# Используем относительный импорт для LookupTable внутри пакета evaluator
from .ofc_5card_lookup import LookupTable

class Evaluator:
    """
    Оценивает 5-карточные руки, используя таблицы поиска для максимальной скорости.
    Работает с целочисленными представлениями карт из src.card.
    """

    def __init__(self):
        """
        Инициализирует эвалуатор, загружая или генерируя таблицы поиска.
        """
        # Создаем экземпляр LookupTable, который сам сгенерирует таблицы
        self.table = LookupTable()

    def evaluate(self, cards: List[int]) -> int:
        """
        Основная функция для оценки 5-карточной руки.

        Args:
            cards (List[int]): Список из 5 целочисленных представлений карт.
                               Карты должны быть валидными (не INVALID_CARD).

        Returns:
            int: Ранг руки в диапазоне [1, 7462]. Меньшее значение соответствует
                 более сильной руке. Возвращает очень большое значение при ошибке.

        Raises:
            ValueError: Если количество карт не равно 5 или карты невалидны.
            TypeError: Если элементы списка не являются целыми числами.
        """
        if len(cards) != 5:
            raise ValueError(f"OFC 5-card evaluator requires exactly 5 cards, received {len(cards)}.")

        valid_cards = []
        for i, c in enumerate(cards):
             if not isinstance(c, int):
                  raise TypeError(f"Card {i+1} is not an integer (type: {type(c)}). Expected int representation.")
             if c == INVALID_CARD or c < 0:
                  raise ValueError(f"Card {i+1} is invalid ({c}). Hand: {[card_to_str(card) for card in cards]}")
             valid_cards.append(c)

        # Проверка на флеш (все 5 карт имеют одинаковую масть)
        # Масть хранится в битах 12-15. Побитовое И (&) для всех карт
        # оставит эти биты установленными, только если они одинаковы у всех.
        suit_mask = valid_cards[0] & valid_cards[1] & valid_cards[2] & valid_cards[3] & valid_cards[4] & 0xF000

        if suit_mask != 0: # Если suit_mask не ноль, значит масть у всех карт одинаковая
            # Это флеш (или стрит-флеш)
            # Вычисляем битовую маску рангов всех карт в руке
            rank_bitmask = (valid_cards[0] | valid_cards[1] | valid_cards[2] | valid_cards[3] | valid_cards[4]) >> 16
            # Находим произведение простых чисел для этой маски рангов
            prime_product = CardUtils.prime_product_from_rankbits(rank_bitmask)
            # Ищем ранг в таблице флешей
            rank = self.table.flush_lookup.get(prime_product)
            if rank is None:
                 # Это не должно происходить для валидных флешей, но на всякий случай
                 print(f"Warning: Flush prime product {prime_product} not found in flush_lookup. Hand: {[card_to_str(vc) for vc in valid_cards]}")
                 # Попробуем найти как обычную руку (хотя это будет неверно)
                 prime_unsuited = CardUtils.prime_product_from_hand(valid_cards)
                 rank = self.table.unsuited_lookup.get(prime_unsuited, LookupTable.MAX_HIGH_CARD + 1)
            return rank
        else:
            # Это не флеш
            # Вычисляем произведение простых чисел рангов карт
            prime_product = CardUtils.prime_product_from_hand(valid_cards)
            # Ищем ранг в таблице для не-флешовых рук
            # Если ключ не найден (ошибка), возвращаем ранг хуже худшей руки
            return self.table.unsuited_lookup.get(prime_product, LookupTable.MAX_HIGH_CARD + 1)

    def get_rank_class(self, hand_rank: int) -> int:
        """
        Возвращает класс руки (1-9) по её рангу.

        Args:
            hand_rank (int): Ранг руки, полученный от evaluate().

        Returns:
            int: Класс руки (1: Стрит-флеш, ..., 9: Старшая карта).
                 Возвращает 9 (Старшая карта) для невалидных рангов.
        """
        # Используем константы из LookupTable для определения класса
        if hand_rank >= 1 and hand_rank <= LookupTable.MAX_STRAIGHT_FLUSH:
            return 1 # Straight Flush
        elif hand_rank <= LookupTable.MAX_FOUR_OF_A_KIND:
            return 2 # Four of a Kind
        elif hand_rank <= LookupTable.MAX_FULL_HOUSE:
            return 3 # Full House
        elif hand_rank <= LookupTable.MAX_FLUSH:
            return 4 # Flush
        elif hand_rank <= LookupTable.MAX_STRAIGHT:
            return 5 # Straight
        elif hand_rank <= LookupTable.MAX_THREE_OF_A_KIND:
            return 6 # Three of a Kind
        elif hand_rank <= LookupTable.MAX_TWO_PAIR:
            return 7 # Two Pair
        elif hand_rank <= LookupTable.MAX_PAIR:
            return 8 # Pair
        elif hand_rank <= LookupTable.MAX_HIGH_CARD:
            return 9 # High Card
        else:
            # Обработка невалидного ранга
            print(f"Warning: Invalid hand rank {hand_rank} in get_rank_class. Returning High Card class.")
            return 9 # Возвращаем класс Старшей карты по умолчанию

    def class_to_string(self, class_int: int) -> str:
        """
        Преобразует целочисленный класс руки в строку.

        Args:
            class_int (int): Класс руки (1-9).

        Returns:
            str: Строковое представление класса руки (например, "Straight Flush", "Pair").
                 Возвращает "Unknown" для невалидных классов.
        """
        # Используем словарь из LookupTable
        return LookupTable.RANK_CLASS_TO_STRING.get(class_int, "Unknown")

# Пример использования внутри модуля (для тестирования)
if __name__ == '__main__':
    print("--- Тестирование 5-Card Evaluator ---")
    evaluator = Evaluator()

    test_hands_str = [
        ['As', 'Ks', 'Qs', 'Js', 'Ts'], # Royal Flush
        ['9d', '8d', '7d', '6d', '5d'], # Straight Flush
        ['Ac', 'Ad', 'Ah', 'As', '2c'], # Four of a Kind (Aces)
        ['2c', '2d', '2h', '2s', 'Ac'], # Four of a Kind (Twos)
        ['Kc', 'Kd', 'Kh', 'Qc', 'Qs'], # Full House (K over Q)
        ['2c', '2d', '2h', 'Ac', 'As'], # Full House (2 over A)
        ['As', 'Qs', '8s', '5s', '3s'], # Flush (Ace high)
        ['Ad', 'Kc', 'Qh', 'Js', 'Td'], # Straight (Ace high)
        ['5d', '4c', '3h', '2s', 'Ad'], # Straight (5 high - Wheel)
        ['Ac', 'Ad', 'Ah', 'Ks', 'Qd'], # Three of a Kind (Aces)
        ['2c', '2d', '2h', '3s', '4d'], # Three of a Kind (Twos)
        ['Ac', 'Ad', 'Kc', 'Kd', '2s'], # Two Pair (Aces and Kings)
        ['3c', '3d', '2c', '2d', 'As'], # Two Pair (Threes and Twos)
        ['Ac', 'Ad', 'Ks', 'Qd', 'Jc'], # Pair (Aces)
        ['2c', '2d', '3s', '4d', '5c'], # Pair (Twos)
        ['Ac', 'Kc', 'Qs', 'Js', '9d'], # High Card (Ace high)
        ['7d', '5c', '4h', '3s', '2d'], # High Card (7 high - Worst hand)
    ]

    results = []
    for i, hand_str in enumerate(test_hands_str):
        try:
            # Преобразуем строки в int
            hand_int = [CardUtils.from_str(c) for c in hand_str]
            rank = evaluator.evaluate(hand_int)
            rank_class = evaluator.get_rank_class(rank)
            class_str = evaluator.class_to_string(rank_class)
            print(f"Тест {i+1}: {' '.join(hand_str)} -> Ранг: {rank}, Класс: {class_str} ({rank_class})")
            results.append((rank, rank_class))
        except Exception as e:
            print(f"Ошибка при тесте {i+1} {' '.join(hand_str)}: {e}")
            traceback.print_exc()

    if results:
         print("\n--- Проверка порядка рангов ---")
         is_sorted = all(results[i][0] <= results[i+1][0] for i in range(len(results)-1))
         print(f"Ранги отсортированы по возрастанию (сильнее -> слабее): {is_sorted}")
         if not is_sorted:
              print("Полученные ранги:", [r[0] for r in results])

         print("\n--- Проверка классов рук ---")
         expected_classes = [1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]
         actual_classes = [r[1] for r in results]
         print(f"Полученные классы: {actual_classes}")
         print(f"Ожидаемые классы: {expected_classes}")
         print(f"Совпадают: {actual_classes == expected_classes}")

         if is_sorted and actual_classes == expected_classes:
              print("\nЭвалуатор 5-карточных рук работает корректно.")
         else:
              print("\nВНИМАНИЕ: Обнаружены несоответствия в рангах или классах!")
