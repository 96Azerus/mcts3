# src/evaluator/ofc_3card_evaluator.py
# -*- coding: utf-8 -*-
"""
Функция для оценки 3-карточной руки OFC (верхний бокс).
Использует предрасчитанную таблицу поиска.
"""

import traceback
from typing import Tuple

# --- ИСПРАВЛЕНО: Импортируем таблицу и Card из правильных мест ---
from .ofc_3card_lookup import three_card_lookup
# Используем прямой импорт из src пакета
from src.card import Card as RootCardUtils, INVALID_CARD, card_to_str


def evaluate_3_card_ofc(card1: int, card2: int, card3: int) -> Tuple[int, str, str]:
    """
    Оценивает 3-карточную руку по правилам OFC, используя предрасчитанную таблицу.
    Карты должны быть целочисленными представлениями из src.card.

    Args:
        card1 (int): Первая карта.
        card2 (int): Вторая карта.
        card3 (int): Третья карта.

    Returns:
        Tuple[int, str, str]: Кортеж (rank, type_string, rank_string).
                              Меньший ранг соответствует более сильной руке.
                              rank: 1 (AAA) - 455 (432 разномастные).
                              type_string: 'Trips', 'Pair', 'High Card'.
                              rank_string: Строка рангов карт (отсортированная, например, 'AAK', 'KQJ').

    Raises:
        ValueError: Если переданы некорректные карты (None, INVALID_CARD) или их количество не равно 3.
        TypeError: Если переданные значения не являются целыми числами.
    """
    ranks = []
    input_cards = [card1, card2, card3]

    if len(input_cards) != 3:
        raise ValueError("Должно быть передано ровно 3 карты")

    for i, card_int in enumerate(input_cards):
        if not isinstance(card_int, int):
            raise TypeError(f"Карта {i+1} не является целым числом (получен тип {type(card_int)}).")
        if card_int == INVALID_CARD:
             raise ValueError(f"Карта {i+1} невалидна (INVALID_CARD).")
        if card_int < 0: # Дополнительная проверка на отрицательные значения
             raise ValueError(f"Карта {i+1} имеет некорректное значение {card_int}.")

        try:
            rank_int = RootCardUtils.get_rank_int(card_int)
            # Проверяем, что ранг находится в допустимом диапазоне 0-12
            if 0 <= rank_int <= 12:
                ranks.append(rank_int)
            else:
                raise ValueError(f"Неверный ранг {rank_int}, извлеченный из карты int: {card_int} ({card_to_str(card_int)})")
        except Exception as e:
            # Перехватываем другие возможные ошибки при обработке карты
            raise ValueError(f"Ошибка при обработке карты int {card_int} ({card_to_str(card_int)}): {e}")

    # Сортируем ранги по убыванию для ключа поиска
    # Кортеж используется как ключ словаря, так как он неизменяемый
    lookup_key = tuple(sorted(ranks, reverse=True))

    # Ищем результат в таблице
    result = three_card_lookup.get(lookup_key)

    if result is None:
        # Эта ситуация не должна возникать, если таблица полная и карты валидные
        raise ValueError(f"Не найдена комбинация для ключа рангов: {lookup_key} (исходные ранги: {ranks})")

    # Возвращаем найденный кортеж (rank, type_string, rank_string)
    return result

# Пример использования внутри модуля (для тестирования)
if __name__ == '__main__':
    # Тесты используют RootCardUtils, который теперь импортируется из src.card
    print("--- Тестирование evaluate_3_card_ofc ---")
    test_hands_str = [
        ('Ah', 'Ad', 'As'), ('Qh', 'Qs', '2d'), ('Kd', '5s', '2h'),
        ('6c', '6d', 'Ts'), ('2h', '2d', '2s'), ('Jd', 'Th', '9s'),
        ('5h', '3d', '2c'), ('Ac', 'Kc', 'Qc') # Добавим старшую карту
    ]
    results = []
    expected_ranks = [1, 40, 287, 114, 13, 336, 455, 170] # Ожидаемые ранги для тестов

    for i, hand_str in enumerate(test_hands_str):
        try:
            # Преобразуем строки в int
            hand_int = tuple(RootCardUtils.from_str(c) for c in hand_str)
            rank, type_str, rank_str = evaluate_3_card_ofc(*hand_int)
            print(f"Тест {i+1}: {hand_str} -> Ранг: {rank}, Тип: {type_str}, Строка: {rank_str}")
            results.append(rank)
        except Exception as e:
            print(f"Ошибка при тесте {i+1} {hand_str}: {e}")
            traceback.print_exc()

    if results:
        print("\n--- Проверка результатов ---")
        print(f"Полученные ранги: {results}")
        print(f"Ожидаемые ранги: {expected_ranks}")
        correct_count = sum(1 for i, r in enumerate(results) if i < len(expected_ranks) and r == expected_ranks[i])
        print(f"Совпадений рангов: {correct_count} из {len(expected_ranks)}")

        if correct_count == len(expected_ranks):
             print("\nФункция оценки 3-карточных рук работает корректно.")
        else:
             print("\nВНИМАНИЕ: Обнаружены несоответствия в рангах!")
