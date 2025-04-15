# tests/test_card.py
"""
Unit-тесты для модуля src.card.
"""

import pytest

# --- ИСПРАВЛЕНО: Импорт из src ---
from src.card import Card, card_from_str, card_to_str, INVALID_CARD, CARD_PLACEHOLDER, PRIMES, RANK_MAP, SUIT_CHAR_TO_INT

# --- Тесты для Card.from_str ---

def test_card_from_str_valid():
    """Тестирует создание карт из валидных строк."""
    assert card_from_str('As') > 0 # Просто проверяем, что результат > 0
    assert card_from_str('Td') > 0
    assert card_from_str('2c') > 0
    assert card_from_str('KH') > 0 # Проверка заглавной масти
    assert card_from_str('qs') > 0 # Проверка строчной буквы ранга

def test_card_from_str_invalid_length():
    """Тестирует ошибки при неверной длине строки."""
    with pytest.raises(ValueError):
        card_from_str('A')
    with pytest.raises(ValueError):
        card_from_str('Asd')
    with pytest.raises(ValueError):
        card_from_str('')

def test_card_from_str_invalid_rank():
    """Тестирует ошибки при неверном ранге."""
    with pytest.raises(ValueError):
        card_from_str('1s')
    with pytest.raises(ValueError):
        card_from_str('Xs')
    with pytest.raises(ValueError):
        card_from_str('Zs')

def test_card_from_str_invalid_suit():
    """Тестирует ошибки при неверной масти."""
    with pytest.raises(ValueError):
        card_from_str('Ax')
    with pytest.raises(ValueError):
        card_from_str('A1')
    with pytest.raises(ValueError):
        card_from_str('Tr')

# --- Тесты для Card.to_str ---

def test_card_to_str_valid():
    """Тестирует конвертацию валидных int карт в строки."""
    ace_spades = card_from_str('As')
    ten_diamonds = card_from_str('Td')
    two_clubs = card_from_str('2c')
    king_hearts = card_from_str('Kh')
    assert card_to_str(ace_spades) == 'As'
    assert card_to_str(ten_diamonds) == 'Td'
    assert card_to_str(two_clubs) == '2c'
    assert card_to_str(king_hearts) == 'Kh'

def test_card_to_str_invalid():
    """Тестирует конвертацию невалидных значений в строку."""
    assert card_to_str(None) == CARD_PLACEHOLDER
    assert card_to_str(INVALID_CARD) == CARD_PLACEHOLDER
    assert card_to_str(-5) == CARD_PLACEHOLDER
    assert card_to_str(0) == CARD_PLACEHOLDER # 0 не является валидной картой
    # Создадим "сломанный" int вручную (не через from_str)
    invalid_rank_int = (1 << 15 << 16) | (SUIT_CHAR_TO_INT['s'] << 12) | (15 << 8) | 99
    invalid_suit_int = (1 << 5 << 16) | (5 << 12) | (5 << 8) | PRIMES[5]
    assert card_to_str(invalid_rank_int) == CARD_PLACEHOLDER
    assert card_to_str(invalid_suit_int) == CARD_PLACEHOLDER

# --- Тесты для извлечения атрибутов ---

def test_card_getters():
    """Тестирует функции извлечения ранга, масти, битовой маски, простого числа."""
    card_int = card_from_str('Kc') # Король треф
    # Ожидаемые значения для 'Kc'
    expected_rank_int = 11 # Индекс K
    expected_suit_int = SUIT_CHAR_TO_INT['c'] # 8
    expected_rank_bitmask = 1 << expected_rank_int # 1 << 11 = 2048
    expected_prime = PRIMES[expected_rank_int] # PRIMES[11] = 37

    assert Card.get_rank_int(card_int) == expected_rank_int
    assert Card.get_suit_int(card_int) == expected_suit_int
    assert Card.get_rank_bitmask(card_int) == expected_rank_bitmask
    assert Card.get_prime(card_int) == expected_prime

    # Тест для Туза
    card_int_ace = card_from_str('As')
    assert Card.get_rank_int(card_int_ace) == 12
    assert Card.get_suit_int(card_int_ace) == SUIT_CHAR_TO_INT['s'] # 1
    assert Card.get_rank_bitmask(card_int_ace) == (1 << 12) # 4096
    assert Card.get_prime(card_int_ace) == PRIMES[12] # 41

    # Тест для Двойки
    card_int_two = card_from_str('2h')
    assert Card.get_rank_int(card_int_two) == 0
    assert Card.get_suit_int(card_int_two) == SUIT_CHAR_TO_INT['h'] # 2
    assert Card.get_rank_bitmask(card_int_two) == (1 << 0) # 1
    assert Card.get_prime(card_int_two) == PRIMES[0] # 2

# --- Тесты для работы с простыми числами ---

def test_prime_product_from_hand():
    """Тестирует вычисление произведения простых чисел для руки."""
    hand_strs = ['As', 'Kc', 'Td', '7h', '2s']
    hand_ints = Card.hand_to_int(hand_strs)
    expected_product = PRIMES[12] * PRIMES[11] * PRIMES[8] * PRIMES[5] * PRIMES[0]
    # 41 * 37 * 23 * 17 * 2 = 1185194
    assert Card.prime_product_from_hand(hand_ints) == expected_product

    # Тест с None/Invalid картами
    hand_with_invalid = [card_from_str('As'), None, card_from_str('Kc'), INVALID_CARD]
    expected_product_invalid = PRIMES[12] * PRIMES[11] # 41 * 37 = 1517
    assert Card.prime_product_from_hand(hand_with_invalid) == expected_product_invalid

def test_prime_product_from_rankbits():
    """Тестирует вычисление произведения простых чисел из битовой маски рангов."""
    # Пример: маска для стрита A2345 (Wheel) -> биты 0, 1, 2, 3, 12
    wheel_mask = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 12)
    expected_product = PRIMES[0] * PRIMES[1] * PRIMES[2] * PRIMES[3] * PRIMES[12]
    # 2 * 3 * 5 * 7 * 41 = 8610
    assert Card.prime_product_from_rankbits(wheel_mask) == expected_product

    # Пример: маска для пары двоек с кикерами K, Q, J -> биты 0, 9, 10, 11
    # (В этой функции маска показывает только уникальные ранги, не количество)
    pair_mask = (1 << 0) | (1 << 9) | (1 << 10) | (1 << 11)
    expected_product_pair = PRIMES[0] * PRIMES[9] * PRIMES[10] * PRIMES[11]
    # 2 * 29 * 31 * 37 = 66506
    assert Card.prime_product_from_rankbits(pair_mask) == expected_product_pair

# --- Тесты для конвертации списков ---

def test_hand_to_int():
    """Тестирует конвертацию списка строк в список int."""
    hand_strs = ['As', 'Td', '2c', CARD_PLACEHOLDER, 'XX', None]
    expected_ints = [card_from_str('As'), card_from_str('Td'), card_from_str('2c'), None, None, None]
    result_ints = Card.hand_to_int(hand_strs)
    # Сравниваем элементы по одному, т.к. None не имеет порядка
    assert len(result_ints) == len(expected_ints)
    for i in range(len(result_ints)):
        assert result_ints[i] == expected_ints[i]

def test_hand_to_str():
    """Тестирует конвертацию списка int в список строк."""
    hand_ints = [card_from_str('As'), card_from_str('Td'), card_from_str('2c'), None, INVALID_CARD, -10]
    expected_strs = ['As', 'Td', '2c', CARD_PLACEHOLDER, CARD_PLACEHOLDER, CARD_PLACEHOLDER]
    assert Card.hand_to_str(hand_ints) == expected_strs

# --- Тесты для красивого вывода ---

def test_to_pretty_str():
    """Тестирует красивое форматирование карт."""
    # Точный вид зависит от поддержки Unicode и termcolor в среде выполнения тестов
    # Проверяем базовый формат без цвета
    assert Card.to_pretty_str(card_from_str('As')) == '[A\u2660]' # [A♠]
    assert Card.to_pretty_str(card_from_str('Kh')) == '[K\u2665]' # [K♥]
    assert Card.to_pretty_str(card_from_str('Td')) == '[T\u2666]' # [T♦]
    assert Card.to_pretty_str(card_from_str('2c')) == '[2\u2663]' # [2♣]
    assert Card.to_pretty_str(None) == '[__]'
    assert Card.to_pretty_str(INVALID_CARD) == '[__]'
