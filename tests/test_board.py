# tests/test_board.py
"""
Unit-тесты для модуля src.board.
"""

import pytest

# Импорты из src пакета
from src.board import PlayerBoard
from src.card import card_from_str, card_to_str, INVALID_CARD, CARD_PLACEHOLDER
# Импортируем функции скоринга для проверки результатов
from src.scoring import get_row_royalty, check_board_foul, get_fantasyland_entry_cards, check_fantasyland_stay

# Хелпер для создания рук
def hand(card_strs):
    return [card_from_str(s) if s else None for s in card_strs]

# --- Тесты инициализации ---

def test_board_init():
    """Тестирует начальное состояние доски."""
    board = PlayerBoard()
    assert board.get_total_cards() == 0
    assert not board.is_complete()
    assert not board.is_foul
    assert len(board.get_available_slots()) == 13 # 3 + 5 + 5
    for row_name in PlayerBoard.ROW_NAMES:
        assert len(board.rows[row_name]) == PlayerBoard.ROW_CAPACITY[row_name]
        assert all(card is None for card in board.rows[row_name])
        assert board._get_rank(row_name) > 7462 # Ранг пустой руки должен быть плохим
    assert board.get_total_royalty() == 0

# --- Тесты добавления/удаления карт ---

def test_board_add_card_valid():
    """Тестирует добавление валидной карты в пустой слот."""
    board = PlayerBoard()
    card_as = card_from_str('As')
    assert board.add_card(card_as, 'top', 0)
    assert board.get_total_cards() == 1
    assert board.rows['top'][0] == card_as
    assert board.rows['top'][1] is None
    assert board._cached_ranks['top'] is None # Кэш должен сброситься
    assert board._cached_royalties['top'] is None

    card_kd = card_from_str('Kd')
    assert board.add_card(card_kd, 'bottom', 4)
    assert board.get_total_cards() == 2
    assert board.rows['bottom'][4] == card_kd

def test_board_add_card_invalid_slot():
    """Тестирует добавление карты в занятый или неверный слот."""
    board = PlayerBoard()
    card_as = card_from_str('As')
    card_ks = card_from_str('Ks')
    board.add_card(card_as, 'top', 0)

    # Попытка добавить в занятый слот
    assert not board.add_card(card_ks, 'top', 0)
    assert board.get_total_cards() == 1
    assert board.rows['top'][0] == card_as # Карта не должна измениться

    # Попытка добавить в неверный индекс
    assert not board.add_card(card_ks, 'top', 3)
    assert board.get_total_cards() == 1

    # Попытка добавить в неверный ряд
    assert not board.add_card(card_ks, 'upper', 0)
    assert board.get_total_cards() == 1

def test_board_add_card_invalid_card():
    """Тестирует добавление невалидной карты."""
    board = PlayerBoard()
    assert not board.add_card(INVALID_CARD, 'top', 0)
    assert board.get_total_cards() == 0
    # assert not board.add_card(None, 'top', 0) # None не должен передаваться

def test_board_remove_card_valid():
    """Тестирует удаление карты из слота."""
    board = PlayerBoard()
    card_as = card_from_str('As')
    board.add_card(card_as, 'middle', 2)
    assert board.get_total_cards() == 1
    board._get_rank('middle') # Заполняем кэш
    assert board._cached_ranks['middle'] is not None

    removed_card = board.remove_card('middle', 2)
    assert removed_card == card_as
    assert board.get_total_cards() == 0
    assert board.rows['middle'][2] is None
    assert board._cached_ranks['middle'] is None # Кэш должен сброситься

def test_board_remove_card_invalid():
    """Тестирует удаление из пустого или неверного слота."""
    board = PlayerBoard()
    card_as = card_from_str('As')
    board.add_card(card_as, 'middle', 2)

    # Удаление из пустого слота
    assert board.remove_card('middle', 0) is None
    assert board.get_total_cards() == 1

    # Удаление из неверного индекса
    assert board.remove_card('middle', 5) is None
    assert board.get_total_cards() == 1

    # Удаление из неверного ряда
    assert board.remove_card('center', 2) is None
    assert board.get_total_cards() == 1

# --- Тесты установки полной доски ---

def test_board_set_full_board_valid():
    """Тестирует установку валидной полной доски."""
    board = PlayerBoard()
    top = hand(['Ah', 'Ad', 'Ac'])
    middle = hand(['Ks', 'Kd', 'Qc', 'Qd', '2s'])
    bottom = hand(['As', 'Ks', 'Qs', 'Js', 'Ts'])
    board.set_full_board(top, middle, bottom)

    assert board.is_complete()
    assert board.get_total_cards() == 13
    assert board.rows['top'] == top
    assert board.rows['middle'] == middle
    assert board.rows['bottom'] == bottom
    assert board.is_foul # Эта доска - фол
    assert board._cached_ranks['top'] is None # Кэши сброшены
    assert board.get_total_royalty() == 0 # Роялти 0 из-за фола

def test_board_set_full_board_invalid_counts():
    """Тестирует установку доски с неверным количеством карт."""
    board = PlayerBoard()
    top = hand(['Ah', 'Ad']) # Не 3
    middle = hand(['Ks', 'Kd', 'Qc', 'Qd', '2s'])
    bottom = hand(['As', 'Ks', 'Qs', 'Js', 'Ts'])
    with pytest.raises(ValueError):
        board.set_full_board(top, middle, bottom)

def test_board_set_full_board_duplicates():
    """Тестирует установку доски с дубликатами карт."""
    board = PlayerBoard()
    top = hand(['Ah', 'Ad', 'Ac'])
    middle = hand(['Ks', 'Kd', 'Qc', 'Qd', 'Ah']) # Дубликат Ah
    bottom = hand(['As', 'Ks', 'Qs', 'Js', 'Ts'])
    with pytest.raises(ValueError):
        board.set_full_board(top, middle, bottom)

def test_board_set_full_board_invalid_cards():
    """Тестирует установку доски с невалидными картами."""
    board = PlayerBoard()
    top = hand(['Ah', 'Ad', 'Ac'])
    middle = hand(['Ks', 'Kd', 'Qc', 'Qd', INVALID_CARD]) # Невалидная карта
    bottom = hand(['As', 'Ks', 'Qs', 'Js', 'Ts'])
    with pytest.raises(ValueError):
        board.set_full_board(top, middle, bottom)

# --- Тесты вспомогательных методов ---

def test_board_helpers():
    """Тестирует is_row_full, get_available_slots и т.д."""
    board = PlayerBoard()
    assert len(board.get_available_slots()) == 13
    assert not board.is_row_full('top')

    board.add_card(card_from_str('As'), 'top', 0)
    board.add_card(card_from_str('Ks'), 'top', 1)
    board.add_card(card_from_str('Qs'), 'top', 2)
    assert board.is_row_full('top')
    assert len(board.get_available_slots()) == 10 # 13 - 3
    assert board.get_total_cards() == 3
    assert not board.is_complete()

    # Заполняем остальные ряды
    mid = hand(['2c','3c','4c','5c','6c'])
    bot = hand(['7d','8d','9d','Td','Jd'])
    for i, c in enumerate(mid): board.add_card(c, 'middle', i)
    for i, c in enumerate(bot): board.add_card(c, 'bottom', i)

    assert board.is_complete()
    assert board.get_total_cards() == 13
    assert len(board.get_available_slots()) == 0
    assert board.is_row_full('middle')
    assert board.is_row_full('bottom')

# --- Тесты получения рангов и роялти ---

def test_board_get_rank_and_royalty():
    """Тестирует получение рангов и роялти (делегирование в scoring)."""
    board = PlayerBoard()
    # Ставим Фулл-хаус на боттом
    bottom = hand(['Ac', 'Ad', 'Ah', 'Ks', 'Kd'])
    for i, c in enumerate(bottom): board.add_card(c, 'bottom', i)
    # Ставим Флеш на миддл
    middle = hand(['2h', '5h', '8h', 'Th', 'Qh'])
    for i, c in enumerate(middle): board.add_card(c, 'middle', i)
    # Ставим пару дам на топ
    top = hand(['Qc', 'Qd', '2s'])
    for i, c in enumerate(top): board.add_card(c, 'top', i)

    assert board.is_complete()
    assert not board.check_and_set_foul() # Проверяем, что не фол

    # Проверяем ранги (точные значения зависят от эвалуаторов)
    rank_t = board._get_rank('top')
    rank_m = board._get_rank('middle')
    rank_b = board._get_rank('bottom')
    assert rank_b < rank_m < rank_t # Проверка порядка

    # Проверяем роялти
    expected_royalty_t = 7 # QQ
    expected_royalty_m = 8 # Flush
    expected_royalty_b = 6 # Full House
    royalties = board.get_royalties()
    assert royalties['top'] == expected_royalty_t
    assert royalties['middle'] == expected_royalty_m
    assert royalties['bottom'] == expected_royalty_b
    assert board.get_total_royalty() == expected_royalty_t + expected_royalty_m + expected_royalty_b

    # Проверяем кэширование (второй вызов должен вернуть то же самое)
    assert board.get_royalties() == royalties
    assert board._get_rank('top') == rank_t

    # Проверяем роялти для фол-руки
    board_foul = PlayerBoard()
    top_f = hand(['Ah', 'Ad', 'Ac']) # Trips
    mid_f = hand(['Ks', 'Kd', 'Qc', 'Qd', '2s']) # Two Pair
    bot_f = hand(['As', 'Ks', 'Qs', 'Js', 'Ts']) # RF
    board_foul.set_full_board(top_f, mid_f, bot_f)
    assert board_foul.is_foul
    assert board_foul.get_royalties() == {'top': 0, 'middle': 0, 'bottom': 0}
    assert board_foul.get_total_royalty() == 0

# --- Тесты Fantasyland ---

def test_board_fantasyland_methods():
    """Тестирует методы, связанные с Fantasyland."""
    # Доска для входа в ФЛ (QQ)
    board_qq = PlayerBoard()
    board_qq.set_full_board(hand(['Qc','Qd','2s']), hand(['3h','4h','5h','6h','7h']), hand(['Ac','Ad','Ah','Ks','Kd']))
    assert not board_qq.is_foul
    assert board_qq.get_fantasyland_qualification_cards() == 14 # QQ -> 14 карт
    assert not board_qq.check_fantasyland_stay_conditions() # Нет условий для Re-FL

    # Доска для входа в ФЛ (AAA)
    board_aaa = PlayerBoard()
    board_aaa.set_full_board(hand(['Ac','Ad','Ah']), hand(['Ks','Kd','Qc','Qd','2s']), hand(['As','Ks','Qs','Js','Ts']))
    assert not board_aaa.is_foul
    assert board_aaa.get_fantasyland_qualification_cards() == 17 # AAA -> 17 карт
    assert board_aaa.check_fantasyland_stay_conditions() # Trips on top -> Stay

    # Доска для Re-FL (Quads on bottom)
    board_quads = PlayerBoard()
    board_quads.set_full_board(hand(['Kh','Qd','2c']), hand(['Ac','Kd','Qh','Js','9d']), hand(['7h','7d','7c','7s','Ad']))
    assert not board_quads.is_foul
    assert board_quads.get_fantasyland_qualification_cards() == 0 # Нет QQ+ на топе
    assert board_quads.check_fantasyland_stay_conditions() # Quads on bottom -> Stay

    # Фол доска - нет ФЛ
    board_foul = PlayerBoard()
    board_foul.set_full_board(hand(['Ah','Ad','Ac']), hand(['Ks','Kd','Qc','Qd','2s']), hand(['As','Ks','Qs','Js','Ts']))
    assert board_foul.is_foul
    assert board_foul.get_fantasyland_qualification_cards() == 0
    assert not board_foul.check_fantasyland_stay_conditions()

# --- Тесты копирования ---

def test_board_copy():
    """Тестирует копирование доски."""
    board1 = PlayerBoard()
    board1.add_card(card_from_str('As'), 'top', 0)
    board1.add_card(card_from_str('Kd'), 'bottom', 1)
    board1._get_rank('top') # Заполняем кэш

    board2 = board1.copy()

    assert board1 is not board2
    assert board1.rows is not board2.rows
    assert board1.rows['top'] is not board2.rows['top'] # Списки должны быть разными объектами
    assert board1.rows == board2.rows # Но содержимое одинаковое
    assert board1._cards_placed == board2._cards_placed
    assert board1.is_foul == board2.is_foul
    assert board1._cached_ranks == board2._cached_ranks # Кэши копируются

    # Изменяем копию
    board2.add_card(card_from_str('Qh'), 'middle', 0)
    assert board1.get_total_cards() == 2
    assert board2.get_total_cards() == 3
    assert board1.rows['middle'][0] is None
    assert board2.rows['middle'][0] == card_from_str('Qh')

    # Изменяем оригинал
    board1.remove_card('top', 0)
    assert board1.get_total_cards() == 1
    assert board2.get_total_cards() == 3
    assert board1.rows['top'][0] is None
    assert board2.rows['top'][0] == card_from_str('As')

# --- Тест get_board_state_tuple ---
def test_get_board_state_tuple():
    """Тестирует создание каноничного кортежа состояния доски."""
    board = PlayerBoard()
    # Неполная доска
    board.add_card(card_from_str('As'), 'top', 0)
    board.add_card(card_from_str('2c'), 'top', 2)
    board.add_card(card_from_str('Kd'), 'middle', 1)
    board.add_card(card_from_str('Qh'), 'middle', 3)
    board.add_card(card_from_str('Ts'), 'bottom', 4)

    state_tuple = board.get_board_state_tuple()
    # Проверяем структуру
    assert isinstance(state_tuple, tuple)
    assert len(state_tuple) == 3
    assert all(isinstance(row_tuple, tuple) for row_tuple in state_tuple)
    # Проверяем содержимое и сортировку (As > 2c)
    assert state_tuple[0] == ('As', '2c', CARD_PLACEHOLDER)
    # Проверяем содержимое и сортировку (Kd > Qh)
    assert state_tuple[1] == ('Kd', 'Qh', CARD_PLACEHOLDER, CARD_PLACEHOLDER, CARD_PLACEHOLDER)
    # Проверяем содержимое
    assert state_tuple[2] == ('Ts', CARD_PLACEHOLDER, CARD_PLACEHOLDER, CARD_PLACEHOLDER, CARD_PLACEHOLDER)

    # Полная доска
    board_full = PlayerBoard()
    board_full.set_full_board(hand(['Qc','Qd','2s']), hand(['3h','4h','5h','6h','7h']), hand(['Ac','Ad','Ah','Ks','Kd']))
    state_tuple_full = board_full.get_board_state_tuple()
    # Проверяем сортировку
    assert state_tuple_full[0] == ('Qd', 'Qc', '2s') # Сортировка по масти при равном ранге
    assert state_tuple_full[1] == ('7h', '6h', '5h', '4h', '3h')
    assert state_tuple_full[2] == ('As', 'Ah', 'Ad', 'Ks', 'Kd')
