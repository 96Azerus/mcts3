# tests/test_board.py v1.1
"""
Unit-тесты для модуля src.board.
"""
import pytest
from src.board import PlayerBoard
# --- ИСПРАВЛЕНО: Импортируем Card для Card.hand_to_int ---
from src.card import Card, card_from_str, card_to_str, INVALID_CARD, CARD_PLACEHOLDER
from src.scoring import get_row_royalty, check_board_foul, get_fantasyland_entry_cards, check_fantasyland_stay

# --- ИСПРАВЛЕНО: Обновленный хелпер hand ---
def hand(card_strs):
    """Преобразует список строк карт в список int, обрабатывая None и INVALID_CARD."""
    result = []
    for s in card_strs:
        if isinstance(s, str) and s and s != CARD_PLACEHOLDER: # Добавлена проверка на CARD_PLACEHOLDER
            try:
                result.append(card_from_str(s))
            except ValueError:
                result.append(None) # Ошибка парсинга -> None
        # --- ИСПРАВЛЕНО: INVALID_CARD и None должны стать None ---
        elif s == INVALID_CARD or s is None:
             result.append(None)
        else:
             # Обработка CARD_PLACEHOLDER или других нестроковых значений
             result.append(None)
    return result

# --- Тесты инициализации ---
def test_board_init():
    board = PlayerBoard()
    assert board.get_total_cards() == 0; assert not board.is_complete(); assert not board.is_foul
    assert len(board.get_available_slots()) == 13
    for row_name in PlayerBoard.ROW_NAMES:
        assert len(board.rows[row_name]) == PlayerBoard.ROW_CAPACITY[row_name]
        assert all(card is None for card in board.rows[row_name])
        # --- ИСПРАВЛЕНО: Проверяем ранг относительно худшего для 3 или 5 карт ---
        if row_name == 'top': assert board._get_rank(row_name) > 455
        else: assert board._get_rank(row_name) > 7462
    assert board.get_total_royalty() == 0

# (Тесты add/remove без изменений)
def test_board_add_card_valid():
    board = PlayerBoard(); card_as = card_from_str('As'); assert board.add_card(card_as, 'top', 0); assert board.get_total_cards() == 1; assert board.rows['top'][0] == card_as; assert board.rows['top'][1] is None; assert board._cached_ranks['top'] is None; assert board._cached_royalties['top'] is None; card_kd = card_from_str('Kd'); assert board.add_card(card_kd, 'bottom', 4); assert board.get_total_cards() == 2; assert board.rows['bottom'][4] == card_kd
def test_board_add_card_invalid_slot():
    board = PlayerBoard(); card_as = card_from_str('As'); card_ks = card_from_str('Ks'); board.add_card(card_as, 'top', 0); assert not board.add_card(card_ks, 'top', 0); assert board.get_total_cards() == 1; assert board.rows['top'][0] == card_as; assert not board.add_card(card_ks, 'top', 3); assert board.get_total_cards() == 1; assert not board.add_card(card_ks, 'upper', 0); assert board.get_total_cards() == 1
def test_board_add_card_invalid_card():
    board = PlayerBoard(); assert not board.add_card(INVALID_CARD, 'top', 0); assert board.get_total_cards() == 0
def test_board_remove_card_valid():
    board = PlayerBoard(); card_as = card_from_str('As'); board.add_card(card_as, 'middle', 2); assert board.get_total_cards() == 1; board._get_rank('middle'); assert board._cached_ranks['middle'] is not None; removed_card = board.remove_card('middle', 2); assert removed_card == card_as; assert board.get_total_cards() == 0; assert board.rows['middle'][2] is None; assert board._cached_ranks['middle'] is None
def test_board_remove_card_invalid():
    board = PlayerBoard(); card_as = card_from_str('As'); board.add_card(card_as, 'middle', 2); assert board.remove_card('middle', 0) is None; assert board.get_total_cards() == 1; assert board.remove_card('middle', 5) is None; assert board.get_total_cards() == 1; assert board.remove_card('center', 2) is None; assert board.get_total_cards() == 1

# --- Тесты установки полной доски ---
def test_board_set_full_board_valid():
    board = PlayerBoard()
    # --- ИСПРАВЛЕНО: Убран дубликат Ks ---
    top = hand(['Ah', 'Ad', 'Ac'])
    middle = hand(['Ks', 'Kd', 'Qc', 'Qd', '2s']) # Был Ks
    bottom = hand(['As', 'Kh', 'Qs', 'Js', 'Ts']) # Заменили Ks на Kh
    board.set_full_board(top, middle, bottom)
    assert board.is_complete(); assert board.get_total_cards() == 13
    assert board.rows['top'] == top; assert board.rows['middle'] == middle; assert board.rows['bottom'] == bottom
    # --- ИСПРАВЛЕНО: Эта доска теперь не фол ---
    assert not board.is_foul
    assert board._cached_ranks['top'] is None
    # --- ИСПРАВЛЕНО: Роялти теперь должны считаться ---
    expected_royalty = 22 + 0 + 25 # AAA + TwoPair + RF
    assert board.get_total_royalty() == expected_royalty

def test_board_set_full_board_invalid_counts():
    board = PlayerBoard(); top = hand(['Ah', 'Ad']); middle = hand(['Ks', 'Kd', 'Qc', 'Qd', '2s']); bottom = hand(['As', 'Ks', 'Qs', 'Js', 'Ts']);
    with pytest.raises(ValueError): board.set_full_board(top, middle, bottom)
def test_board_set_full_board_duplicates():
    board = PlayerBoard(); top = hand(['Ah', 'Ad', 'Ac']); middle = hand(['Ks', 'Kd', 'Qc', 'Qd', 'Ah']); bottom = hand(['As', 'Ks', 'Qs', 'Js', 'Ts']);
    with pytest.raises(ValueError): board.set_full_board(top, middle, bottom)
def test_board_set_full_board_invalid_cards():
    board = PlayerBoard(); top = hand(['Ah', 'Ad', 'Ac']);
    # --- ИСПРАВЛЕНО: Передаем INVALID_CARD в hand ---
    middle = hand(['Ks', 'Kd', 'Qc', 'Qd', INVALID_CARD])
    bottom = hand(['As', 'Ks', 'Qs', 'Js', 'Ts']);
    with pytest.raises(ValueError): board.set_full_board(top, middle, bottom)

# (Тесты helpers, copy без изменений)
def test_board_helpers():
    board = PlayerBoard(); assert len(board.get_available_slots()) == 13; assert not board.is_row_full('top'); board.add_card(card_from_str('As'), 'top', 0); board.add_card(card_from_str('Ks'), 'top', 1); board.add_card(card_from_str('Qs'), 'top', 2); assert board.is_row_full('top'); assert len(board.get_available_slots()) == 10; assert board.get_total_cards() == 3; assert not board.is_complete(); mid = hand(['2c','3c','4c','5c','6c']); bot = hand(['7d','8d','9d','Td','Jd']);
    for i, c in enumerate(mid): board.add_card(c, 'middle', i)
    for i, c in enumerate(bot): board.add_card(c, 'bottom', i)
    assert board.is_complete(); assert board.get_total_cards() == 13; assert len(board.get_available_slots()) == 0; assert board.is_row_full('middle'); assert board.is_row_full('bottom')
def test_board_copy():
    board1 = PlayerBoard(); board1.add_card(card_from_str('As'), 'top', 0); board1.add_card(card_from_str('Kd'), 'bottom', 1); board1._get_rank('top'); board2 = board1.copy(); assert board1 is not board2; assert board1.rows is not board2.rows; assert board1.rows['top'] is not board2.rows['top']; assert board1.rows == board2.rows; assert board1._cards_placed == board2._cards_placed; assert board1.is_foul == board2.is_foul; assert board1._cached_ranks == board2._cached_ranks; board2.add_card(card_from_str('Qh'), 'middle', 0); assert board1.get_total_cards() == 2; assert board2.get_total_cards() == 3; assert board1.rows['middle'][0] is None; assert board2.rows['middle'][0] == card_from_str('Qh'); board1.remove_card('top', 0); assert board1.get_total_cards() == 1; assert board2.get_total_cards() == 3; assert board1.rows['top'][0] is None; assert board2.rows['top'][0] == card_from_str('As')

# --- Тесты получения рангов и роялти ---
def test_board_get_rank_and_royalty():
    board = PlayerBoard()
    bottom = hand(['Ac', 'Ad', 'Ah', 'Ks', 'Kd']) # FH
    middle = hand(['2h', '5h', '8h', 'Th', 'Qh']) # Flush
    top = hand(['Qc', 'Qd', '2s']) # Pair Q
    board.set_full_board(top, middle, bottom)
    assert board.is_complete()
    # --- ИСПРАВЛЕНО: Теперь не фол ---
    assert not board.check_and_set_foul()
    rank_t = board._get_rank('top'); rank_m = board._get_rank('middle'); rank_b = board._get_rank('bottom')
    assert rank_b < rank_m < rank_t # Проверка порядка
    expected_royalty_t = 7; expected_royalty_m = 8; expected_royalty_b = 6
    royalties = board.get_royalties()
    assert royalties['top'] == expected_royalty_t; assert royalties['middle'] == expected_royalty_m; assert royalties['bottom'] == expected_royalty_b
    assert board.get_total_royalty() == expected_royalty_t + expected_royalty_m + expected_royalty_b
    assert board.get_royalties() == royalties; assert board._get_rank('top') == rank_t
    # Проверяем роялти для фол-руки (создаем фол вручную)
    board_foul = PlayerBoard(); top_f = hand(['Ah', 'Ad', 'Ac']); mid_f = hand(['Ks', 'Kh', 'Qc', 'Qd', '2s']); bot_f = hand(['As', 'Ah', 'Qs', 'Js', 'Ts'])
    board_foul.set_full_board(top_f, mid_f, bot_f) # Trips > TwoPair > Pair -> Foul
    assert board_foul.is_foul
    assert board_foul.get_royalties() == {'top': 0, 'middle': 0, 'bottom': 0}
    assert board_foul.get_total_royalty() == 0

# --- Тесты Fantasyland ---
def test_board_fantasyland_methods():
    board_qq = PlayerBoard(); board_qq.set_full_board(hand(['Qc','Qd','2s']), hand(['3h','4h','5h','6h','7h']), hand(['Ac','Ad','Ah','Ks','Kd']))
    # --- ИСПРАВЛЕНО: Эта доска не фол ---
    assert not board_qq.is_foul
    assert board_qq.get_fantasyland_qualification_cards() == 14
    assert not board_qq.check_fantasyland_stay_conditions()
    # --- ИСПРАВЛЕНО: Создаем валидную доску для AAA ---
    board_aaa = PlayerBoard(); board_aaa.set_full_board(hand(['Ac','Ad','Ah']), hand(['Ks','Kd','Qc','Qd','2s']), hand(['As','Kh','Qs','Js','Ts'])) # Trips > TwoPair > A-High
    assert not board_aaa.is_foul
    assert board_aaa.get_fantasyland_qualification_cards() == 17
    assert board_aaa.check_fantasyland_stay_conditions() # Trips on top -> Stay
    board_quads = PlayerBoard(); board_quads.set_full_board(hand(['Kh','Qd','2c']), hand(['Ac','Kd','Qh','Js','9d']), hand(['7h','7d','7c','7s','Ad']))
    assert not board_quads.is_foul
    assert board_quads.get_fantasyland_qualification_cards() == 0
    assert board_quads.check_fantasyland_stay_conditions()
    board_foul = PlayerBoard(); board_foul.set_full_board(hand(['Ah','Ad','Ac']), hand(['Ks','Kh','Qc','Qd','2s']), hand(['As','Ah','Qs','Js','Ts'])) # Foul
    assert board_foul.is_foul
    assert board_foul.get_fantasyland_qualification_cards() == 0
    assert not board_foul.check_fantasyland_stay_conditions()

# --- Тест get_board_state_tuple ---
def test_get_board_state_tuple():
    board = PlayerBoard(); board.add_card(card_from_str('As'), 'top', 0); board.add_card(card_from_str('2c'), 'top', 2); board.add_card(card_from_str('Kd'), 'middle', 1); board.add_card(card_from_str('Qh'), 'middle', 3); board.add_card(card_from_str('Ts'), 'bottom', 4)
    state_tuple = board.get_board_state_tuple()
    assert isinstance(state_tuple, tuple); assert len(state_tuple) == 3; assert all(isinstance(row_tuple, tuple) for row_tuple in state_tuple)
    # --- ИСПРАВЛЕНО: Проверка сортировки с плейсхолдерами ---
    assert state_tuple[0] == ('As', '2c', CARD_PLACEHOLDER)
    assert state_tuple[1] == ('Kd', 'Qh', CARD_PLACEHOLDER, CARD_PLACEHOLDER, CARD_PLACEHOLDER)
    assert state_tuple[2] == ('Ts', CARD_PLACEHOLDER, CARD_PLACEHOLDER, CARD_PLACEHOLDER, CARD_PLACEHOLDER)
    board_full = PlayerBoard(); board_full.set_full_board(hand(['Qc','Qd','2s']), hand(['3h','4h','5h','6h','7h']), hand(['Ac','Ad','Ah','Ks','Kd']))
    state_tuple_full = board_full.get_board_state_tuple()
    assert state_tuple_full[0] == ('Qd', 'Qc', '2s'); assert state_tuple_full[1] == ('7h', '6h', '5h', '4h', '3h'); assert state_tuple_full[2] == ('As', 'Ah', 'Ad', 'Ks', 'Kd')
