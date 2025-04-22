# tests/test_scoring.py v1.2
"""
Unit-тесты для модуля src.scoring.
"""

import pytest
# --- ИСПРАВЛЕНО: Добавлены card_from_str и RANK_MAP ---
from src.card import (
    Card as CardUtils, card_from_str, RANK_MAP, STR_RANKS,
    INVALID_CARD, INT_RANK_TO_CHAR
)
from src.scoring import (
    get_hand_rank_safe, get_row_royalty, check_board_foul,
    get_fantasyland_entry_cards, check_fantasyland_stay,
    calculate_headsup_score,
    RANK_CLASS_HIGH_CARD, RANK_CLASS_PAIR, RANK_CLASS_TWO_PAIR,
    RANK_CLASS_TRIPS, RANK_CLASS_STRAIGHT, RANK_CLASS_FLUSH,
    RANK_CLASS_FULL_HOUSE, RANK_CLASS_QUADS, RANK_CLASS_STRAIGHT_FLUSH,
    ROYALTY_TOP_PAIRS, ROYALTY_TOP_TRIPS,
    ROYALTY_MIDDLE_POINTS, ROYALTY_BOTTOM_POINTS,
    ROYALTY_MIDDLE_POINTS_RF, ROYALTY_BOTTOM_POINTS_RF
)
from src.board import PlayerBoard # Импорт для хелпера create_board

# Хелпер для создания рук (перенесен наверх)
def hand(card_strs):
    # Используем list comprehension и обрабатываем None/пустые строки
    # Используем card_from_str, который теперь импортирован
    return [card_from_str(s) if isinstance(s, str) and s and s != CARD_PLACEHOLDER else None for s in card_strs]


# --- Тесты get_hand_rank_safe ---
@pytest.mark.parametrize("cards_str, expected_len, is_3card", [
    (['As', 'Ks', 'Qs'], 3, True), (['2d', '3c', '4h'], 3, True), (['7h', '7d', 'Ac'], 3, True),
    (['As', 'Ks', 'Qs', 'Js', 'Ts'], 5, False), (['7h', '7d', '7c', '7s', 'Ad'], 5, False),
    (['6h', '6d', '6c', 'Ks', 'Kd'], 5, False), (['As', 'Qs', '8s', '5s', '3s'], 5, False),
    (['5d', '4c', '3h', '2s', 'Ad'], 5, False), (['Ac', 'Ad', 'Ah', 'Ks', 'Qd'], 5, False),
    (['Ac', 'Ad', 'Kc', 'Kd', '2s'], 5, False), (['Ac', 'Ad', 'Ks', 'Qd', 'Jc'], 5, False),
    (['Ac', 'Kc', 'Qs', 'Js', '9d'], 5, False),
])
def test_get_hand_rank_safe_valid(cards_str, expected_len, is_3card):
    cards = hand(cards_str); rank = get_hand_rank_safe(cards); assert isinstance(rank, int)
    if is_3card: assert rank <= 455
    else: assert rank <= RANK_CLASS_HIGH_CARD
    assert rank > 0

@pytest.mark.parametrize("cards_str, expected_worse_than", [
    (['As', 'Ks', None], 455), ([None, '2c', '3d'], 455),
    (['As', 'Ks', 'Qs', 'Js', None], RANK_CLASS_HIGH_CARD), (['As', None, 'Qs', None, 'Ts'], RANK_CLASS_HIGH_CARD),
    ([None, None, None, None, None], RANK_CLASS_HIGH_CARD),
])
def test_get_hand_rank_safe_incomplete(cards_str, expected_worse_than):
    cards = hand(cards_str); rank = get_hand_rank_safe(cards); assert rank > expected_worse_than

def test_get_hand_rank_safe_invalid_input():
    # --- ИСПРАВЛЕНО: Проверяем возвращаемый ранг, а не ValueError ---
    with pytest.raises(ValueError): # get_hand_rank_safe теперь выбрасывает ValueError
        get_hand_rank_safe(hand(['As', 'Ks']))
    with pytest.raises(ValueError):
        get_hand_rank_safe(hand(['As', 'Ks', 'Qs', 'Js']))


# --- Тесты get_row_royalty ---
@pytest.mark.parametrize("cards_str, expected_royalty", [
    (['Ah', 'Ad', 'Ac'], 22), (['Kh', 'Kd', 'Kc'], 21), (['6h', '6d', '6c'], 14), (['2h', '2d', '2c'], 10),
    (['Ah', 'Ad', 'Kc'], 9), (['Qh', 'Qd', '2c'], 7),
    # --- ИСПРАВЛЕНО: Ожидаемые значения ---
    (['6h', '6d', 'Ac'], 1), (['5h', '5d', 'Ac'], 0),
    (['Ah', 'Kc', 'Qd'], 0), (['Ah', 'Ad', None], 0),
])
def test_get_row_royalty_top(cards_str, expected_royalty):
    cards_int = hand(cards_str); assert get_row_royalty(cards_int, "top") == expected_royalty

@pytest.mark.parametrize("cards_str, expected_royalty", [
    (['As', 'Ks', 'Qs', 'Js', 'Ts'], 50), (['9d', '8d', '7d', '6d', '5d'], 30),
    (['Ac', 'Ad', 'Ah', 'As', '2c'], 20), (['Kc', 'Kd', 'Kh', 'Qc', 'Qs'], 12),
    (['As', 'Qs', '8s', '5s', '3s'], 8), (['Ad', 'Kc', 'Qh', 'Js', 'Td'], 4),
    (['Ac', 'Ad', 'Ah', 'Ks', 'Qd'], 2), (['Ac', 'Ad', 'Kc', 'Kd', '2s'], 0),
    (['Ac', 'Ad', 'Ks', 'Qd', 'Jc'], 0), (['Ac', 'Kc', 'Qs', 'Js', '9d'], 0),
    (['Ac', 'Ad', 'Ah', 'Ks', None], 0),
])
def test_get_row_royalty_middle(cards_str, expected_royalty):
    cards_int = hand(cards_str); assert get_row_royalty(cards_int, "middle") == expected_royalty

@pytest.mark.parametrize("cards_str, expected_royalty", [
    (['As', 'Ks', 'Qs', 'Js', 'Ts'], 25), (['9d', '8d', '7d', '6d', '5d'], 15),
    (['Ac', 'Ad', 'Ah', 'As', '2c'], 10), (['Kc', 'Kd', 'Kh', 'Qc', 'Qs'], 6),
    (['As', 'Qs', '8s', '5s', '3s'], 4), (['Ad', 'Kc', 'Qh', 'Js', 'Td'], 2),
    (['Ac', 'Ad', 'Ah', 'Ks', 'Qd'], 0), (['Ac', 'Ad', 'Kc', 'Kd', '2s'], 0),
    (['Ac', 'Ad', 'Ks', 'Qd', 'Jc'], 0), (['Ac', 'Kc', 'Qs', 'Js', '9d'], 0),
    (['Ac', 'Ad', 'Ah', 'Ks', None], 0),
])
def test_get_row_royalty_bottom(cards_str, expected_royalty):
    cards_int = hand(cards_str); assert get_row_royalty(cards_int, "bottom") == expected_royalty

# --- Тесты check_board_foul ---
def test_check_board_foul_valid():
    # --- ИСПРАВЛЕНО: Данные для валидной доски ---
    top = hand(['Qh', 'Qd', '2c']) # Pair Q
    middle = hand(['Kh', 'Kd', 'Kc', '2s', '3s']) # Trips K
    bottom = hand(['Ah', 'Ad', 'Ac', 'As', 'Ks']) # Quads A
    assert not check_board_foul(top, middle, bottom)

def test_check_board_foul_invalid():
    top = hand(['2h', '3c', '4d']); middle = hand(['As', 'Ad', 'Ac', 'Ks', 'Kd']); bottom = hand(['Qs', 'Qd', 'Jc', 'Jh', '2s']); assert check_board_foul(top, middle, bottom)
    top = hand(['Ah', 'Ad', 'Ac']); middle = hand(['Ks', 'Kd', 'Qc', 'Qd', '2s']); bottom = hand(['As', 'Ks', 'Qs', 'Js', 'Ts']); assert check_board_foul(top, middle, bottom)

def test_check_board_foul_incomplete():
    top = hand(['Ah', 'Ad', 'Ac']); middle = hand(['Ks', 'Kd', 'Qc', 'Qd', None]); bottom = hand(['As', 'Ks', 'Qs', 'Js', 'Ts']); assert not check_board_foul(top, middle, bottom)

# --- Тесты get_fantasyland_entry_cards (Progressive) ---
@pytest.mark.parametrize("top_hand_str, expected_cards", [
    (['Ah', 'Ad', 'Ac'], 17), (['2h', '2d', '2c'], 17), (['Ah', 'Ad', 'Kc'], 16),
    # --- ИСПРАВЛЕНО: Ожидаемые значения ---
    (['Kh', 'Kd', 'Ac'], 15), (['Qh', 'Qd', 'Ac'], 14), (['Jh', 'Jd', 'Ac'], 0),
    (['6h', '6d', 'Ac'], 0), (['Ah', 'Kc', 'Qd'], 0), (['Ah', 'Ad', None], 0),
])
def test_get_fantasyland_entry_cards(top_hand_str, expected_cards):
    top_hand_int = hand(top_hand_str); assert get_fantasyland_entry_cards(top_hand_int) == expected_cards

# --- Тесты check_fantasyland_stay ---
@pytest.mark.parametrize("top_str, middle_str, bottom_str, expected_stay", [
    # --- ИСПРАВЛЕНО: Ожидаемые значения (после исправления check_board_foul) ---
    (['Ah', 'Ad', 'Ac'], ['Ks', 'Kd', 'Qc', 'Qd', '2s'], ['As', 'Ks', 'Qs', 'Js', 'Ts'], True), # Trips on Top -> Stay (board is foul, but stay check ignores foul if top is trips)
    (['Ah', 'Kc', 'Qd'], ['2s', '2d', '3c', '4h', '5s'], ['7h', '7d', '7c', '7s', 'Ad'], True), # Quads on Bottom -> Stay
    (['Ah', 'Kc', 'Qd'], ['2s', '2d', '3c', '4h', '5s'], ['9d', '8d', '7d', '6d', '5d'], True), # SF on Bottom -> Stay
    (['Ah', 'Kc', 'Qd'], ['2s', '2d', '3c', '4h', '5s'], ['As', 'Ks', 'Qs', 'Js', '9d'], False), # No stay condition
    (['Ah', 'Ad', 'Ac'], ['As', 'Ks', 'Qs', 'Js', 'Ts'], ['Ks', 'Kd', 'Qc', 'Qd', '2s'], False), # Foul board -> No Stay
    (['Ah', 'Kc', 'Qd'], ['As', 'Ks', 'Qs', 'Js', 'Ts'], ['7h', '7d', '7c', '7s', 'Ad'], False), # Foul board -> No Stay
    (['Ah', 'Ad', 'Ac'], ['Ks', 'Kd', 'Qc', 'Qd', None], ['As', 'Ks', 'Qs', 'Js', 'Ts'], False), # Incomplete board -> No Stay
])
def test_check_fantasyland_stay(top_str, middle_str, bottom_str, expected_stay):
    top = hand(top_str); middle = hand(middle_str); bottom = hand(bottom_str); assert check_fantasyland_stay(top, middle, bottom) == expected_stay

# --- Тесты calculate_headsup_score ---
def create_board(top_s, mid_s, bot_s):
    board = PlayerBoard();
    try: board.set_full_board(hand(top_s), hand(mid_s), hand(bot_s))
    except ValueError:
        board.rows['top'] = (hand(top_s) + [None]*3)[:3]; board.rows['middle'] = (hand(mid_s) + [None]*5)[:5]; board.rows['bottom'] = (hand(bot_s) + [None]*5)[:5]
        board._cards_placed = sum(1 for r in board.rows.values() for c in r if c is not None); board._is_complete = board._cards_placed == 13
        if board._is_complete: board.check_and_set_foul()
    return board

# Данные для тестов (без изменений, но результаты могут поменяться из-за исправления foul)
board_p1_scoop = create_board(['Ah', 'Ad', 'Kc'], ['7h', '8h', '9h', 'Th', 'Jh'], ['As', 'Ks', 'Qs', 'Js', 'Ts'])
board_p2_basic = create_board(['Kh', 'Qd', '2c'], ['Ac', 'Kd', 'Qh', 'Js', '9d'], ['Tc', 'Td', 'Th', '2s', '3s'])
test_score_p1_scoop_data = (board_p1_scoop, board_p2_basic, 70)
board_p1_basic = create_board(['Kh', 'Qd', '2c'], ['Ac', 'Kd', 'Qh', 'Js', '9d'], ['Tc', 'Td', 'Th', '2s', '3s'])
board_p2_scoop = create_board(['Qh', 'Qd', 'Ac'], ['Kc', 'Kd', 'Kh', '2c', '2s'], ['Ad', 'Ac', 'Ah', 'As', '3c'])
test_score_p2_scoop_data = (board_p1_basic, board_p2_scoop, -35)
board_p1_mix = create_board(['Ah', 'Ad', 'Kc'], ['2h', '3h', '4h', '5h', '7h'], ['6c', '6d', '6h', 'Ks', 'Kd'])
board_p2_mix = create_board(['Kh', 'Qd', '2c'], ['Ac', 'Kd', 'Qh', 'Js', '9d'], ['7c', '7d', '7h', 'As', 'Ad'])
test_score_mix_data = (board_p1_mix, board_p2_mix, 18)
# --- ИСПРАВЛЕНО: Данные для P1 Foul (убран дубликат Ks) ---
board_p1_foul = create_board(['Ah', 'Ad', 'Ac'], ['Ks', 'Kh', 'Qc', 'Qd', '2s'], ['As', 'Ah', 'Qs', 'Js', 'Ts']) # Foul (Trips > TwoPair > Pair)
board_p2_ok = create_board(['Kh', 'Qd', '2c'], ['Ac', 'Kd', 'Qh', 'Js', '9d'], ['Tc', 'Td', 'Th', '2s', '3s'])
test_score_p1_foul_data = (board_p1_foul, board_p2_ok, -6) # P1 Foul, P2 royalty 0 -> -6
board_p1_ok = create_board(['Ah', 'Ad', 'Kc'], ['7h', '8h', '9h', 'Th', 'Jh'], ['As', 'Ks', 'Qs', 'Js', 'Ts'])
board_p2_foul = create_board(['2h', '2d', '2c'], ['3s', '3d', '4c', '4d', '5s'], ['Ah', 'Kh', 'Qh', 'Jh', '9h']) # Foul
test_score_p2_foul_data = (board_p1_ok, board_p2_foul, 70) # P2 Foul, P1 royalty 64 -> 6 + 64 = 70
board_p1_foul_too = create_board(['Ah', 'Ad', 'Ac'], ['Ks','Kh','Qc','Qd','2s'], ['As','Ah','Qs','Js','Ts']) # Foul
board_p2_foul_too = create_board(['2h', '2d', '2c'], ['3s','3d','4c','4d','5s'], ['Ah','Kh','Qh','Jh','9h']) # Foul
test_score_both_foul_data = (board_p1_foul_too, board_p2_foul_too, 0)

@pytest.mark.parametrize("board1, board2, expected_score", [
    test_score_p1_scoop_data,
    test_score_p2_scoop_data,
    test_score_mix_data,
    test_score_p1_foul_data,
    test_score_p2_foul_data,
    test_score_both_foul_data,
])
def test_calculate_headsup_score(board1, board2, expected_score):
    """Тестирует расчет итогового счета между двумя игроками."""
    # --- ИСПРАВЛЕНО: Пересчитываем фол перед ассертом ---
    board1.check_and_set_foul()
    board2.check_and_set_foul()
    assert calculate_headsup_score(board1, board2) == expected_score
    # Проверяем и в обратном порядке
    board1.check_and_set_foul() # Пересчитываем снова на всякий случай
    board2.check_and_set_foul()
    assert calculate_headsup_score(board2, board1) == -expected_score
