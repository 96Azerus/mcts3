# tests/test_scoring.py
"""
Unit-тесты для модуля src.scoring.
"""

import pytest

# Импорты из src пакета
from src.card import card_from_str, Card as CardUtils, INVALID_CARD
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
# Импортируем PlayerBoard для тестов calculate_headsup_score
from src.board import PlayerBoard

# Хелпер для создания рук
def hand(card_strs):
    return [card_from_str(s) if s else None for s in card_strs]

# --- Тесты get_hand_rank_safe ---

@pytest.mark.parametrize("cards, expected_len, is_3card", [
    (hand(['As', 'Ks', 'Qs']), 3, True),
    (hand(['2d', '3c', '4h']), 3, True),
    (hand(['7h', '7d', 'Ac']), 3, True),
    (hand(['As', 'Ks', 'Qs', 'Js', 'Ts']), 5, False), # RF
    (hand(['7h', '7d', '7c', '7s', 'Ad']), 5, False), # Quads
    (hand(['6h', '6d', '6c', 'Ks', 'Kd']), 5, False), # Full House
    (hand(['As', 'Qs', '8s', '5s', '3s']), 5, False), # Flush
    (hand(['5d', '4c', '3h', '2s', 'Ad']), 5, False), # Straight (Wheel)
    (hand(['Ac', 'Ad', 'Ah', 'Ks', 'Qd']), 5, False), # Trips
    (hand(['Ac', 'Ad', 'Kc', 'Kd', '2s']), 5, False), # Two Pair
    (hand(['Ac', 'Ad', 'Ks', 'Qd', 'Jc']), 5, False), # Pair
    (hand(['Ac', 'Kc', 'Qs', 'Js', '9d']), 5, False), # High Card
])
def test_get_hand_rank_safe_valid(cards, expected_len, is_3card):
    """Тестирует get_hand_rank_safe с валидными полными руками."""
    rank = get_hand_rank_safe(cards)
    assert isinstance(rank, int)
    if is_3card:
        assert rank <= 455 # Худший ранг для 3 карт
    else:
        assert rank <= RANK_CLASS_HIGH_CARD # Худший ранг для 5 карт
    assert rank > 0

@pytest.mark.parametrize("cards, expected_worse_than", [
    (hand(['As', 'Ks', None]), 455), # 3-карточная неполная
    (hand([None, '2c', '3d']), 455),
    (hand(['As', 'Ks', 'Qs', 'Js', None]), RANK_CLASS_HIGH_CARD), # 5-карточная неполная
    (hand(['As', None, 'Qs', None, 'Ts']), RANK_CLASS_HIGH_CARD),
    (hand([None, None, None, None, None]), RANK_CLASS_HIGH_CARD),
])
def test_get_hand_rank_safe_incomplete(cards, expected_worse_than):
    """Тестирует get_hand_rank_safe с неполными руками."""
    rank = get_hand_rank_safe(cards)
    assert rank > expected_worse_than

def test_get_hand_rank_safe_invalid_input():
    """Тестирует get_hand_rank_safe с невалидным вводом."""
    # Неверная длина списка
    with pytest.raises(ValueError): # Ожидаем ошибку от эвалуаторов или самой функции
         get_hand_rank_safe(hand(['As', 'Ks']))
    with pytest.raises(ValueError):
         get_hand_rank_safe(hand(['As', 'Ks', 'Qs', 'Js']))
    # Неверный тип карты (хотя get_hand_rank_safe ожидает Optional[int])
    # Эта проверка должна быть на уровне вызова
    # assert get_hand_rank_safe(['As', 'Ks', 'Qs']) > 455 # Строки вместо int

# --- Тесты get_row_royalty ---

# Top Row Royalty
@pytest.mark.parametrize("cards_str, expected_royalty", [
    (['Ah', 'Ad', 'Ac'], ROYALTY_TOP_TRIPS[RANK_MAP['A']]), # AAA = 22
    (['Kh', 'Kd', 'Kc'], ROYALTY_TOP_TRIPS[RANK_MAP['K']]), # KKK = 21
    (['6h', '6d', '6c'], ROYALTY_TOP_TRIPS[RANK_MAP['6']]), # 666 = 14
    (['2h', '2d', '2c'], ROYALTY_TOP_TRIPS[RANK_MAP['2']]), # 222 = 10
    (['Ah', 'Ad', 'Kc'], ROYALTY_TOP_PAIRS[RANK_MAP['A']]), # AAK = 9
    (['Qh', 'Qd', '2c'], ROYALTY_TOP_PAIRS[RANK_MAP['Q']]), # QQ2 = 7
    (['6h', '6d', 'Ac'], ROYALTY_TOP_PAIRS[RANK_MAP['6']]), # 66A = 1
    (['5h', '5d', 'Ac'], 0), # 55A = 0
    (['Ah', 'Kc', 'Qd'], 0), # AKQ = 0
    (['Ah', 'Ad', None], 0), # Incomplete
])
def test_get_row_royalty_top(cards_str, expected_royalty):
    cards_int = hand(cards_str)
    assert get_row_royalty(cards_int, "top") == expected_royalty

# Middle Row Royalty
@pytest.mark.parametrize("cards_str, expected_royalty", [
    (['As', 'Ks', 'Qs', 'Js', 'Ts'], ROYALTY_MIDDLE_POINTS_RF), # RF = 50
    (['9d', '8d', '7d', '6d', '5d'], ROYALTY_MIDDLE_POINTS["Straight Flush"]), # SF = 30
    (['Ac', 'Ad', 'Ah', 'As', '2c'], ROYALTY_MIDDLE_POINTS["Four of a Kind"]), # 4K = 20
    (['Kc', 'Kd', 'Kh', 'Qc', 'Qs'], ROYALTY_MIDDLE_POINTS["Full House"]), # FH = 12
    (['As', 'Qs', '8s', '5s', '3s'], ROYALTY_MIDDLE_POINTS["Flush"]), # Flush = 8
    (['Ad', 'Kc', 'Qh', 'Js', 'Td'], ROYALTY_MIDDLE_POINTS["Straight"]), # Straight = 4
    (['Ac', 'Ad', 'Ah', 'Ks', 'Qd'], ROYALTY_MIDDLE_POINTS["Three of a Kind"]), # 3K = 2
    (['Ac', 'Ad', 'Kc', 'Kd', '2s'], 0), # Two Pair = 0
    (['Ac', 'Ad', 'Ks', 'Qd', 'Jc'], 0), # Pair = 0
    (['Ac', 'Kc', 'Qs', 'Js', '9d'], 0), # High Card = 0
    (['Ac', 'Ad', 'Ah', 'Ks', None], 0), # Incomplete
])
def test_get_row_royalty_middle(cards_str, expected_royalty):
    cards_int = hand(cards_str)
    assert get_row_royalty(cards_int, "middle") == expected_royalty

# Bottom Row Royalty
@pytest.mark.parametrize("cards_str, expected_royalty", [
    (['As', 'Ks', 'Qs', 'Js', 'Ts'], ROYALTY_BOTTOM_POINTS_RF), # RF = 25
    (['9d', '8d', '7d', '6d', '5d'], ROYALTY_BOTTOM_POINTS["Straight Flush"]), # SF = 15
    (['Ac', 'Ad', 'Ah', 'As', '2c'], ROYALTY_BOTTOM_POINTS["Four of a Kind"]), # 4K = 10
    (['Kc', 'Kd', 'Kh', 'Qc', 'Qs'], ROYALTY_BOTTOM_POINTS["Full House"]), # FH = 6
    (['As', 'Qs', '8s', '5s', '3s'], ROYALTY_BOTTOM_POINTS["Flush"]), # Flush = 4
    (['Ad', 'Kc', 'Qh', 'Js', 'Td'], ROYALTY_BOTTOM_POINTS["Straight"]), # Straight = 2
    (['Ac', 'Ad', 'Ah', 'Ks', 'Qd'], 0), # Three of a Kind = 0 on bottom
    (['Ac', 'Ad', 'Kc', 'Kd', '2s'], 0), # Two Pair = 0
    (['Ac', 'Ad', 'Ks', 'Qd', 'Jc'], 0), # Pair = 0
    (['Ac', 'Kc', 'Qs', 'Js', '9d'], 0), # High Card = 0
    (['Ac', 'Ad', 'Ah', 'Ks', None], 0), # Incomplete
])
def test_get_row_royalty_bottom(cards_str, expected_royalty):
    cards_int = hand(cards_str)
    assert get_row_royalty(cards_int, "bottom") == expected_royalty

# --- Тесты check_board_foul ---

def test_check_board_foul_valid():
    """Тестирует валидные (не фол) доски."""
    # Простой пример: A-high < Pair < Trips
    top = hand(['Ah', 'Kc', 'Qd']) # A High
    middle = hand(['2s', '2d', '3c', '4h', '5s']) # Pair 2s
    bottom = hand(['7h', '7d', '7c', 'As', 'Ks']) # Trips 7s
    assert not check_board_foul(top, middle, bottom)

    # Пример с равными руками (допустимо)
    top = hand(['Ah', 'Kc', 'Qd']) # A High
    middle = hand(['Ah', 'Kd', 'Qc', 'Js', '9h']) # A High (same rank class)
    bottom = hand(['Ah', 'Kh', 'Qh', 'Jh', '8d']) # A High (same rank class)
    assert not check_board_foul(top, middle, bottom)

    # Пример: Flush < Full House < Quads
    top = hand(['As', 'Ks', 'Qs']) # A High
    middle = hand(['2h', '3h', '8h', 'Th', 'Jh']) # Flush
    bottom = hand(['Ad', 'Ac', 'Ah', 'Kd', 'Kc']) # Full House
    assert not check_board_foul(top, middle, bottom)

def test_check_board_foul_invalid():
    """Тестирует фол доски."""
    # Middle > Bottom
    top = hand(['2h', '3c', '4d'])
    middle = hand(['As', 'Ad', 'Ac', 'Ks', 'Kd']) # Full House
    bottom = hand(['Qs', 'Qd', 'Jc', 'Jh', '2s']) # Two Pair
    assert check_board_foul(top, middle, bottom)

    # Top > Middle
    top = hand(['Ah', 'Ad', 'Ac']) # Trips
    middle = hand(['Ks', 'Kd', 'Qc', 'Qd', '2s']) # Two Pair
    bottom = hand(['As', 'Ks', 'Qs', 'Js', 'Ts']) # Royal Flush
    assert check_board_foul(top, middle, bottom)

def test_check_board_foul_incomplete():
    """Тестирует неполные доски (не должны быть фолом)."""
    top = hand(['Ah', 'Ad', 'Ac'])
    middle = hand(['Ks', 'Kd', 'Qc', 'Qd', None]) # Неполный средний
    bottom = hand(['As', 'Ks', 'Qs', 'Js', 'Ts'])
    assert not check_board_foul(top, middle, bottom)

# --- Тесты get_fantasyland_entry_cards (Progressive) ---

@pytest.mark.parametrize("top_hand_str, expected_cards", [
    (['Ah', 'Ad', 'Ac'], 17), # AAA
    (['2h', '2d', '2c'], 17), # 222
    (['Ah', 'Ad', 'Kc'], 16), # AA
    (['Kh', 'Kd', 'Ac'], 15), # KK
    (['Qh', 'Qd', 'Ac'], 14), # QQ
    (['Jh', 'Jd', 'Ac'], 0),  # JJ
    (['6h', '6d', 'Ac'], 0),  # 66
    (['Ah', 'Kc', 'Qd'], 0),  # High Card
    (['Ah', 'Ad', None], 0),  # Incomplete
])
def test_get_fantasyland_entry_cards(top_hand_str, expected_cards):
    top_hand_int = hand(top_hand_str)
    assert get_fantasyland_entry_cards(top_hand_int) == expected_cards

# --- Тесты check_fantasyland_stay ---

@pytest.mark.parametrize("top_str, middle_str, bottom_str, expected_stay", [
    # Trips on Top -> Stay
    (['Ah', 'Ad', 'Ac'], ['Ks', 'Kd', 'Qc', 'Qd', '2s'], ['As', 'Ks', 'Qs', 'Js', 'Ts'], True),
    # Quads on Bottom -> Stay
    (['Ah', 'Kc', 'Qd'], ['2s', '2d', '3c', '4h', '5s'], ['7h', '7d', '7c', '7s', 'Ad'], True),
    # SF on Bottom -> Stay
    (['Ah', 'Kc', 'Qd'], ['2s', '2d', '3c', '4h', '5s'], ['9d', '8d', '7d', '6d', '5d'], True),
    # No stay condition (valid board)
    (['Ah', 'Kc', 'Qd'], ['2s', '2d', '3c', '4h', '5s'], ['As', 'Ks', 'Qs', 'Js', '9d'], False),
    # Trips on Top BUT Foul board -> No Stay
    (['Ah', 'Ad', 'Ac'], ['As', 'Ks', 'Qs', 'Js', 'Ts'], ['Ks', 'Kd', 'Qc', 'Qd', '2s'], False),
    # Quads on Bottom BUT Foul board -> No Stay
    (['Ah', 'Kc', 'Qd'], ['As', 'Ks', 'Qs', 'Js', 'Ts'], ['7h', '7d', '7c', '7s', 'Ad'], False),
    # Incomplete board -> No Stay
    (['Ah', 'Ad', 'Ac'], ['Ks', 'Kd', 'Qc', 'Qd', None], ['As', 'Ks', 'Qs', 'Js', 'Ts'], False),
])
def test_check_fantasyland_stay(top_str, middle_str, bottom_str, expected_stay):
    top = hand(top_str)
    middle = hand(middle_str)
    bottom = hand(bottom_str)
    assert check_fantasyland_stay(top, middle, bottom) == expected_stay

# --- Тесты calculate_headsup_score ---

def create_board(top_s, mid_s, bot_s):
    """Хелпер для создания объекта PlayerBoard из строк."""
    board = PlayerBoard()
    try:
        board.set_full_board(hand(top_s), hand(mid_s), hand(bot_s))
    except ValueError: # Обработка случая фола при создании
        # Устанавливаем карты как есть и вручную ставим фол, если нужно
        board.rows['top'] = hand(top_s) + [None]*(3-len(top_s))
        board.rows['middle'] = hand(mid_s) + [None]*(5-len(mid_s))
        board.rows['bottom'] = hand(bot_s) + [None]*(5-len(bot_s))
        board._cards_placed = sum(1 for r in board.rows.values() for c in r if c is not None)
        board._is_complete = board._cards_placed == 13
        if board._is_complete:
             board.check_and_set_foul() # Установит is_foul, если нужно
    return board

# P1 Wins (Scoop + Royalties) vs P2 Basic
board_p1_scoop = create_board(['Ah', 'Ad', 'Kc'], # AA = 9 royalty
                              ['7h', '8h', '9h', 'Th', 'Jh'], # SF = 30 royalty
                              ['As', 'Ks', 'Qs', 'Js', 'Ts']) # RF = 25 royalty
board_p2_basic = create_board(['Kh', 'Qd', '2c'], # K High = 0
                              ['Ac', 'Kd', 'Qh', 'Js', '9d'], # A High = 0
                              ['Tc', 'Td', 'Th', '2s', '3s']) # Trips T = 0
# P1 Lines: AA > K High, SF > A High, RF > Trips T -> +3 points, +3 scoop = +6
# P1 Royalties: 9 + 30 + 25 = 64
# P2 Royalties: 0
# Total Score (P1 vs P2) = 6 + (64 - 0) = 70
test_score_p1_scoop_data = (board_p1_scoop, board_p2_basic, 70)

# P2 Wins (Scoop + Royalties) vs P1 Basic
board_p1_basic = create_board(['Kh', 'Qd', '2c'], # 0
                              ['Ac', 'Kd', 'Qh', 'Js', '9d'], # 0
                              ['Tc', 'Td', 'Th', '2s', '3s']) # 0
board_p2_scoop = create_board(['Qh', 'Qd', 'Ac'], # QQ = 7 royalty
                              ['Kc', 'Kd', 'Kh', '2c', '2s'], # FH = 12 royalty
                              ['Ad', 'Ac', 'Ah', 'As', '3c']) # 4K = 10 royalty
# P2 Lines: QQ > K High, FH > A High, 4K > Trips T -> P1 loses 3 lines -> -3 points, -3 scoop = -6
# P1 Royalties: 0
# P2 Royalties: 7 + 12 + 10 = 29
# Total Score (P1 vs P2) = -6 + (0 - 29) = -35
test_score_p2_scoop_data = (board_p1_basic, board_p2_scoop, -35)

# Mixed Wins + Royalties
board_p1_mix = create_board(['Ah', 'Ad', 'Kc'], # AA = 9 royalty (P1 wins line +1)
                            ['2h', '3h', '4h', '5h', '7h'], # Flush = 8 royalty (P1 wins line +1)
                            ['6c', '6d', '6h', 'Ks', 'Kd']) # FH = 6 royalty (P2 wins line -1)
board_p2_mix = create_board(['Kh', 'Qd', '2c'], # K High = 0
                            ['Ac', 'Kd', 'Qh', 'Js', '9d'], # A High = 0
                            ['7c', '7d', '7h', 'As', 'Ad']) # FH = 6 royalty
# Line Scores: P1(+1), P1(+1), P2(-1) -> Total = +1
# Royalties: P1 = 9 + 8 + 6 = 23; P2 = 0 + 0 + 6 = 6
# Total Score (P1 vs P2) = 1 + (23 - 6) = 1 + 17 = 18
test_score_mix_data = (board_p1_mix, board_p2_mix, 18)

# P1 Foul vs P2 Non-Foul
board_p1_foul = create_board(['Ah', 'Ad', 'Ac'], # Trips (Foul source)
                             ['Ks', 'Kd', 'Qc', 'Qd', '2s'], # Two Pair
                             ['As', 'Ks', 'Qs', 'Js', 'Ts']) # RF = 25 royalty (ignored due to foul)
board_p2_ok = create_board(['Kh', 'Qd', '2c'], # K High = 0
                           ['Ac', 'Kd', 'Qh', 'Js', '9d'], # A High = 0
                           ['Tc', 'Td', 'Th', '2s', '3s']) # Trips T = 0
# P1 Foul -> P1 loses 6 points + P2 royalties
# Total Score (P1 vs P2) = -6 + (0 - 0) = -6
test_score_p1_foul_data = (board_p1_foul, board_p2_ok, -6)

# P2 Foul vs P1 Non-Foul
board_p1_ok = create_board(['Ah', 'Ad', 'Kc'], # AA = 9 royalty
                           ['7h', '8h', '9h', 'Th', 'Jh'], # SF = 30 royalty
                           ['As', 'Ks', 'Qs', 'Js', 'Ts']) # RF = 25 royalty
board_p2_foul = create_board(['2h', '2d', '2c'], # Trips (Foul source)
                             ['3s', '3d', '4c', '4d', '5s'], # Two Pair
                             ['Ah', 'Kh', 'Qh', 'Jh', '9h']) # Flush = 4 royalty (ignored)
# P2 Foul -> P1 wins 6 points + P1 royalties
# P1 Royalties = 9 + 30 + 25 = 64
# Total Score (P1 vs P2) = 6 + (64 - 0) = 70
test_score_p2_foul_data = (board_p1_ok, board_p2_foul, 70)

# Both Foul
board_p1_foul_too = create_board(['Ah', 'Ad', 'Ac'], ['Ks','Kd','Qc','Qd','2s'], ['As','Ks','Qs','Js','Ts'])
board_p2_foul_too = create_board(['2h', '2d', '2c'], ['3s','3d','4c','4d','5s'], ['Ah','Kh','Qh','Jh','9h'])
# Both Foul -> 0 score
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
    assert calculate_headsup_score(board1, board2) == expected_score
    # Проверяем и в обратном порядке (должен быть инвертированный знак)
    assert calculate_headsup_score(board2, board1) == -expected_score
