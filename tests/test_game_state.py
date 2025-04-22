# tests/test_game_state.py v1.1
"""
Unit-тесты для модуля src.game_state.
"""
import pytest
import random
from src.game_state import GameState
# --- ИСПРАВЛЕНО: Добавлен импорт Card ---
from src.card import Card, card_from_str, card_to_str, INVALID_CARD
from src.deck import Deck
from src.board import PlayerBoard
from src.scoring import calculate_headsup_score

# Хелперы
def hand(card_strs):
    # Используем Card.hand_to_int для консистентности
    return Card.hand_to_int(card_strs)

# (create_deck_with_known_cards без изменений)
def create_deck_with_known_cards(top_cards_strs: list[str]) -> Deck:
    deck = Deck(); top_cards_ints = [card_from_str(s) for s in top_cards_strs]; deck.remove(top_cards_ints); remaining_cards = deck.get_remaining_cards(); random.shuffle(remaining_cards); final_card_list = top_cards_ints + remaining_cards; test_deck = Deck(); test_deck.cards = set(final_card_list); return test_deck

# --- Тесты инициализации и старта раунда ---
def test_gamestate_init_default():
    state = GameState(); assert state.dealer_idx == 0; assert state._internal_current_player_idx == 1; assert state.street == 0; assert len(state.boards) == 2; assert len(state.deck) == 52; assert not state.is_fantasyland_round; assert state.fantasyland_status == [False, False]; assert state.next_fantasyland_status == [False, False]; assert state.fantasyland_cards_to_deal == [0, 0]; assert state._player_finished_round == [False, False]

def test_gamestate_start_new_round_normal():
    state = GameState(dealer_idx=1); state.start_new_round(dealer_button_idx=1)
    assert state.street == 1; assert state.dealer_idx == 1; assert state._internal_current_player_idx == 0; assert not state.is_fantasyland_round
    assert state.current_hands.get(0) is not None; assert len(state.current_hands[0]) == 5
    # --- ИСПРАВЛЕНО: У P1 не должно быть руки ---
    assert state.current_hands.get(1) is None
    assert len(state.deck) == 52 - 5; assert state._player_acted_this_street == [False, False]; assert state._player_finished_round == [False, False]

def test_gamestate_start_new_round_fantasyland():
    state = GameState(next_fantasyland_status=[True, False], fantasyland_cards_to_deal=[15, 0]); state.start_new_round(dealer_button_idx=0)
    assert state.street == 1; assert state.dealer_idx == 0
    # --- ИСПРАВЛЕНО: Первым ходит не-ФЛ игрок ---
    assert state._internal_current_player_idx == 1
    assert state.is_fantasyland_round; assert state.fantasyland_status == [True, False]; assert state.fantasyland_hands[0] is not None; assert len(state.fantasyland_hands[0]) == 15; assert state.fantasyland_hands[1] is None; assert state.current_hands.get(0) is None; assert state.current_hands.get(1) is not None; assert len(state.current_hands[1]) == 5; assert len(state.deck) == 52 - 15 - 5

def test_gamestate_start_new_round_fl_carryover():
    state = GameState(); state.next_fantasyland_status = [False, True]; state.fantasyland_cards_to_deal = [0, 14]; state.start_new_round(dealer_button_idx=1)
    assert state.is_fantasyland_round; assert state.fantasyland_status == [False, True]; assert state.fantasyland_hands[1] is not None; assert len(state.fantasyland_hands[1]) == 14; assert state.current_hands.get(0) is not None; assert len(state.current_hands[0]) == 5

# (Тесты apply_action_street1, apply_action_pineapple, apply_fantasyland_placement, apply_fantasyland_foul без изменений)
def test_gamestate_apply_action_street1():
    state = GameState(dealer_idx=1); state.start_new_round(1); hand_p0 = state.current_hands[0]; assert hand_p0 is not None and len(hand_p0) == 5; placements = [(hand_p0[0], 'bottom', 0), (hand_p0[1], 'bottom', 1), (hand_p0[2], 'middle', 0), (hand_p0[3], 'middle', 1), (hand_p0[4], 'top', 0)]; action = (placements, []); next_state = state.apply_action(0, action); assert next_state is not state; assert next_state.current_hands.get(0) is None; assert next_state.boards[0].get_total_cards() == 5; assert next_state.boards[0].rows['bottom'][0] == hand_p0[0]; assert next_state.boards[0].rows['top'][0] == hand_p0[4]; assert next_state._player_acted_this_street[0] is True; assert next_state._last_player_acted == 0
def test_gamestate_apply_action_pineapple():
    state = GameState(dealer_idx=0); state.start_new_round(0); state.street = 2; state._internal_current_player_idx = 0; hand_p0 = hand(['As', 'Ks', 'Qs']); state.current_hands[0] = hand_p0; state.current_hands[1] = None; state._player_acted_this_street = [False, False]; action = ((hand_p0[0], 'top', 0), (hand_p0[1], 'middle', 0), hand_p0[2]); next_state = state.apply_action(0, action); assert next_state is not state; assert next_state.current_hands.get(0) is None; assert next_state.boards[0].rows['top'][0] == hand_p0[0]; assert next_state.boards[0].rows['middle'][0] == hand_p0[1]; assert hand_p0[2] in next_state.private_discard[0]; assert next_state._player_acted_this_street[0] is True; assert next_state._last_player_acted == 0
def test_gamestate_apply_action_completes_board():
    state = GameState(); board = state.boards[0]; cards_on_board = hand(['2c','3c','4c','5c','6c','7c','8c','9c','Tc','Jc','Qc']); idx = 0;
    for r in ['bottom', 'middle']:
        for i in range(5): board.add_card(cards_on_board[idx], r, i); idx+=1
    board.add_card(cards_on_board[idx], 'top', 0); idx+=1 # Добавляем 11-ю карту
    assert board.get_total_cards() == 11
    state.street = 5; state._internal_current_player_idx = 0; hand_p0 = hand(['Ac', 'Ad', 'Ah']); state.current_hands[0] = hand_p0
    # Действие: Положить Ac, Ad в top[1], top[2], сбросить Ah
    action_correct = ((hand_p0[0], 'top', 1), (hand_p0[1], 'top', 2), hand_p0[2])
    next_state = state.apply_action(0, action_correct)
    assert next_state.boards[0].is_complete(); assert next_state._player_finished_round[0] is True; assert next_state.next_fantasyland_status[0] is False; assert next_state.fantasyland_cards_to_deal[0] == 0
def test_gamestate_apply_fantasyland_placement():
    state = GameState(fantasyland_status=[True, False], fantasyland_cards_to_deal=[14, 0]); state.start_new_round(0); hand_fl = state.fantasyland_hands[0]; assert hand_fl is not None and len(hand_fl) == 14; placement_dict = {'bottom': hand_fl[0:5], 'middle': hand_fl[5:10], 'top': hand_fl[10:13]}; discarded = [hand_fl[13]]; next_state = state.apply_fantasyland_placement(0, placement_dict, discarded); assert next_state is not state; assert next_state.fantasyland_hands[0] is None; assert next_state.boards[0].is_complete(); assert next_state._player_finished_round[0] is True; assert discarded[0] in next_state.private_discard[0]; assert next_state._last_player_acted == 0
def test_gamestate_apply_fantasyland_foul():
    state = GameState(fantasyland_status=[True, False], fantasyland_cards_to_deal=[14, 0]); state.start_new_round(0); hand_fl = state.fantasyland_hands[0]; assert hand_fl is not None; next_state = state.apply_fantasyland_foul(0, hand_fl); assert next_state is not state; assert next_state.fantasyland_hands[0] is None; assert next_state.boards[0].is_foul is True; assert next_state._player_finished_round[0] is True; assert set(hand_fl).issubset(set(next_state.private_discard[0])); assert next_state.next_fantasyland_status[0] is False; assert next_state._last_player_acted == 0

# --- Тесты продвижения состояния ---
def test_gamestate_get_player_to_move():
    state = GameState(dealer_idx=0); state.start_new_round(0)
    # --- ИСПРАВЛЕНО: После старта P0 получил карты, P1 нет -> ход P0 ---
    assert state.get_player_to_move() == 0
    state = state.advance_state() # Пытаемся раздать P1 (не должен получить)
    assert state.get_player_to_move() == 0 # Ход все еще P0

    # P0 сходил, P1 еще нет и нет карт -> ожидание (-1)
    state._player_acted_this_street[0] = True; state.current_hands[0] = None; state.current_hands[1] = None
    state._internal_current_player_idx = 1 # Передаем ход P1
    assert state.get_player_to_move() == -1

    # P0 сходил, P1 получил карты -> ход P1
    state.current_hands[1] = hand(['Ac','Kc','Qc','Jc','Tc'])
    assert state.get_player_to_move() == 1

    # P1 сходил, P0 ждет карт -> ожидание (-1)
    state._player_acted_this_street[1] = True; state.current_hands[1] = None; state.current_hands[0] = None
    state._internal_current_player_idx = 0 # Передали ход P0
    assert state.get_player_to_move() == -1

    state_fl = GameState(fantasyland_status=[True, False], fantasyland_cards_to_deal=[14,0]); state_fl.start_new_round(0)
    # --- ИСПРАВЛЕНО: Первым ходит P1 (не-ФЛ) ---
    assert state_fl.get_player_to_move() == 1

    state_fl.current_hands[1] = None; state_fl._player_acted_this_street[1] = True # P1 сходил
    assert state_fl.get_player_to_move() == 0 # Теперь ходит P0 (ФЛ)

    state_fl.fantasyland_hands[0] = None; state_fl.current_hands[1] = hand(['Ac','Kc','Qc','Jc','Tc'])
    state_fl._player_finished_round[0] = True # P0 закончил (без руки)
    assert state_fl.get_player_to_move() == 1 # Ходит P1

    state._player_finished_round = [True, True]; assert state.is_round_over(); assert state.get_player_to_move() == -1

def test_gamestate_advance_state_normal_round():
    state = GameState(dealer_idx=1); state.start_new_round(1) # Ход P0, P0 получил карты
    assert state.get_player_to_move() == 0
    hand_p0 = state.current_hands[0]; action_p0 = ([(hand_p0[i], 'bottom', i) for i in range(5)], []); state_after_p0 = state.apply_action(0, action_p0)
    state_after_advance1 = state_after_p0.advance_state() # Переход хода и раздача P1
    assert state_after_advance1._internal_current_player_idx == 1; assert state_after_advance1.current_hands.get(1) is not None; assert len(state_after_advance1.current_hands[1]) == 5; assert state_after_advance1.get_player_to_move() == 1
    hand_p1 = state_after_advance1.current_hands[1]; action_p1 = ([(hand_p1[i], 'middle', i) for i in range(5)], []); state_after_p1 = state_after_advance1.apply_action(1, action_p1)
    state_after_advance2 = state_after_p1.advance_state() # Переход на улицу 2, раздача обоим
    assert state_after_advance2.street == 2; assert state_after_advance2._player_acted_this_street == [False, False]; assert state_after_advance2._internal_current_player_idx == 0; assert state_after_advance2.current_hands.get(0) is not None
    # --- ИСПРАВЛЕНО: Ожидаем 3 карты ---
    assert len(state_after_advance2.current_hands[0]) == 3
    assert state_after_advance2.current_hands.get(1) is not None
    assert len(state_after_advance2.current_hands[1]) == 3
    assert state_after_advance2.get_player_to_move() == 0

def test_gamestate_advance_state_fantasyland():
    state = GameState(fantasyland_status=[True, False], fantasyland_cards_to_deal=[14, 0]); state.start_new_round(0)
    # --- ИСПРАВЛЕНО: Первым ходит P1 (не-ФЛ) ---
    assert state.get_player_to_move() == 1
    hand_p1_s1 = state.current_hands[1]; action_p1_s1 = ([(hand_p1_s1[i], 'bottom', i) for i in range(5)], []); state_after_p1_s1 = state.apply_action(1, action_p1_s1)
    state_after_advance1 = state_after_p1_s1.advance_state() # Переход хода к P0 (ФЛ)
    assert state_after_advance1.street == 1; assert state_after_advance1.get_player_to_move() == 0
    hand_p0_fl = state_after_advance1.fantasyland_hands[0]; placement_p0 = {'bottom': hand_p0_fl[0:5], 'middle': hand_p0_fl[5:10], 'top': hand_p0_fl[10:13]}; discarded_p0 = [hand_p0_fl[13]]; state_after_p0_fl = state_after_advance1.apply_fantasyland_placement(0, placement_p0, discarded_p0)
    state_after_advance2 = state_after_p0_fl.advance_state() # Переход на улицу 2 для P1
    assert state_after_advance2.street == 2; assert state_after_advance2.current_hands.get(1) is not None; assert len(state_after_advance2.current_hands[1]) == 3; assert state_after_advance2.get_player_to_move() == 1

# --- Тесты конца раунда и счета ---
def test_gamestate_end_of_round_and_score():
    state = GameState()
    board1 = PlayerBoard(); board1.set_full_board(hand(['Ah','Ad','Kc']), hand(['7h','8h','9h','Th','Jh']), hand(['As','Ks','Qs','Js','Ts']))
    board2 = PlayerBoard(); board2.set_full_board(hand(['Kh','Qd','2c']), hand(['Ac','Kd','Qh','Js','9d']), hand(['Tc','Td','Th','2s','3s']))
    state.boards = [board1, board2]; state._player_finished_round = [True, True]; state.street = 6
    assert state.is_round_over(); assert state.get_terminal_score() == 70
    # --- ИСПРАВЛЕНО: Данные для фол-доски ---
    board1_foul = PlayerBoard(); board1_foul.set_full_board(hand(['Ah','Ad','Ac']), hand(['Ks','Kh','Qc','Qd','2s']), hand(['As','Ah','Qs','Js','Ts'])) # Foul
    state.boards = [board1_foul, board2]
    assert state.is_round_over()
    assert state.get_terminal_score() == -6

# --- Тесты логики Fantasyland ---
def test_gamestate_fantasyland_entry_and_stay():
    state = GameState(); state.street = 5; state._internal_current_player_idx = 0
    hand_entry = hand(['Ac', 'Ad', 'Kc']); state.current_hands[0] = hand_entry
    board = state.boards[0]; cards_on_board = hand(['2c','3c','4c','5c','6c','7c','8c','9c','Tc','Jc','Qc']) # 11 карт
    idx = 0
    for r in ['bottom', 'middle']:
        for i in range(5): board.add_card(cards_on_board[idx], r, i); idx+=1
    board.add_card(cards_on_board[idx], 'top', 0); idx+=1
    assert board.get_total_cards() == 11

    # --- ИСПРАВЛЕНО: Действие для завершения доски (2 карты) ---
    action_entry = ((hand_entry[0], 'top', 1), (hand_entry[1], 'top', 2), hand_entry[2])
    state_after_entry = state.apply_action(0, action_entry)

    assert state_after_entry.boards[0].is_complete(); assert state_after_entry._player_finished_round[0] is True
    assert state_after_entry.next_fantasyland_status[0] is True; assert state_after_entry.fantasyland_cards_to_deal[0] == 16

    state_fl = GameState(fantasyland_status=[True, False], fantasyland_cards_to_deal=[16, 0]); state_fl.start_new_round(0)
    fl_hand_stay = hand(['Ah','Ad','Ac','Ks','Kd','Qs','Qd','Js','Jd','Ts','Td','2s','2d','2h','3c','3d'])
    state_fl.fantasyland_hands[0] = fl_hand_stay
    placement_stay = {'top': fl_hand_stay[0:3], 'middle': fl_hand_stay[3:8], 'bottom': fl_hand_stay[8:13]}; discarded_stay = fl_hand_stay[13:]
    state_after_fl_stay = state_fl.apply_fantasyland_placement(0, placement_stay, discarded_stay)
    assert state_after_fl_stay.boards[0].is_complete(); assert state_after_fl_stay._player_finished_round[0] is True
    assert state_after_fl_stay.next_fantasyland_status[0] is True; assert state_after_fl_stay.fantasyland_cards_to_deal[0] == 14

# --- Тесты сериализации и копирования ---
def test_gamestate_serialization():
    state = GameState(dealer_idx=1, street=1); state.start_new_round(1); state = state.advance_state() # Deal P0
    if state.get_player_to_move() == 0: hand0 = state.current_hands[0]; action0 = ([(hand0[i], 'bottom', i) for i in range(5)], []); state = state.apply_action(0, action0)
    state = state.advance_state() # Pass turn, deal P1
    if state.get_player_to_move() == 1: hand1 = state.current_hands[1]; action1 = ([(hand1[i], 'middle', i) for i in range(5)], []); state = state.apply_action(1, action1)
    state = state.advance_state() # Change street, deal both
    state_dict = state.to_dict(); assert isinstance(state_dict, dict); assert isinstance(state_dict['boards'], list); assert isinstance(state_dict['fantasyland_status'], list); assert isinstance(state_dict['street'], int)
    restored_state = GameState.from_dict(state_dict); assert state.get_state_representation() == restored_state.get_state_representation(); assert state.street == restored_state.street; assert state.dealer_idx == restored_state.dealer_idx; assert state.boards[0].get_total_cards() == restored_state.boards[0].get_total_cards(); assert len(state.deck) == len(restored_state.deck)

def test_gamestate_copy():
    state1 = GameState(dealer_idx=1); state1.start_new_round(1); state1 = state1.advance_state() # Deal P0
    hand_p0 = state1.current_hands[0]; action_p0 = ([(hand_p0[i], 'bottom', i) for i in range(5)], []); state1 = state1.apply_action(0, action_p0)
    state2 = state1.copy(); assert state1 is not state2; assert state1.boards is not state2.boards; assert state1.boards[0] is not state2.boards[0]; assert state1.deck is not state2.deck; assert state1.get_state_representation() == state2.get_state_representation()
    state2 = state2.advance_state(); # Deal P1 in copy
    # --- ИСПРАВЛЕНО: Проверяем, что рука P1 в оригинале все еще None ---
    assert state1.current_hands.get(1) is None
    assert state2.current_hands.get(1) is not None # P1 получил карты в копии
    hand_p1 = state2.current_hands[1]; action_p1 = ([(hand_p1[i], 'middle', i) for i in range(5)], []); state2 = state2.apply_action(1, action_p1)
    assert state1.street == 1; assert state1.boards[1].get_total_cards() == 0
    assert state2.street == 1; assert state2.boards[1].get_total_cards() == 5; assert state2.current_hands.get(1) is None
