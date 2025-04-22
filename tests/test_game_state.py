# tests/test_game_state.py
"""
Unit-тесты для модуля src.game_state.
"""

import pytest
import random

# Импорты из src пакета
from src.game_state import GameState
from src.card import card_from_str, card_to_str, INVALID_CARD
from src.deck import Deck
from src.board import PlayerBoard
from src.scoring import calculate_headsup_score # Для проверки get_terminal_score

# Хелперы
def hand(card_strs):
    return [card_from_str(s) if s else None for s in card_strs]

def create_deck_with_known_cards(top_cards_strs: list[str]) -> Deck:
    """Создает колоду, где верхние карты известны."""
    deck = Deck()
    top_cards_ints = [card_from_str(s) for s in top_cards_strs]
    # Удаляем эти карты из колоды
    deck.remove(top_cards_ints)
    # Создаем новую колоду, добавляя известные карты в начало (неэффективно, но для тестов ок)
    remaining_cards = deck.get_remaining_cards()
    random.shuffle(remaining_cards) # Перемешиваем остальные
    final_card_list = top_cards_ints + remaining_cards
    # Создаем set из списка для конструктора Deck
    # Важно: set не гарантирует порядок, поэтому этот подход не сработает для deal!
    # Правильный подход: Модифицировать Deck.deal или создать mock-колоду.
    # Пока что будем проверять наличие карт в руке, а не точный порядок.
    test_deck = Deck()
    test_deck.cards = set(final_card_list) # Просто создаем колоду с нужным набором карт
    return test_deck

# --- Тесты инициализации и старта раунда ---

def test_gamestate_init_default():
    """Тест инициализации по умолчанию."""
    state = GameState()
    assert state.dealer_idx == 0
    assert state._internal_current_player_idx == 1 # Игрок слева от дилера 0
    assert state.street == 0
    assert len(state.boards) == 2
    assert len(state.deck) == 52
    assert not state.is_fantasyland_round
    assert state.fantasyland_status == [False, False]
    assert state.next_fantasyland_status == [False, False]
    assert state.fantasyland_cards_to_deal == [0, 0]
    assert state._player_finished_round == [False, False] # Доски пустые

def test_gamestate_start_new_round_normal():
    """Тест старта обычного раунда."""
    state = GameState(dealer_idx=1) # Дилер P1
    state.start_new_round(dealer_button_idx=1)

    assert state.street == 1
    assert state.dealer_idx == 1
    assert state._internal_current_player_idx == 0 # Ходит P0 (слева от P1)
    assert not state.is_fantasyland_round
    assert state.current_hands.get(0) is not None # У P0 должна быть рука
    assert len(state.current_hands[0]) == 5
    assert state.current_hands.get(1) is None # У P1 пока нет руки
    assert len(state.deck) == 52 - 5 # 5 карт сдано
    assert state._player_acted_this_street == [False, False]
    assert state._player_finished_round == [False, False]

def test_gamestate_start_new_round_fantasyland():
    """Тест старта раунда Фантазии."""
    # P0 входит в ФЛ с 15 картами, P1 - нет
    state = GameState(next_fantasyland_status=[True, False], fantasyland_cards_to_deal=[15, 0])
    state.start_new_round(dealer_button_idx=0) # Дилер P0

    assert state.street == 1
    assert state.dealer_idx == 0
    assert state._internal_current_player_idx == 1 # Первым ходит P1 (не-ФЛ)
    assert state.is_fantasyland_round
    assert state.fantasyland_status == [True, False] # Статус текущего раунда
    assert state.fantasyland_hands[0] is not None # P0 получил руку ФЛ
    assert len(state.fantasyland_hands[0]) == 15
    assert state.fantasyland_hands[1] is None
    assert state.current_hands.get(0) is None # У P0 нет обычной руки
    assert state.current_hands.get(1) is not None # P1 (не-ФЛ) получил руку улицы 1
    assert len(state.current_hands[1]) == 5
    assert len(state.deck) == 52 - 15 - 5 # Сдано 15 (ФЛ) + 5 (улица 1)

def test_gamestate_start_new_round_fl_carryover():
    """Тест переноса статуса ФЛ между раундами."""
    state = GameState()
    state.next_fantasyland_status = [False, True] # P1 будет в ФЛ
    state.fantasyland_cards_to_deal = [0, 14]
    state.start_new_round(dealer_button_idx=1) # Новый дилер P1

    assert state.is_fantasyland_round
    assert state.fantasyland_status == [False, True]
    assert state.fantasyland_hands[1] is not None
    assert len(state.fantasyland_hands[1]) == 14
    assert state.current_hands.get(0) is not None # P0 (не-ФЛ) получил руку улицы 1
    assert len(state.current_hands[0]) == 5

# --- Тесты применения действий ---

def test_gamestate_apply_action_street1():
    """Тест применения действия на улице 1."""
    state = GameState(dealer_idx=1)
    state.start_new_round(1) # Ход P0, рука из 5 карт
    hand_p0 = state.current_hands[0]
    assert hand_p0 is not None and len(hand_p0) == 5

    # Простое действие: разместить первые 5 карт в первые слоты
    placements = [
        (hand_p0[0], 'bottom', 0), (hand_p0[1], 'bottom', 1),
        (hand_p0[2], 'middle', 0), (hand_p0[3], 'middle', 1),
        (hand_p0[4], 'top', 0)
    ]
    action = (placements, [])
    next_state = state.apply_action(0, action)

    assert next_state is not state # Должен вернуться новый объект
    assert next_state.current_hands.get(0) is None # Рука должна быть убрана
    assert next_state.boards[0].get_total_cards() == 5
    assert next_state.boards[0].rows['bottom'][0] == hand_p0[0]
    assert next_state.boards[0].rows['top'][0] == hand_p0[4]
    assert next_state._player_acted_this_street[0] is True
    assert next_state._last_player_acted == 0

def test_gamestate_apply_action_pineapple():
    """Тест применения действия Pineapple (улицы 2-5)."""
    state = GameState(dealer_idx=0)
    state.start_new_round(0) # Ход P1
    # Пропускаем ход P1, переходим на улицу 2, ход P0
    state.street = 2
    state._internal_current_player_idx = 0
    hand_p0 = hand(['As', 'Ks', 'Qs'])
    state.current_hands[0] = hand_p0
    state.current_hands[1] = None # Убираем руку P1 для чистоты
    state._player_acted_this_street = [False, False]

    # Действие: Положить As в top[0], Ks в middle[0], сбросить Qs
    action = ((hand_p0[0], 'top', 0), (hand_p0[1], 'middle', 0), hand_p0[2])
    next_state = state.apply_action(0, action)

    assert next_state is not state
    assert next_state.current_hands.get(0) is None
    assert next_state.boards[0].rows['top'][0] == hand_p0[0]
    assert next_state.boards[0].rows['middle'][0] == hand_p0[1]
    assert hand_p0[2] in next_state.private_discard[0]
    assert next_state._player_acted_this_street[0] is True
    assert next_state._last_player_acted == 0

def test_gamestate_apply_action_completes_board():
    """Тест, что apply_action корректно обрабатывает завершение доски."""
    state = GameState()
    # Почти полная доска P0
    board = state.boards[0]
    cards_on_board = hand(['2c','3c','4c','5c','6c','7c','8c','9c','Tc','Jc','Qc'])
    idx = 0
    for r in ['bottom', 'middle']:
        for i in range(5): board.add_card(cards_on_board[idx], r, i); idx+=1
    for i in range(1): board.add_card(cards_on_board[idx], 'top', i); idx+=1
    assert board.get_total_cards() == 11

    state.street = 5 # Последняя улица
    state._internal_current_player_idx = 0
    hand_p0 = hand(['Ac', 'Ad', 'Ah']) # Рука для завершения
    state.current_hands[0] = hand_p0

    # Действие: Положить Ac, Ad в top[1], top[2], сбросить Ah
    action = ((hand_p0[0], 'top', 1), (hand_p0[1], 'top', 2), hand_p0[2])
    next_state = state.apply_action(0, action)

    assert next_state.boards[0].is_complete()
    assert next_state._player_finished_round[0] is True
    # Проверяем, что статус ФЛ обновился (здесь не должно быть ФЛ)
    assert next_state.next_fantasyland_status[0] is False
    assert next_state.fantasyland_cards_to_deal[0] == 0

def test_gamestate_apply_fantasyland_placement():
    """Тест применения размещения Фантазии."""
    state = GameState(fantasyland_status=[True, False], fantasyland_cards_to_deal=[14, 0])
    state.start_new_round(0) # P0 в ФЛ, P1 ходит первым
    hand_fl = state.fantasyland_hands[0]
    assert hand_fl is not None and len(hand_fl) == 14

    # Простое размещение (первые 13 карт) и сброс последней
    placement_dict = {
        'bottom': hand_fl[0:5],
        'middle': hand_fl[5:10],
        'top': hand_fl[10:13]
    }
    discarded = [hand_fl[13]]

    next_state = state.apply_fantasyland_placement(0, placement_dict, discarded)

    assert next_state is not state
    assert next_state.fantasyland_hands[0] is None
    assert next_state.boards[0].is_complete()
    assert next_state._player_finished_round[0] is True
    assert discarded[0] in next_state.private_discard[0]
    assert next_state._last_player_acted == 0
    # Статус Re-Fantasy зависит от конкретной руки, здесь не проверяем детально

def test_gamestate_apply_fantasyland_foul():
    """Тест применения фола в Фантазии."""
    state = GameState(fantasyland_status=[True, False], fantasyland_cards_to_deal=[14, 0])
    state.start_new_round(0)
    hand_fl = state.fantasyland_hands[0]
    assert hand_fl is not None

    next_state = state.apply_fantasyland_foul(0, hand_fl)

    assert next_state is not state
    assert next_state.fantasyland_hands[0] is None
    assert next_state.boards[0].is_foul is True
    # assert next_state.boards[0].get_total_cards() == 0 # Проверяем, что доска очищена (если так реализовано)
    assert next_state._player_finished_round[0] is True
    assert set(hand_fl).issubset(set(next_state.private_discard[0]))
    assert next_state.next_fantasyland_status[0] is False # Фол не дает Re-FL
    assert next_state._last_player_acted == 0

# --- Тесты продвижения состояния ---

def test_gamestate_get_player_to_move():
    """Тестирует определение игрока для хода."""
    # Начало раунда, дилер 0, ходит P1
    state = GameState(dealer_idx=0); state.start_new_round(0)
    assert state.get_player_to_move() == 1

    # P1 сходил, P0 еще нет, но у P0 нет карт -> никто не ходит (-1)
    state._player_acted_this_street[1] = True; state.current_hands[1] = None; state.current_hands[0] = None
    assert state.get_player_to_move() == -1

    # P1 сходил, P0 получил карты -> ходит P0
    state.current_hands[0] = hand(['Ac','Kc','Qc','Jc','Tc'])
    assert state.get_player_to_move() == 0

    # P0 сходил, P1 ждет карт -> никто не ходит (-1)
    state._player_acted_this_street[0] = True; state.current_hands[0] = None; state.current_hands[1] = None
    state._internal_current_player_idx = 1 # Передали ход P1
    assert state.get_player_to_move() == -1

    # Раунд ФЛ, P0 в ФЛ с картами, P1 не в ФЛ без карт -> ходит P0
    state_fl = GameState(fantasyland_status=[True, False], fantasyland_cards_to_deal=[14,0])
    state_fl.start_new_round(0) # Дилер P0, первым ходит P1
    state_fl.current_hands[1] = None # Убираем карты P1
    assert state_fl.get_player_to_move() == 0 # P0 (ФЛ) должен ходить

    # Раунд ФЛ, P0 в ФЛ без карт, P1 не в ФЛ с картами -> ходит P1
    state_fl.fantasyland_hands[0] = None
    state_fl.current_hands[1] = hand(['Ac','Kc','Qc','Jc','Tc'])
    assert state_fl.get_player_to_move() == 1

    # Раунд завершен -> никто не ходит (-1)
    state._player_finished_round = [True, True]
    assert state.is_round_over()
    assert state.get_player_to_move() == -1


def test_gamestate_advance_state_normal_round():
    """Тестирует advance_state в обычном раунде."""
    # Дилер P1, Улица 1, Ход P0
    state = GameState(dealer_idx=1); state.start_new_round(1)
    hand_p0 = state.current_hands[0]
    action_p0 = ([(hand_p0[i], 'bottom', i) for i in range(5)], [])
    state_after_p0 = state.apply_action(0, action_p0)
    assert state_after_p0._last_player_acted == 0
    assert state_after_p0._internal_current_player_idx == 0 # Ход еще не передан

    # Продвигаем состояние: должен передаться ход P1 и раздаться карты P1
    state_after_advance1 = state_after_p0.advance_state()
    assert state_after_advance1._internal_current_player_idx == 1 # Ход перешел к P1
    assert state_after_advance1.current_hands.get(1) is not None # P1 получил карты
    assert len(state_after_advance1.current_hands[1]) == 5
    assert state_after_advance1.get_player_to_move() == 1 # Теперь ход P1

    # P1 ходит
    hand_p1 = state_after_advance1.current_hands[1]
    action_p1 = ([(hand_p1[i], 'middle', i) for i in range(5)], [])
    state_after_p1 = state_after_advance1.apply_action(1, action_p1)
    assert state_after_p1._last_player_acted == 1
    assert state_after_p1._internal_current_player_idx == 1 # Ход еще не передан

    # Продвигаем состояние: оба сходили на улице 1 -> переход на улицу 2, раздача обоим
    state_after_advance2 = state_after_p1.advance_state()
    assert state_after_advance2.street == 2
    assert state_after_advance2._player_acted_this_street == [False, False] # Флаги сброшены
    assert state_after_advance2._internal_current_player_idx == 0 # Ход вернулся к P0
    assert state_after_advance2.current_hands.get(0) is not None # P0 получил карты
    assert len(state_after_advance2.current_hands[0]) == 3
    assert state_after_advance2.current_hands.get(1) is not None # P1 получил карты
    assert len(state_after_advance2.current_hands[1]) == 3
    assert state_after_advance2.get_player_to_move() == 0 # Ход P0

def test_gamestate_advance_state_fantasyland():
    """Тестирует advance_state в раунде Фантазии."""
    # P0 в ФЛ (14 карт), P1 не в ФЛ. Дилер P0.
    state = GameState(fantasyland_status=[True, False], fantasyland_cards_to_deal=[14, 0])
    state.start_new_round(0) # Раздаются карты P0(FL) и P1(Street 1)
    assert state.get_player_to_move() == 1 # Первым ходит P1 (не-ФЛ)

    # P1 ходит на улице 1
    hand_p1_s1 = state.current_hands[1]
    action_p1_s1 = ([(hand_p1_s1[i], 'bottom', i) for i in range(5)], [])
    state_after_p1_s1 = state.apply_action(1, action_p1_s1)
    assert state_after_p1_s1._last_player_acted == 1

    # Продвигаем состояние: P1 сходил, P0 (ФЛ) еще нет. Ход должен остаться у P0? Нет, advance не меняет игрока в ФЛ.
    state_after_advance1 = state_after_p1_s1.advance_state()
    # Раздача карт не должна произойти, т.к. P0 в ФЛ, а P1 уже сходил
    assert state_after_advance1.street == 1 # Улица не меняется
    assert state_after_advance1.get_player_to_move() == 0 # Теперь должен ходить P0 (ФЛ)

    # P0 размещает ФЛ
    hand_p0_fl = state_after_advance1.fantasyland_hands[0]
    placement_p0 = {'bottom': hand_p0_fl[0:5], 'middle': hand_p0_fl[5:10], 'top': hand_p0_fl[10:13]}
    discarded_p0 = [hand_p0_fl[13]]
    state_after_p0_fl = state_after_advance1.apply_fantasyland_placement(0, placement_p0, discarded_p0)
    assert state_after_p0_fl._player_finished_round[0] is True
    assert state_after_p0_fl._last_player_acted == 0

    # Продвигаем состояние: P0 закончил, P1 еще нет. Должна начаться улица 2 для P1.
    state_after_advance2 = state_after_p0_fl.advance_state()
    assert state_after_advance2.street == 2 # Улица должна смениться для P1
    assert state_after_advance2.current_hands.get(1) is not None # P1 получил карты улицы 2
    assert len(state_after_advance2.current_hands[1]) == 3
    assert state_after_advance2.get_player_to_move() == 1 # Ход P1

# --- Тесты конца раунда и счета ---

def test_gamestate_end_of_round_and_score():
    """Тестирует is_round_over и get_terminal_score."""
    state = GameState()
    # Создаем завершенное состояние (можно взять из тестов scoring)
    board1 = PlayerBoard(); board1.set_full_board(hand(['Ah','Ad','Kc']), hand(['7h','8h','9h','Th','Jh']), hand(['As','Ks','Qs','Js','Ts'])) # P1 = 70 очков
    board2 = PlayerBoard(); board2.set_full_board(hand(['Kh','Qd','2c']), hand(['Ac','Kd','Qh','Js','9d']), hand(['Tc','Td','Th','2s','3s']))
    state.boards = [board1, board2]
    state._player_finished_round = [True, True] # Оба завершили
    state.street = 6 # Условно ставим улицу > 5

    assert state.is_round_over()
    # get_terminal_score возвращает счет с точки зрения P0
    assert state.get_terminal_score() == 70

    # Случай с фолом P1
    board1_foul = PlayerBoard(); board1_foul.set_full_board(hand(['Ah','Ad','Ac']), hand(['Ks','Kd','Qc','Qd','2s']), hand(['As','Ks','Qs','Js','Ts'])) # Foul
    state.boards = [board1_foul, board2]
    assert state.is_round_over()
    assert state.get_terminal_score() == -6 # P1 фол, P2 роялти 0 -> P0 получает -6

# --- Тесты логики Fantasyland ---

def test_gamestate_fantasyland_entry_and_stay():
    """Тестирует обновление next_fantasyland_status."""
    state = GameState()
    state.street = 5
    state._internal_current_player_idx = 0
    # Рука для входа в ФЛ (AAK -> 16 карт)
    hand_entry = hand(['Ac', 'Ad', 'Kc'])
    state.current_hands[0] = hand_entry
    # Заполняем доску P0 так, чтобы она была валидной и завершилась этим ходом
    board = state.boards[0]
    cards_on_board = hand(['2c','3c','4c','5c','6c','7c','8c','9c','Tc','Jc']) # 10 карт
    idx = 0
    for r in ['bottom', 'middle']:
        for i in range(5): board.add_card(cards_on_board[idx], r, i); idx+=1
    assert board.get_total_cards() == 10

    # Действие: Положить Ac, Ad в top[0], top[1], сбросить Kc
    action_entry = ((hand_entry[0], 'top', 0), (hand_entry[1], 'top', 1), hand_entry[2])
    state_after_entry = state.apply_action(0, action_entry)

    assert state_after_entry.boards[0].is_complete()
    assert state_after_entry._player_finished_round[0] is True
    assert state_after_entry.next_fantasyland_status[0] is True # Должен войти в ФЛ
    assert state_after_entry.fantasyland_cards_to_deal[0] == 16 # AA -> 16 карт

    # Теперь симулируем раунд ФЛ и проверяем Re-Fantasy
    state_fl = GameState(fantasyland_status=[True, False], fantasyland_cards_to_deal=[16, 0])
    state_fl.start_new_round(0)
    hand_fl = state_fl.fantasyland_hands[0]
    # Размещение с трипсом наверху для Re-Fantasy
    placement_stay = {
        'top': hand(['Ah', 'Ad', 'Ac']), # Trips -> Stay
        'middle': hand_fl[3:8],
        'bottom': hand_fl[8:13]
    }
    discarded_stay = hand_fl[13:] # Сбрасываем остальные
    # Убедимся, что карты валидны (нужно взять из руки)
    # Проще создать руку с нужными картами
    fl_hand_stay = hand(['Ah','Ad','Ac','Ks','Kd','Qs','Qd','Js','Jd','Ts','Td','2s','2d','2h','3c','3d']) # 16 карт
    state_fl.fantasyland_hands[0] = fl_hand_stay
    placement_stay = {'top': fl_hand_stay[0:3], 'middle': fl_hand_stay[3:8], 'bottom': fl_hand_stay[8:13]}
    discarded_stay = fl_hand_stay[13:]

    state_after_fl_stay = state_fl.apply_fantasyland_placement(0, placement_stay, discarded_stay)
    assert state_after_fl_stay.boards[0].is_complete()
    assert state_after_fl_stay._player_finished_round[0] is True
    assert state_after_fl_stay.next_fantasyland_status[0] is True # Должен остаться
    assert state_after_fl_stay.fantasyland_cards_to_deal[0] == 14 # Re-Fantasy всегда 14

# --- Тесты сериализации и копирования ---

def test_gamestate_serialization():
    """Тестирует to_dict и from_dict."""
    state = GameState(dealer_idx=1, street=3)
    # Сделаем несколько ходов для наполнения состояния
    state.start_new_round(1)
    state = state.advance_state() # Раздали P1
    if state.get_player_to_move() == 0: # Ход P0
         hand0 = state.current_hands[0]
         action0 = ([(hand0[i], 'bottom', i) for i in range(5)], [])
         state = state.apply_action(0, action0)
         state = state.advance_state() # Переход хода/раздача P1
    if state.get_player_to_move() == 1: # Ход P1
         hand1 = state.current_hands[1]
         action1 = ([(hand1[i], 'middle', i) for i in range(5)], [])
         state = state.apply_action(1, action1)
         state = state.advance_state() # Переход на улицу 2, раздача обоим

    state_dict = state.to_dict()
    # Проверяем базовые типы в словаре
    assert isinstance(state_dict, dict)
    assert isinstance(state_dict['boards'], list)
    assert isinstance(state_dict['fantasyland_status'], list)
    assert isinstance(state_dict['street'], int)

    restored_state = GameState.from_dict(state_dict)

    # Сравниваем состояния через их представления
    assert state.get_state_representation() == restored_state.get_state_representation()
    # Проверяем несколько ключевых атрибутов
    assert state.street == restored_state.street
    assert state.dealer_idx == restored_state.dealer_idx
    assert state.boards[0].get_total_cards() == restored_state.boards[0].get_total_cards()
    assert len(state.deck) == len(restored_state.deck) # Длина колоды должна совпадать

def test_gamestate_copy():
    """Тестирует метод copy()."""
    state1 = GameState(dealer_idx=1)
    state1.start_new_round(1)
    state1.apply_action(0, ([(state1.current_hands[0][i], 'bottom', i) for i in range(5)], []))

    state2 = state1.copy()

    assert state1 is not state2
    assert state1.boards is not state2.boards
    assert state1.boards[0] is not state2.boards[0]
    assert state1.deck is not state2.deck
    assert state1.get_state_representation() == state2.get_state_representation()

    # Изменяем копию
    state2 = state2.advance_state() # Раздаем P1
    hand_p1 = state2.current_hands[1]
    action_p1 = ([(hand_p1[i], 'middle', i) for i in range(5)], [])
    state2 = state2.apply_action(1, action_p1)

    # Оригинал не должен измениться
    assert state1.street == 1
    assert state1.boards[1].get_total_cards() == 0
    assert state1.current_hands.get(1) is None

    # Копия изменилась
    assert state2.street == 1 # Улица еще не сменилась
    assert state2.boards[1].get_total_cards() == 5
    assert state2.current_hands.get(1) is None # Рука P1 применена
