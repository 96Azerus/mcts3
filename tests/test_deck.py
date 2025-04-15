# tests/test_deck.py
"""
Unit-тесты для модуля src.deck.
"""

import pytest
import random

# --- ИСПРАВЛЕНО: Импорт из src ---
from src.deck import Deck
from src.card import Card, card_from_str, card_to_str, INVALID_CARD

# --- Тесты инициализации ---

def test_deck_init_full():
    """Тестирует создание полной колоды."""
    deck = Deck()
    assert len(deck) == 52
    assert len(deck.cards) == 52
    # Проверяем наличие нескольких известных карт
    assert card_from_str('As') in deck
    assert card_from_str('2c') in deck
    assert card_from_str('Kd') in deck

def test_deck_init_with_cards():
    """Тестирует создание колоды из заданного набора карт."""
    initial_cards_strs = {'As', 'Ks', 'Qs'}
    initial_cards_ints = {card_from_str(c) for c in initial_cards_strs}
    deck = Deck(cards=initial_cards_ints)
    assert len(deck) == 3
    assert deck.cards == initial_cards_ints
    # Убедимся, что исходный set не изменился (из-за копирования в конструкторе)
    initial_cards_ints.add(card_from_str('Js'))
    assert len(deck) == 3 # Длина колоды не должна измениться

def test_deck_init_empty():
    """Тестирует создание пустой колоды."""
    deck = Deck(cards=set())
    assert len(deck) == 0
    assert len(deck.cards) == 0

# --- Тесты раздачи карт (deal) ---

def test_deck_deal_single():
    """Тестирует раздачу одной карты."""
    deck = Deck()
    initial_len = len(deck)
    dealt_card_list = deck.deal(1)
    assert len(dealt_card_list) == 1
    assert len(deck) == initial_len - 1
    dealt_card = dealt_card_list[0]
    assert isinstance(dealt_card, int)
    assert dealt_card != INVALID_CARD
    assert dealt_card not in deck # Карта должна быть удалена из колоды

def test_deck_deal_multiple():
    """Тестирует раздачу нескольких карт."""
    deck = Deck()
    initial_len = len(deck)
    num_deal = 5
    dealt_cards = deck.deal(num_deal)
    assert len(dealt_cards) == num_deal
    assert len(deck) == initial_len - num_deal
    assert len(set(dealt_cards)) == num_deal # Проверка уникальности розданных карт
    for card in dealt_cards:
        assert card not in deck

def test_deck_deal_all():
    """Тестирует раздачу всех карт из колоды."""
    deck = Deck()
    dealt_cards = deck.deal(52)
    assert len(dealt_cards) == 52
    assert len(deck) == 0
    assert len(set(dealt_cards)) == 52
    assert set(dealt_cards) == Deck.FULL_DECK_CARDS

def test_deck_deal_more_than_available():
    """Тестирует раздачу большего количества карт, чем есть в колоде."""
    initial_cards = {card_from_str(c) for c in ['Ah', 'Kh', 'Qh']}
    deck = Deck(cards=initial_cards)
    dealt_cards = deck.deal(5) # Пытаемся раздать 5 из 3
    assert len(dealt_cards) == 3 # Должны раздать только 3
    assert len(deck) == 0
    assert set(dealt_cards) == initial_cards

def test_deck_deal_zero_or_negative():
    """Тестирует раздачу 0 или отрицательного количества карт."""
    deck = Deck()
    initial_len = len(deck)
    assert deck.deal(0) == []
    assert len(deck) == initial_len
    assert deck.deal(-5) == []
    assert len(deck) == initial_len

# --- Тесты удаления карт (remove) ---

def test_deck_remove_existing():
    """Тестирует удаление существующих карт."""
    deck = Deck()
    initial_len = len(deck)
    cards_to_remove_strs = ['As', 'Kd']
    cards_to_remove_ints = [card_from_str(c) for c in cards_to_remove_strs]
    deck.remove(cards_to_remove_ints)
    assert len(deck) == initial_len - 2
    assert card_from_str('As') not in deck
    assert card_from_str('Kd') not in deck

def test_deck_remove_non_existing():
    """Тестирует удаление несуществующих карт (не должно ничего менять)."""
    deck = Deck()
    initial_len = len(deck)
    cards_to_remove_ints = [card_from_str('As'), 999999] # 999999 - невалидная карта
    deck.remove(cards_to_remove_ints)
    assert len(deck) == initial_len - 1 # Удален только туз
    assert card_from_str('As') not in deck

def test_deck_remove_empty_list():
    """Тестирует удаление с пустым списком."""
    deck = Deck()
    initial_len = len(deck)
    deck.remove([])
    assert len(deck) == initial_len

# --- Тесты добавления карт (add) ---

def test_deck_add_cards():
    """Тестирует добавление карт в колоду."""
    deck = Deck(cards={card_from_str('2c')})
    cards_to_add_strs = ['3d', '4h']
    cards_to_add_ints = [card_from_str(c) for c in cards_to_add_strs]
    deck.add(cards_to_add_ints)
    assert len(deck) == 3
    assert card_from_str('3d') in deck
    assert card_from_str('4h') in deck

def test_deck_add_duplicate():
    """Тестирует добавление карты, которая уже есть в колоде."""
    deck = Deck(cards={card_from_str('2c')})
    deck.add([card_from_str('2c')])
    assert len(deck) == 1 # Длина не должна измениться

def test_deck_add_invalid():
    """Тестирует добавление невалидных карт."""
    deck = Deck(cards={card_from_str('2c')})
    deck.add([INVALID_CARD, None, -5])
    assert len(deck) == 1 # Длина не должна измениться

# --- Тесты копирования (copy) ---

def test_deck_copy():
    """Тестирует создание копии колоды."""
    deck1 = Deck()
    deck1.deal(10) # Изменяем первую колоду
    deck2 = deck1.copy()

    assert deck1 is not deck2 # Объекты разные
    assert deck1.cards is not deck2.cards # Наборы карт тоже разные объекты
    assert len(deck1) == len(deck2) # Содержимое одинаковое
    assert deck1.cards == deck2.cards

    # Изменяем копию, оригинал не должен измениться
    deck2.deal(5)
    assert len(deck1) == 42
    assert len(deck2) == 37

# --- Тесты магических методов ---

def test_deck_len():
    """Тестирует __len__."""
    assert len(Deck()) == 52
    assert len(Deck(cards=set())) == 0
    deck = Deck()
    deck.deal(10)
    assert len(deck) == 42

def test_deck_contains():
    """Тестирует __contains__."""
    deck = Deck()
    ace_spades = card_from_str('As')
    two_clubs = card_from_str('2c')
    assert ace_spades in deck
    deck.remove([ace_spades])
    assert ace_spades not in deck
    assert two_clubs in deck
    assert INVALID_CARD not in deck
    assert None not in deck

# --- Тест на полноту FULL_DECK_CARDS ---
def test_full_deck_completeness():
    """Проверяет, что константа FULL_DECK_CARDS содержит ровно 52 уникальные карты."""
    assert len(Deck.FULL_DECK_CARDS) == 52
    # Дополнительно можно проверить, что все карты валидны
    assert all(isinstance(c, int) and c != INVALID_CARD and c > 0 for c in Deck.FULL_DECK_CARDS)
