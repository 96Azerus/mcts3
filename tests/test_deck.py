# tests/test_deck.py v1.1
"""
Unit-тесты для модуля src.deck.
"""
import pytest
import random
from src.deck import Deck
# --- ИСПРАВЛЕНО: Импортируем Card для Card.hand_to_int ---
from src.card import Card, card_from_str, card_to_str, INVALID_CARD

# (Тесты init, deal, remove, copy, len, contains, completeness без изменений)
def test_deck_init_full(): deck = Deck(); assert len(deck) == 52; assert len(deck.cards) == 52; assert card_from_str('As') in deck; assert card_from_str('2c') in deck; assert card_from_str('Kd') in deck
def test_deck_init_with_cards(): initial_cards_strs = {'As', 'Ks', 'Qs'}; initial_cards_ints = {card_from_str(c) for c in initial_cards_strs}; deck = Deck(cards=initial_cards_ints); assert len(deck) == 3; assert deck.cards == initial_cards_ints; initial_cards_ints.add(card_from_str('Js')); assert len(deck) == 3
def test_deck_init_empty(): deck = Deck(cards=set()); assert len(deck) == 0; assert len(deck.cards) == 0
def test_deck_deal_single(): deck = Deck(); initial_len = len(deck); dealt_card_list = deck.deal(1); assert len(dealt_card_list) == 1; assert len(deck) == initial_len - 1; dealt_card = dealt_card_list[0]; assert isinstance(dealt_card, int); assert dealt_card != INVALID_CARD; assert dealt_card not in deck
def test_deck_deal_multiple(): deck = Deck(); initial_len = len(deck); num_deal = 5; dealt_cards = deck.deal(num_deal); assert len(dealt_cards) == num_deal; assert len(deck) == initial_len - num_deal; assert len(set(dealt_cards)) == num_deal; assert all(card not in deck for card in dealt_cards)
def test_deck_deal_all(): deck = Deck(); dealt_cards = deck.deal(52); assert len(dealt_cards) == 52; assert len(deck) == 0; assert len(set(dealt_cards)) == 52; assert set(dealt_cards) == Deck.FULL_DECK_CARDS
def test_deck_deal_more_than_available(): initial_cards = {card_from_str(c) for c in ['Ah', 'Kh', 'Qh']}; deck = Deck(cards=initial_cards); dealt_cards = deck.deal(5); assert len(dealt_cards) == 3; assert len(deck) == 0; assert set(dealt_cards) == initial_cards
def test_deck_deal_zero_or_negative(): deck = Deck(); initial_len = len(deck); assert deck.deal(0) == []; assert len(deck) == initial_len; assert deck.deal(-5) == []; assert len(deck) == initial_len
def test_deck_remove_existing(): deck = Deck(); initial_len = len(deck); cards_to_remove_strs = ['As', 'Kd']; cards_to_remove_ints = [card_from_str(c) for c in cards_to_remove_strs]; deck.remove(cards_to_remove_ints); assert len(deck) == initial_len - 2; assert card_from_str('As') not in deck; assert card_from_str('Kd') not in deck
def test_deck_remove_non_existing(): deck = Deck(); initial_len = len(deck); cards_to_remove_ints = [card_from_str('As'), 999999]; deck.remove(cards_to_remove_ints); assert len(deck) == initial_len - 1; assert card_from_str('As') not in deck
def test_deck_remove_empty_list(): deck = Deck(); initial_len = len(deck); deck.remove([]); assert len(deck) == initial_len
def test_deck_add_cards(): deck = Deck(cards={card_from_str('2c')}); cards_to_add_strs = ['3d', '4h']; cards_to_add_ints = [card_from_str(c) for c in cards_to_add_strs]; deck.add(cards_to_add_ints); assert len(deck) == 3; assert card_from_str('3d') in deck; assert card_from_str('4h') in deck
def test_deck_add_duplicate(): deck = Deck(cards={card_from_str('2c')}); deck.add([card_from_str('2c')]); assert len(deck) == 1
def test_deck_add_invalid():
    """Тестирует добавление невалидных карт."""
    deck = Deck(cards={card_from_str('2c')})
    deck.add([INVALID_CARD, None, -5])
    # --- ИСПРАВЛЕНО: Длина не должна измениться ---
    assert len(deck) == 1
def test_deck_copy(): deck1 = Deck(); deck1.deal(10); deck2 = deck1.copy(); assert deck1 is not deck2; assert deck1.cards is not deck2.cards; assert len(deck1) == len(deck2); assert deck1.cards == deck2.cards; deck2.deal(5); assert len(deck1) == 42; assert len(deck2) == 37
def test_deck_len(): assert len(Deck()) == 52; assert len(Deck(cards=set())) == 0; deck = Deck(); deck.deal(10); assert len(deck) == 42
def test_deck_contains(): deck = Deck(); ace_spades = card_from_str('As'); two_clubs = card_from_str('2c'); assert ace_spades in deck; deck.remove([ace_spades]); assert ace_spades not in deck; assert two_clubs in deck; assert INVALID_CARD not in deck; assert None not in deck
def test_full_deck_completeness(): assert len(Deck.FULL_DECK_CARDS) == 52; assert all(isinstance(c, int) and c != INVALID_CARD and c >= 0 for c in Deck.FULL_DECK_CARDS)
