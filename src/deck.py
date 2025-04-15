# src/deck.py
"""
Реализация колоды карт с использованием set для эффективности.
"""
import random
import sys
import traceback
from typing import List, Set, Optional

# --- ИСПРАВЛЕНО: Используем относительный импорт внутри src ---
from .card import Card, card_from_str, card_to_str, STR_RANKS, SUIT_CHAR_TO_INT, INVALID_CARD

class Deck:
    """
    Представляет стандартную 52-карточную колоду для покера.
    Использует set целочисленных представлений карт для быстрой проверки наличия и удаления.
    """
    # Создаем полный набор строк карт один раз при загрузке модуля
    _FULL_DECK_STRS: Set[str] = {r + s for r in STR_RANKS for s in SUIT_CHAR_TO_INT.keys()}
    # Создаем полный набор целочисленных представлений карт один раз
    FULL_DECK_CARDS: Set[int] = set()
    _initialization_error: bool = False

    try:
        for card_s in _FULL_DECK_STRS:
            try:
                card_int = card_from_str(card_s)
                # Проверка на всякий случай (хотя card_from_str должен быть надежным)
                if not isinstance(card_int, int) or card_int == INVALID_CARD:
                    print(f"ERROR Deck Init: Card('{card_s}') created invalid int representation: {card_int}!")
                    _initialization_error = True
                else:
                    FULL_DECK_CARDS.add(card_int)
            except Exception as e_inner:
                print(f"ERROR Deck Init: Failed to create Card('{card_s}'): {e_inner}")
                traceback.print_exc()
                _initialization_error = True

        if len(FULL_DECK_CARDS) != 52 or _initialization_error:
            print(f"CRITICAL ERROR: FULL_DECK_CARDS contains {len(FULL_DECK_CARDS)} cards (expected 52) or initialization errors occurred.")
            # Можно выбросить исключение, чтобы предотвратить запуск с неполной колодой
            # raise RuntimeError("Failed to initialize the standard 52-card deck.")
    except Exception as e_outer:
        print(f"CRITICAL ERROR during Deck class initialization: {e_outer}")
        traceback.print_exc()
        raise # Перевыбрасываем исключение, чтобы остановить приложение

    def __init__(self, cards: Optional[Set[int]] = None):
        """
        Инициализирует колоду.

        Args:
            cards (Optional[Set[int]]): Набор целочисленных карт для инициализации колоды.
                                         Если None, создается полная 52-карточная колода.
                                         Переданный set копируется.
        """
        if cards is None:
            # Копируем из предсозданного набора int
            self.cards: Set[int] = self.FULL_DECK_CARDS.copy()
        else:
            # Важно копировать переданный set
            self.cards: Set[int] = cards.copy()

    def deal(self, n: int) -> List[int]:
        """
        Раздает n случайных карт из колоды и удаляет их.

        Args:
            n (int): Количество карт для раздачи.

        Returns:
            List[int]: Список розданных карт (целочисленные представления).
                       Возвращает пустой список, если n <= 0 или карт в колоде не хватает.
        """
        if n <= 0:
            return []

        current_len = len(self.cards)
        num_to_deal = min(n, current_len) # Раздаем не больше, чем есть

        if n > current_len:
            # Используем логгер или print для предупреждения
            print(f"Warning: Deck.deal trying to deal {n} cards, only {current_len} left. Dealing {num_to_deal}.")
        if num_to_deal == 0:
             return []

        try:
            # Преобразуем set в list для random.sample (O(N) времени, но N <= 52)
            # random.sample эффективен для выбора k элементов из n без повторений
            card_list = list(self.cards)
            dealt_cards = random.sample(card_list, num_to_deal)

            # Удаляем розданные карты из set (O(k) времени, где k=num_to_deal)
            self.cards.difference_update(dealt_cards)
            return dealt_cards
        except Exception as e:
             # Логгируем неожиданную ошибку
             print(f"ERROR in Deck.deal({n}): {e}")
             traceback.print_exc()
             # Возвращаем пустой список в случае ошибки
             return []

    def remove(self, cards_to_remove: List[int]):
        """
        Удаляет конкретные карты из колоды.

        Args:
            cards_to_remove (List[int]): Список целочисленных карт для удаления.
                                          Карты, которых нет в колоде, игнорируются.
        """
        # difference_update работает эффективно (в среднем O(len(cards_to_remove)))
        self.cards.difference_update(cards_to_remove)

    def add(self, cards_to_add: List[int]):
        """
        Добавляет карты обратно в колоду (например, при откате хода).

        Args:
            cards_to_add (List[int]): Список целочисленных карт для добавления.
                                       Дубликаты игнорируются благодаря set.
        """
        # update работает эффективно (в среднем O(len(cards_to_add)))
        self.cards.update(c for c in cards_to_add if c != INVALID_CARD and isinstance(c, int))

    def get_remaining_cards(self) -> List[int]:
        """Возвращает список оставшихся карт в колоде."""
        return list(self.cards)

    def copy(self) -> 'Deck':
        """Создает копию текущего состояния колоды."""
        # Передаем копию текущего набора карт (int)
        return Deck(self.cards.copy())

    def __len__(self) -> int:
        """Возвращает количество карт, оставшихся в колоде."""
        return len(self.cards)

    def __contains__(self, card: int) -> bool:
        """Проверяет наличие карты в колоде (O(1) в среднем)."""
        return card in self.cards

    def __str__(self) -> str:
        """Строковое представление колоды (для отладки)."""
        return f"Deck({len(self.cards)} cards)"

    def __repr__(self) -> str:
        """Репрезентация объекта Deck."""
        return self.__str__()
