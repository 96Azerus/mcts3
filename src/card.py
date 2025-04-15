# src/card.py
"""
Представление и утилиты для работы с картами.
Карты представляются 32-битными целыми числами для эффективности.

Структура int:
  bits 31-16: битовая маска ранга (1 << rank_index)
  bits 15-12: масть (1, 2, 4, 8 для s, h, d, c)
  bits 11-8:  индекс ранга (0-12 для 2-A)
  bits 7-0:   простое число для ранга (для быстрого определения комбинаций)
"""
from typing import List, Optional, Dict

# --- Константы на уровне модуля ---
STR_RANKS: str = '23456789TJQKA'
INT_RANKS: range = range(13)
# Простые числа для рангов (2-A)
PRIMES: List[int] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]

RANK_CHAR_TO_INT: Dict[str, int] = {rank: i for i, rank in enumerate(STR_RANKS)}
SUIT_CHAR_TO_INT: Dict[str, int] = {'s': 1, 'h': 2, 'd': 4, 'c': 8} # Spades, Hearts, Diamonds, Clubs

INT_RANK_TO_CHAR: Dict[int, str] = {i: rank for i, rank in enumerate(STR_RANKS)}
INT_SUIT_TO_CHAR: Dict[int, str] = {1: 's', 2: 'h', 4: 'd', 8: 'c'}

# Для удобного доступа (алиас)
RANK_MAP: Dict[str, int] = RANK_CHAR_TO_INT

# Для красивого вывода
PRETTY_SUITS: Dict[int, str] = {
    1: "\u2660",  # Spades (♠)
    2: "\u2665",  # Hearts (♥) - Используем красный вариант
    4: "\u2666",  # Diamonds (♦)
    8: "\u2663"   # Clubs (♣)
}
PRETTY_REDS: List[int] = [2, 4] # Масти, которые обычно рисуются красным

# Невалидное значение карты (например, для пустых слотов)
INVALID_CARD: int = -1
CARD_PLACEHOLDER: str = "__"

class Card:
    """
    Класс-обертка для статических методов работы с целочисленным представлением карт.
    Не предназначен для создания экземпляров карт.
    """

    # --- Публичные статические методы ---

    @staticmethod
    def from_str(card_str: str) -> int:
        """
        Преобразует строковое представление карты (например, 'As', 'Td', '2c')
        в целочисленное представление.

        Args:
            card_str: Строка карты (2 символа).

        Returns:
            Целочисленное представление карты.

        Raises:
            ValueError: Если строка имеет неверный формат, ранг или масть.
        """
        if not isinstance(card_str, str) or len(card_str) != 2:
            raise ValueError(f"Invalid card string format: '{card_str}'. Expected 2 characters (e.g., 'As', 'Td').")

        rank_char = card_str[0].upper()
        suit_char = card_str[1].lower()

        rank_int = RANK_CHAR_TO_INT.get(rank_char)
        suit_int = SUIT_CHAR_TO_INT.get(suit_char)

        if rank_int is None:
            raise ValueError(f"Invalid rank character: '{rank_char}' in card string '{card_str}'. Valid ranks: {STR_RANKS}")
        if suit_int is None:
            raise ValueError(f"Invalid suit character: '{suit_char}' in card string '{card_str}'. Valid suits: {''.join(SUIT_CHAR_TO_INT.keys())}")

        # Получаем простое число для ранга
        try:
            rank_prime = PRIMES[rank_int]
        except IndexError:
            # Это не должно произойти, если RANK_CHAR_TO_INT и PRIMES согласованы
            raise ValueError(f"Internal error: Prime number not found for rank index {rank_int}")

        # Собираем 32-битное число
        bitrank = 1 << rank_int << 16 # Сдвигаем битовую маску ранга на 16 позиций влево
        suit = suit_int << 12         # Сдвигаем масть на 12 позиций
        rank = rank_int << 8          # Сдвигаем индекс ранга на 8 позиций

        # Объединяем все части с помощью побитового ИЛИ
        return bitrank | suit | rank | rank_prime

    @staticmethod
    def to_str(card_int: Optional[int]) -> str:
        """
        Преобразует целочисленное представление карты обратно в строку ('As', 'Td', '__').

        Args:
            card_int: Целочисленное представление карты или None.

        Returns:
            Строковое представление карты или CARD_PLACEHOLDER ('__'), если карта невалидна или None.
        """
        if card_int is None or card_int == INVALID_CARD:
            return CARD_PLACEHOLDER

        if not isinstance(card_int, int) or card_int < 0:
             # Добавим логгирование для отладки невалидных значений
             # print(f"Warning: Card.to_str received invalid integer: {card_int}")
             return CARD_PLACEHOLDER # Возвращаем плейсхолдер для любых невалидных int

        rank_int = Card.get_rank_int(card_int)
        suit_int = Card.get_suit_int(card_int)

        rank_char = INT_RANK_TO_CHAR.get(rank_int)
        suit_char = INT_SUIT_TO_CHAR.get(suit_int)

        if rank_char and suit_char:
            return rank_char + suit_char
        else:
            # Если ранг или масть не найдены в словарях
            # print(f"Warning: Card.to_str could not map rank/suit for int: {card_int} (rank={rank_int}, suit={suit_int})")
            return CARD_PLACEHOLDER

    @staticmethod
    def get_rank_int(card_int: int) -> int:
        """Извлекает индекс ранга (0-12) из целочисленной карты."""
        return (card_int >> 8) & 0xF

    @staticmethod
    def get_suit_int(card_int: int) -> int:
        """Извлекает целочисленную масть (1, 2, 4, 8) из целочисленной карты."""
        return (card_int >> 12) & 0xF

    @staticmethod
    def get_rank_bitmask(card_int: int) -> int:
        """Извлекает битовую маску ранга (1 << rank_int) из целочисленной карты."""
        return (card_int >> 16) & 0x1FFF # 13 бит для рангов 2-A

    @staticmethod
    def get_prime(card_int: int) -> int:
        """Извлекает простое число для ранга из целочисленной карты."""
        # Простое число хранится в младших битах
        return card_int & 0x3F # 6 бит достаточно для хранения простых чисел до 41

    @staticmethod
    def prime_product_from_hand(card_ints: List[Optional[int]]) -> int:
        """
        Вычисляет произведение простых чисел рангов для карт в руке.
        Игнорирует None или INVALID_CARD.
        """
        product = 1
        for c in card_ints:
            if c is not None and c != INVALID_CARD:
                # Используем get_prime для корректного извлечения
                product *= Card.get_prime(c)
        return product

    @staticmethod
    def prime_product_from_rankbits(rankbits: int) -> int:
        """
        Вычисляет произведение простых чисел для рангов, представленных битовой маской.
        """
        product = 1
        for i in INT_RANKS: # Перебираем все возможные ранги (0-12)
            if rankbits & (1 << i): # Если бит для этого ранга установлен
                try:
                    product *= PRIMES[i]
                except IndexError:
                     # Это не должно произойти, если INT_RANKS и PRIMES согласованы
                     print(f"Warning: prime_product_from_rankbits encountered invalid rank index {i}")
        return product

    @staticmethod
    def hand_to_int(card_strs: List[Optional[str]]) -> List[Optional[int]]:
        """
        Преобразует список строковых представлений карт в список целочисленных.
        Пропускает невалидные строки, заменяя их на None.
        """
        hand_ints = []
        for s in card_strs:
            if s is None or s == CARD_PLACEHOLDER:
                hand_ints.append(None)
            else:
                try:
                    hand_ints.append(Card.from_str(s))
                except ValueError:
                    # print(f"Warning: Could not convert card string '{s}' to int. Skipping.")
                    hand_ints.append(None) # Добавляем None для невалидных строк
        return hand_ints

    @staticmethod
    def hand_to_str(card_ints: List[Optional[int]]) -> List[str]:
        """
        Преобразует список целочисленных представлений карт в список строковых.
        """
        return [Card.to_str(c) for c in card_ints]

    @staticmethod
    def to_pretty_str(card_int: Optional[int]) -> str:
        """
        Создает красивое строковое представление карты с символами мастей (например, "[A♠]", "[K♥]").
        Пытается использовать цвет для красных мастей, если установлен termcolor.
        """
        card_str = Card.to_str(card_int)
        if card_str == CARD_PLACEHOLDER:
            return f"[{CARD_PLACEHOLDER}]"

        rank_char = card_str[0]
        suit_char = card_str[1]
        suit_int = SUIT_CHAR_TO_INT.get(suit_char)
        suit_symbol = PRETTY_SUITS.get(suit_int, '?')

        # Попытка раскрасить
        try:
            from termcolor import colored
            # Проверяем, является ли масть "красной"
            if suit_int in PRETTY_REDS:
                # Раскрашиваем только символ масти
                suit_symbol = colored(suit_symbol, "red")
        except ImportError:
            pass # termcolor не установлен, просто используем стандартные символы

        return f"[{rank_char}{suit_symbol}]"

    @staticmethod
    def print_pretty_cards(card_ints: List[Optional[int]]):
        """Печатает список карт в красивом формате в одну строку."""
        print(" ".join(Card.to_pretty_str(c) for c in card_ints))

# --- Алиасы на уровне модуля для удобства импорта ---
card_from_str = Card.from_str
card_to_str = Card.to_str
