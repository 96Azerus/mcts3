# src/board.py
"""
Представление доски одного игрока в OFC Pineapple.
Содержит карты в трех рядах и управляет их размещением,
а также кэширует и вычисляет ранги/роялти, делегируя логику в scoring.py.
"""
from typing import List, Tuple, Dict, Optional
from collections import Counter
import copy
import traceback

# --- ИСПРАВЛЕНО: Импорты из src пакета ---
from src.card import Card, card_to_str, RANK_MAP, STR_RANKS, INVALID_CARD, CARD_PLACEHOLDER
# Импортируем ВСЕ необходимые функции и константы из scoring
from src.scoring import (
    get_hand_rank_safe, check_board_foul,
    get_fantasyland_entry_cards, check_fantasyland_stay,
    get_row_royalty, RANK_CLASS_HIGH_CARD
)

class PlayerBoard:
    """
    Представляет доску одного игрока с тремя рядами (линиями).
    """
    ROW_CAPACITY: Dict[str, int] = {'top': 3, 'middle': 5, 'bottom': 5}
    ROW_NAMES: List[str] = ['top', 'middle', 'bottom']

    def __init__(self):
        """Инициализирует пустую доску."""
        # Инициализируем ряды списками None нужной длины
        self.rows: Dict[str, List[Optional[int]]] = {
            name: [None] * capacity for name, capacity in self.ROW_CAPACITY.items()
        }
        self._cards_placed: int = 0 # Количество размещенных карт
        self.is_foul: bool = False # Флаг "мертвой" руки
        # Кэши для рангов и роялти (сбрасываются при изменении доски)
        self._cached_ranks: Dict[str, Optional[int]] = {name: None for name in self.ROW_NAMES}
        self._cached_royalties: Dict[str, Optional[int]] = {name: None for name in self.ROW_NAMES}
        self._is_complete: bool = False # Флаг завершенности доски (13 карт)

    def add_card(self, card_int: int, row_name: str, index: int) -> bool:
        """
        Добавляет карту (целочисленное представление) в УКАЗАННЫЙ слот ряда.
        Обновляет счетчик карт и сбрасывает кэши.

        Args:
            card_int (int): Целочисленное представление карты.
            row_name (str): Имя ряда ('top', 'middle', 'bottom').
            index (int): Индекс слота в ряду (0-based).

        Returns:
            bool: True при успехе, False при неудаче (неверный ряд/индекс, слот занят).
        """
        if row_name not in self.ROW_NAMES:
            # print(f"Error: Invalid row name '{row_name}'")
            return False
        if card_int == INVALID_CARD or not isinstance(card_int, int):
             # print(f"Error: Invalid card value '{card_int}'")
             return False

        capacity = self.ROW_CAPACITY[row_name]
        if not (0 <= index < capacity):
            # print(f"Error: Index {index} out of bounds for row '{row_name}' (0-{capacity-1}).")
            return False

        # Проверяем, свободен ли слот
        if self.rows[row_name][index] is not None:
            # print(f"Warning: Slot {row_name}[{index}] is already occupied. Cannot add {card_to_str(card_int)}.")
            return False

        # Добавляем карту
        self.rows[row_name][index] = card_int
        self._cards_placed += 1
        self._is_complete = (self._cards_placed == 13)

        # Сбрасываем кэши при любом изменении доски
        self._reset_caches()
        # Фол будет пересчитан при необходимости (при запросе рангов/роялти или в конце раунда)
        # Сбрасываем флаг фола, так как доска изменилась
        self.is_foul = False
        return True

    def remove_card(self, row_name: str, index: int) -> Optional[int]:
        """
        Удаляет карту из указанного слота (например, для отмены хода в UI).
        Обновляет счетчик карт и сбрасывает кэши.

        Args:
            row_name (str): Имя ряда.
            index (int): Индекс слота.

        Returns:
            Optional[int]: Удаленная карта (int) или None, если слот был пуст или некорректен.
        """
        if row_name not in self.ROW_NAMES or not (0 <= index < self.ROW_CAPACITY[row_name]):
            return None # Некорректный ряд или индекс

        card_int = self.rows[row_name][index]
        if card_int is not None:
            self.rows[row_name][index] = None # Очищаем слот
            self._cards_placed -= 1
            self._is_complete = False # Доска больше не полная
            self._reset_caches()      # Сбрасываем кэши
            self.is_foul = False      # Сбрасываем флаг фола
            return card_int
        return None # Слот был пуст

    def set_full_board(self, top: List[int], middle: List[int], bottom: List[int]):
        """
        Устанавливает всю доску из готовых списков карт (например, для Фантазии).
        Проверяет корректность входных данных и уникальность карт.

        Args:
            top (List[int]): Список из 3 карт для верхнего ряда.
            middle (List[int]): Список из 5 карт для среднего ряда.
            bottom (List[int]): Список из 5 карт для нижнего ряда.

        Raises:
            ValueError: Если количество карт неверно, карты невалидны или есть дубликаты.
        """
        if len(top) != 3 or len(middle) != 5 or len(bottom) != 5:
            raise ValueError(f"Incorrect number of cards provided for setting full board (top={len(top)}, mid={len(middle)}, bot={len(bottom)}).")

        all_cards: List[int] = []
        card_lists = {'top': top, 'middle': middle, 'bottom': bottom}

        # Проверяем валидность карт и собираем все карты для проверки уникальности
        for row_name, card_list in card_lists.items():
             for i, card_int in enumerate(card_list):
                  if not isinstance(card_int, int) or card_int == INVALID_CARD or card_int < 0:
                       raise ValueError(f"Invalid card value '{card_int}' provided in row '{row_name}' at index {i}.")
                  all_cards.append(card_int)

        # Проверяем уникальность всех 13 карт
        if len(all_cards) != len(set(all_cards)):
            counts = Counter(all_cards)
            duplicates = {card: count for card, count in counts.items() if count > 1}
            raise ValueError(f"Duplicate cards provided for setting full board: {duplicates}")

        # Устанавливаем карты в ряды (создаем копии списков)
        self.rows['top'] = list(top)
        self.rows['middle'] = list(middle)
        self.rows['bottom'] = list(bottom)

        # Обновляем состояние доски
        self._cards_placed = 13
        self._is_complete = True
        self._reset_caches() # Сбрасываем кэши
        self.check_and_set_foul() # Проверяем фол сразу после установки полной доски

    def get_row_cards(self, row_name: str) -> List[int]:
        """Возвращает список действительных карт (int) в указанном ряду."""
        if row_name not in self.rows:
            return []
        # Возвращаем только не-None и не-INVALID_CARD карты
        return [card for card in self.rows[row_name] if card is not None and card != INVALID_CARD]

    def is_row_full(self, row_name: str) -> bool:
        """Проверяет, заполнен ли указанный ряд картами."""
        if row_name not in self.rows:
            return False
        # Проверяем, что все слоты в ряду не None и не INVALID_CARD
        return all(slot is not None and slot != INVALID_CARD for slot in self.rows[row_name])

    def get_available_slots(self) -> List[Tuple[str, int]]:
        """Возвращает список доступных (пустых) слотов в формате ('row_name', index)."""
        slots = []
        for row_name in self.ROW_NAMES:
            for i, card in enumerate(self.rows[row_name]):
                # Слот доступен, если он None (не INVALID_CARD)
                if card is None:
                    slots.append((row_name, i))
        return slots

    def get_total_cards(self) -> int:
        """Возвращает общее количество карт, размещенных на доске."""
        return self._cards_placed

    def is_complete(self) -> bool:
        """Проверяет, размещены ли все 13 карт на доске."""
        return self._is_complete

    def _reset_caches(self):
        """Сбрасывает внутренние кэши рангов и роялти."""
        self._cached_ranks = {name: None for name in self.ROW_NAMES}
        self._cached_royalties = {name: None for name in self.ROW_NAMES}

    def _get_rank(self, row_name: str) -> int:
        """
        Получает ранг руки для указанного ряда (из кэша или вычисляет).
        Использует get_hand_rank_safe из scoring.py.

        Args:
            row_name (str): Имя ряда.

        Returns:
            int: Ранг руки (меньше = лучше). Возвращает плохой ранг при ошибке.
        """
        if row_name not in self.ROW_NAMES:
            # Возвращаем ранг хуже худшей возможной руки
            return RANK_CLASS_HIGH_CARD + 300

        # Проверяем кэш
        if self._cached_ranks[row_name] is None:
            # Передаем список с None как есть, get_hand_rank_safe обработает
            cards_with_none = self.rows[row_name]
            # Вызываем функцию из scoring
            self._cached_ranks[row_name] = get_hand_rank_safe(cards_with_none)

        # Возвращаем значение из кэша (или только что вычисленное)
        rank = self._cached_ranks[row_name]
        # Добавим проверку на None на всякий случай
        return rank if rank is not None else RANK_CLASS_HIGH_CARD + 400

    def check_and_set_foul(self) -> bool:
        """
        Проверяет доску на фол (нарушение порядка линий) и устанавливает флаг is_foul.
        Вызывать эту функцию имеет смысл только для полной доски.
        Использует check_board_foul из scoring.py.

        Returns:
            bool: Текущее значение флага is_foul.
        """
        # Не проверяем фол, если доска не полная
        if not self.is_complete():
            # Убедимся, что флаг сброшен, если доска не полная
            if self.is_foul:
                 self.is_foul = False
                 self._reset_caches() # Сбросим кэши, если фол был отменен
            return False

        # Используем функцию из scoring.py, передавая списки с None
        current_foul_status = check_board_foul(
            self.rows['top'],
            self.rows['middle'],
            self.rows['bottom']
        )

        # Обновляем флаг и кэш роялти, если статус фола изменился
        if current_foul_status != self.is_foul:
            self.is_foul = current_foul_status
            if self.is_foul:
                # Если рука стала фолом, обнуляем кэш роялти
                self._cached_royalties = {'top': 0, 'middle': 0, 'bottom': 0}
            else:
                 # Если фол был снят (маловероятно без изменений), сбрасываем кэш
                 self._reset_caches()

        return self.is_foul

    def get_royalties(self) -> Dict[str, int]:
        """
        Считает и возвращает роялти для каждой линии, используя кэш.
        Автоматически проверяет фол для полной доски перед расчетом.
        Использует get_row_royalty из scoring.py.

        Returns:
            Dict[str, int]: Словарь с роялти для 'top', 'middle', 'bottom'.
                            Возвращает нули, если рука "мертвая" (фол).
        """
        # Сначала проверяем фол, если доска полная.
        # check_and_set_foul обновит self.is_foul и кэш роялти при необходимости.
        if self.is_complete() and self.check_and_set_foul():
            # Возвращаем нули, если фол
            return {'top': 0, 'middle': 0, 'bottom': 0}

        # Пересчитываем роялти для рядов, если они не в кэше
        for row_name in self.ROW_NAMES:
            if self._cached_royalties[row_name] is None:
                # Передаем список с None, get_row_royalty обработает
                cards_with_none = self.rows[row_name]
                # Вызываем функцию из scoring
                self._cached_royalties[row_name] = get_row_royalty(cards_with_none, row_name)

        # Возвращаем копию кэша (он уже должен быть обнулен, если был фол)
        # Добавим проверку на None в кэше на всякий случай
        return {
            row: self._cached_royalties.get(row, 0) or 0
            for row in self.ROW_NAMES
        }


    def get_total_royalty(self) -> int:
        """Возвращает сумму роялти по всем линиям, учитывая фол."""
        # Вызов get_royalties() обновит кэш и учтет фол, если нужно
        royalties_dict = self.get_royalties()
        return sum(royalties_dict.values())

    def get_fantasyland_qualification_cards(self) -> int:
        """
        Проверяет, квалифицируется ли доска для входа в Fantasyland (Progressive).
        Возвращает количество карт для ФЛ (0, 14, 15, 16, 17).
        Автоматически проверяет фол.
        Использует get_fantasyland_entry_cards из scoring.py.

        Returns:
            int: Количество карт для Фантазии (0, если нет квалификации или фол).
        """
        if not self.is_complete(): return 0
        if self.check_and_set_foul(): return 0 # Фол не дает ФЛ

        # Вызываем функцию из scoring
        return get_fantasyland_entry_cards(self.rows['top'])

    def check_fantasyland_stay_conditions(self) -> bool:
        """
        Проверяет, выполнены ли условия для удержания Fantasyland (Re-Fantasy).
        (Сет на топе или Каре+ на боттоме, без фола).
        Автоматически проверяет фол.
        Использует check_fantasyland_stay из scoring.py.

        Returns:
            bool: True, если условия выполнены, иначе False.
        """
        if not self.is_complete(): return False
        if self.check_and_set_foul(): return False # Фол не позволяет удержать ФЛ

        # Вызываем функцию из scoring
        return check_fantasyland_stay(
            self.rows['top'],
            self.rows['middle'],
            self.rows['bottom']
        )

    def get_board_state_tuple(self) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
        """
        Возвращает неизменяемое представление доски для использования в качестве ключа словаря или элемента set (например, в MCTS).
        Карты представлены строками (или CARD_PLACEHOLDER).
        Карты внутри каждого ряда сортируются для каноничности представления.

        Returns:
            Tuple[Tuple[str, ...], ...]: Кортеж из трех кортежей (top, middle, bottom),
                                         содержащих отсортированные строки карт.
        """
        # Функция для ключа сортировки: сначала по рангу (от старшего к младшему), потом по масти
        # Используем RANK_MAP из src.card
        def sort_key(card_str: str) -> Tuple[int, str]:
            if card_str == CARD_PLACEHOLDER:
                return (-1, '') # Плейсхолдеры идут в конец
            try:
                rank_char = card_str[0].upper()
                suit_char = card_str[1].lower()
                # Используем отрицательное значение ранга для сортировки по убыванию
                rank_val = -RANK_MAP.get(rank_char, -2)
                return (rank_val, suit_char)
            except IndexError:
                return (-1, '') # В случае некорректной строки

        rows_as_str_tuples: Dict[str, Tuple[str, ...]] = {}
        for r_name in self.ROW_NAMES:
            row_str_list: List[str] = []
            for card_int in self.rows[r_name]:
                # Преобразуем int Card в строку или CARD_PLACEHOLDER
                row_str_list.append(card_to_str(card_int))
            # Сортируем строки карт внутри ряда
            sorted_row = tuple(sorted(row_str_list, key=sort_key))
            rows_as_str_tuples[r_name] = sorted_row

        return (rows_as_str_tuples['top'], rows_as_str_tuples['middle'], rows_as_str_tuples['bottom'])

    def copy(self) -> 'PlayerBoard':
        """Создает глубокую копию объекта доски."""
        new_board = PlayerBoard()
        # Копируем ряды (списки int копируются по значению при создании нового списка)
        new_board.rows = {r: list(cards) for r, cards in self.rows.items()}
        # Копируем остальные атрибуты
        new_board._cards_placed = self._cards_placed
        new_board.is_foul = self.is_foul
        new_board._is_complete = self._is_complete
        # Копируем кэши (они содержат простые типы или None)
        new_board._cached_ranks = self._cached_ranks.copy()
        new_board._cached_royalties = self._cached_royalties.copy()
        return new_board

    def __str__(self) -> str:
        """Возвращает строковое представление доски для вывода."""
        s = ""
        max_len = max(len(self.rows[r_name]) for r_name in self.ROW_NAMES)
        for r_name in self.ROW_NAMES:
            # Преобразуем int Card в строку или CARD_PLACEHOLDER
            row_str = [card_to_str(c) for c in self.rows[r_name]]
            # Дополняем пробелами для выравнивания (используем плейсхолдер)
            row_str += [CARD_PLACEHOLDER] * (max_len - len(row_str))
            s += f"{r_name.upper():<6}: " + " ".join(f"{c:^2}" for c in row_str) + "\n"

        # Добавляем информацию о состоянии доски
        s += f"Cards: {self._cards_placed}/13"
        if self.is_complete():
            # Проверяем фол перед выводом роялти
            is_foul = self.check_and_set_foul()
            s += f", Complete: Yes, Foul: {is_foul}"
            # Вызываем get_royalties для получения актуальных значений (учитывая фол)
            royalties_dict = self.get_royalties()
            total_royalty = sum(royalties_dict.values())
            s += f", Royalties: {total_royalty} {royalties_dict}"
        else:
            s += f", Complete: No"
            # Показываем текущий статус фола, даже если не полная (хотя он должен быть False)
            s += f", Foul: {self.is_foul}"
        return s.strip()

    def __repr__(self) -> str:
        """Возвращает репрезентацию объекта PlayerBoard."""
        return f"PlayerBoard(Cards={self._cards_placed}, Complete={self._is_complete}, Foul={self.is_foul})"
