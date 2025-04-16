# src/game_state.py
"""
Определяет класс GameState, управляющий полным состоянием игры
OFC Pineapple для двух игроков, включая Progressive Fantasyland
и инкапсулированную логику управления ходом игры.
"""
import copy
import random
import sys
import traceback
from itertools import combinations, permutations
from typing import List, Tuple, Optional, Set, Dict, Any

# Импорты из src пакета
from src.card import Card, card_to_str, card_from_str, INVALID_CARD, CARD_PLACEHOLDER
from src.deck import Deck
from src.board import PlayerBoard
from src.scoring import calculate_headsup_score # Только для финального счета

class GameState:
    """
    Представляет полное состояние игры OFC Pineapple для двух игроков.
    Инкапсулирует основную логику перехода состояний.
    """
    NUM_PLAYERS = 2

    # (__init__ и методы получения информации без изменений)
    def __init__(self, boards: Optional[List[PlayerBoard]] = None, deck: Optional[Deck] = None, private_discard: Optional[List[List[int]]] = None, dealer_idx: int = 0, current_player_idx: Optional[int] = None, street: int = 0, current_hands: Optional[Dict[int, Optional[List[int]]]] = None, fantasyland_status: Optional[List[bool]] = None, next_fantasyland_status: Optional[List[bool]] = None, fantasyland_cards_to_deal: Optional[List[int]] = None, is_fantasyland_round: Optional[bool] = None, fantasyland_hands: Optional[List[Optional[List[int]]]] = None, _player_acted_this_street: Optional[List[bool]] = None, _player_finished_round: Optional[List[bool]] = None):
        self.boards: List[PlayerBoard] = boards if boards is not None else [PlayerBoard() for _ in range(self.NUM_PLAYERS)]; self.deck: Deck = deck if deck is not None else Deck(); self.private_discard: List[List[int]] = private_discard if private_discard is not None else [[] for _ in range(self.NUM_PLAYERS)];
        if not (0 <= dealer_idx < self.NUM_PLAYERS): raise ValueError(f"Invalid dealer index: {dealer_idx}"); self.dealer_idx: int = dealer_idx;
        self._internal_current_player_idx = (dealer_idx + 1) % self.NUM_PLAYERS if current_player_idx is None else current_player_idx; self.street: int = street;
        self.current_hands: Dict[int, Optional[List[int]]] = current_hands if current_hands is not None else {i: None for i in range(self.NUM_PLAYERS)}; self.fantasyland_hands: List[Optional[List[int]]] = fantasyland_hands if fantasyland_hands is not None else [None] * self.NUM_PLAYERS;
        self.fantasyland_status: List[bool] = fantasyland_status if fantasyland_status is not None else [False] * self.NUM_PLAYERS; self.next_fantasyland_status: List[bool] = next_fantasyland_status if next_fantasyland_status is not None else list(self.fantasyland_status); self.fantasyland_cards_to_deal: List[int] = fantasyland_cards_to_deal if fantasyland_cards_to_deal is not None else [0] * self.NUM_PLAYERS; self.is_fantasyland_round: bool = any(self.fantasyland_status) if is_fantasyland_round is None else is_fantasyland_round;
        self._player_acted_this_street: List[bool] = _player_acted_this_street if _player_acted_this_street is not None else [False] * self.NUM_PLAYERS; self._player_finished_round: List[bool] = _player_finished_round if _player_finished_round is not None else [b.is_complete() for b in self.boards];
        self._last_player_acted: Optional[int] = None

    def get_player_board(self, player_idx: int) -> PlayerBoard: return self.boards[player_idx]
    def get_player_hand(self, player_idx: int) -> Optional[List[int]]: return self.fantasyland_hands[player_idx] if self.is_fantasyland_round and self.fantasyland_status[player_idx] else self.current_hands.get(player_idx)
    def is_round_over(self) -> bool: return all(self._player_finished_round)
    def get_terminal_score(self) -> int: return 0 if not self.is_round_over() else calculate_headsup_score(self.boards[0], self.boards[1])
    def get_known_dead_cards(self, perspective_player_idx: int) -> Set[int]: dead_cards: Set[int] = set(); [dead_cards.add(c) for b in self.boards for r in b.ROW_NAMES for c in b.rows[r] if c is not None and c != INVALID_CARD]; hand = self.get_player_hand(perspective_player_idx); [dead_cards.add(c) for c in hand if c is not None and c != INVALID_CARD] if hand else None; [dead_cards.add(c) for c in self.private_discard[perspective_player_idx] if c is not None and c != INVALID_CARD]; return dead_cards
    def get_player_to_move(self) -> int:
        if self.is_round_over(): return -1
        if self.is_fantasyland_round: [i for i in range(self.NUM_PLAYERS) if not self._player_finished_round[i] and self.get_player_hand(i)]; return i if 'i' in locals() else -1 # Возвращаем первого найденного
        else: p_idx = self._internal_current_player_idx; return p_idx if not self._player_finished_round[p_idx] and self.get_player_hand(p_idx) else -1 if not self._player_finished_round[(p_idx + 1) % self.NUM_PLAYERS] and self.get_player_hand((p_idx + 1) % self.NUM_PLAYERS) else -1

    # --- Методы для изменения состояния ---
    # (start_new_round без изменений)
    def start_new_round(self, dealer_button_idx: int): fl_status = list(self.next_fantasyland_status); fl_cards = list(self.fantasyland_cards_to_deal); self.__init__(dealer_idx=dealer_button_idx, fantasyland_status=fl_status, fantasyland_cards_to_deal=fl_cards); self.street = 1; self._internal_current_player_idx = (self.dealer_idx + 1) % self.NUM_PLAYERS; self._last_player_acted = None;
        if self.is_fantasyland_round: self._deal_fantasyland_hands(); [self._deal_street_to_player(i) for i in range(self.NUM_PLAYERS) if not self.fantasyland_status[i]]
        else: self._deal_street_to_player(self._internal_current_player_idx); self._deal_cards_if_needed()

    def apply_action(self, player_idx: int, action: Any) -> 'GameState':
        """Применяет легальное действие ОБЫЧНОГО ХОДА. Выбрасывает исключение при ошибке."""
        new_state = self.copy(); board = new_state.boards[player_idx]
        if new_state.is_fantasyland_round and new_state.fantasyland_status[player_idx]: raise RuntimeError("Use apply_fantasyland_placement/foul for FL.")
        current_hand = new_state.current_hands.get(player_idx);
        if not current_hand: raise RuntimeError(f"Player {player_idx} has no hand.")
        # --- ИЗМЕНЕНИЕ: Убрали try...except, ошибки будут выброшены ---
        if new_state.street == 1:
            if len(current_hand) != 5: raise ValueError("Invalid hand size street 1.")
            placements, _ = action;
            if len(placements) != 5: raise ValueError("Street 1 needs 5 placements.")
            placed_cards = set(); hand_set = set(current_hand)
            for card, row, idx in placements:
                if card not in hand_set or card in placed_cards: raise ValueError(f"Invalid/duplicate card {card_to_str(card)}.")
                if not board.add_card(card, row, idx): raise RuntimeError(f"Failed add card {card_to_str(card)} to {row}[{idx}].")
                placed_cards.add(card)
        elif 2 <= new_state.street <= 5:
            if len(current_hand) != 3: raise ValueError(f"Invalid hand size street {new_state.street}.")
            p1, p2, discard = action; c1, r1, i1 = p1; c2, r2, i2 = p2
            action_cards = {c1, c2, discard}; hand_set = set(current_hand)
            if len(action_cards) != 3 or action_cards != hand_set: raise ValueError("Action cards mismatch hand.")
            if not board.add_card(c1, r1, i1): raise RuntimeError(f"Failed add card {card_to_str(c1)} to {r1}[{i1}].")
            try:
                if not board.add_card(c2, r2, i2): raise RuntimeError(f"Failed add card {card_to_str(c2)} to {r2}[{i2}].")
            except Exception as e:
                board.remove_card(r1, i1); raise e # Откат и перевыброс
            new_state.private_discard[player_idx].append(discard)
        else: raise ValueError(f"Invalid street {new_state.street}.")
        new_state.current_hands[player_idx] = None; new_state._player_acted_this_street[player_idx] = True; new_state._last_player_acted = player_idx
        if board.is_complete(): new_state._player_finished_round[player_idx] = True; new_state._check_foul_and_update_fl_status(player_idx)
        return new_state
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---

    # (apply_fantasyland_placement и apply_fantasyland_foul без изменений, они уже используют try/except и фол при ошибке)
    def apply_fantasyland_placement(self, player_idx: int, placement: Dict[str, List[int]], discarded: List[int]) -> 'GameState':
        new_state = self.copy(); board = new_state.boards[player_idx]
        if not new_state.is_fantasyland_round or not new_state.fantasyland_status[player_idx]: raise RuntimeError("Not in FL.")
        original_hand = new_state.fantasyland_hands[player_idx];
        if not original_hand: raise RuntimeError("No FL hand found.")
        try:
            placed_list = [c for r in placement.values() for c in r if c is not None and c != INVALID_CARD]; placed_set = set(placed_list)
            discard_set = set(discarded); hand_set = set(original_hand); n_hand = len(original_hand); expected_discard = n_hand - 13
            if len(placed_list) != 13 or len(placed_set) != 13: raise ValueError("Invalid placement count.")
            if len(discarded) != expected_discard or len(discard_set) != expected_discard: raise ValueError("Invalid discard count.")
            if not placed_set.isdisjoint(discard_set): raise ValueError("Overlap placed/discarded.")
            if placed_set.union(discard_set) != hand_set: raise ValueError("Cards mismatch hand.")
            board.set_full_board(placement['top'], placement['middle'], placement['bottom'])
            new_state.private_discard[player_idx].extend(discarded); new_state.fantasyland_hands[player_idx] = None
            new_state._player_finished_round[player_idx] = True; new_state._check_foul_and_update_fl_status(player_idx); new_state._last_player_acted = player_idx
            return new_state
        except (ValueError, TypeError, RuntimeError) as e:
            print(f"Error applying FL placement P{player_idx}: {e}", file=sys.stderr)
            print("Applying FL foul due to error.", file=sys.stderr)
            return new_state.apply_fantasyland_foul(player_idx, original_hand)

    def apply_fantasyland_foul(self, player_idx: int, hand_to_discard: List[int]) -> 'GameState':
        new_state = self.copy(); board = new_state.boards[player_idx]
        board.is_foul = True; board._cards_placed = 0; board._is_complete = False
        board.rows = {n: [None] * cap for n, cap in PlayerBoard.ROW_CAPACITY.items()}; board._reset_caches(); board._cached_royalties = {'top': 0, 'middle': 0, 'bottom': 0}
        new_state.private_discard[player_idx].extend(hand_to_discard); new_state.fantasyland_hands[player_idx] = None
        new_state._player_finished_round[player_idx] = True; new_state.next_fantasyland_status[player_idx] = False; new_state.fantasyland_cards_to_deal[player_idx] = 0; new_state._last_player_acted = player_idx
        return new_state

    # (Метод advance_state без изменений)
    def advance_state(self) -> 'GameState':
         if self.is_round_over(): return self
         new_state = self.copy()
         initial_state_repr = self.get_state_representation()
         if not new_state.is_fantasyland_round:
              last_acted = new_state._last_player_acted; current_p = new_state._internal_current_player_idx
              player_finished_after_action = new_state._player_finished_round[last_acted] if last_acted is not None else False
              other_p = (last_acted + 1) % new_state.NUM_PLAYERS if last_acted is not None else -1
              both_acted_or_finished = all(new_state._player_acted_this_street) or (last_acted is not None and player_finished_after_action and new_state._player_finished_round[other_p])
              if both_acted_or_finished and new_state.street < 5:
                   new_state.street += 1; new_state._player_acted_this_street = [False] * new_state.NUM_PLAYERS
                   new_state._internal_current_player_idx = (new_state.dealer_idx + 1) % new_state.NUM_PLAYERS; new_state._last_player_acted = None
              elif last_acted is not None and last_acted == current_p:
                   next_p = (current_p + 1) % new_state.NUM_PLAYERS
                   if not new_state._player_finished_round[next_p]: new_state._internal_current_player_idx = next_p
         new_state._deal_cards_if_needed()
         return new_state

    # --- Внутренние методы ---
    # (Методы _deal_cards_if_needed, _deal_street_to_player, _deal_fantasyland_hands, _check_foul_and_update_fl_status без изменений)
    def _deal_cards_if_needed(self): players_to_deal = [];
        if self.is_fantasyland_round: [players_to_deal.append(p_idx) for p_idx in range(self.NUM_PLAYERS) if not self.fantasyland_status[p_idx] and not self._player_finished_round[p_idx] and self.current_hands.get(p_idx) is None]
        else: p_idx = self._internal_current_player_idx;
            if not self._player_finished_round[p_idx] and self.current_hands.get(p_idx) is None: players_to_deal.append(p_idx)
            elif self._last_player_acted is None and self.street > 1: other_p = (p_idx + 1) % self.NUM_PLAYERS;
                if not self._player_finished_round[other_p] and self.current_hands.get(other_p) is None: players_to_deal.append(other_p)
        for p_idx_deal in players_to_deal: self._deal_street_to_player(p_idx_deal)
    def _deal_street_to_player(self, player_idx: int):
        if self._player_finished_round[player_idx] or self.current_hands.get(player_idx) is not None: return
        num_cards = 5 if self.street == 1 else 3;
        if self.street > 5: return
        try: dealt = self.deck.deal(num_cards); self.current_hands[player_idx] = dealt
        except Exception as e: print(f"Error dealing St{self.street} P{player_idx}: {e}", file=sys.stderr); self.current_hands[player_idx] = []
    def _deal_fantasyland_hands(self):
        for i in range(self.NUM_PLAYERS):
            if self.fantasyland_status[i]:
                num_cards = self.fantasyland_cards_to_deal[i];
                if not (14 <= num_cards <= 17): num_cards = 14
                try: dealt = self.deck.deal(num_cards); self.fantasyland_hands[i] = dealt
                except Exception as e: print(f"Error dealing FL P{i}: {e}", file=sys.stderr); self.fantasyland_hands[i] = []
    def _check_foul_and_update_fl_status(self, player_idx: int):
        board = self.boards[player_idx];
        if not board.is_complete(): return
        is_foul = board.check_and_set_foul(); self.next_fantasyland_status[player_idx] = False; self.fantasyland_cards_to_deal[player_idx] = 0
        if not is_foul:
            if self.fantasyland_status[player_idx]:
                if board.check_fantasyland_stay_conditions(): self.next_fantasyland_status[player_idx] = True; self.fantasyland_cards_to_deal[player_idx] = 14
            else:
                fl_cards = board.get_fantasyland_qualification_cards()
                if fl_cards > 0: self.next_fantasyland_status[player_idx] = True; self.fantasyland_cards_to_deal[player_idx] = fl_cards

    # --- Методы для MCTS и Сериализации ---
    # (get_legal_actions_for_player, _get_legal_actions_street1, _get_legal_actions_pineapple без изменений в логике, но с исправленной хешируемостью)
    def get_legal_actions_for_player(self, player_idx: int) -> List[Any]:
        if self._player_finished_round[player_idx]: return []
        if self.is_fantasyland_round and self.fantasyland_status[player_idx]: hand = self.fantasyland_hands[player_idx]; return [("FANTASYLAND_INPUT", tuple(hand))] if hand else [] # Возвращаем tuple
        hand = self.current_hands.get(player_idx);
        if not hand: return []
        if self.street == 1: return self._get_legal_actions_street1(player_idx, hand) if len(hand) == 5 else []
        elif 2 <= self.street <= 5: return self._get_legal_actions_pineapple(player_idx, hand) if len(hand) == 3 else []
        else: return []

    def _get_legal_actions_street1(self, player_idx: int, hand: List[int]) -> List[Tuple[Tuple[Tuple[int, str, int], ...], Tuple]]:
        board = self.boards[player_idx]; slots = board.get_available_slots();
        if len(slots) < 5: return []
        actions = []; MAX_ACTIONS = 500; count = 0
        for slot_combo in combinations(slots, 5):
             if count >= MAX_ACTIONS: break
             card_perm = random.sample(hand, len(hand))
             placement = [(card_perm[i], slot_combo[i][0], slot_combo[i][1]) for i in range(5)]
             actions.append((tuple(placement), tuple())); count += 1 # Конвертируем в tuple
        return actions

    def _get_legal_actions_pineapple(self, player_idx: int, hand: List[int]) -> List[Tuple[Tuple[int, str, int], Tuple[int, str, int], int]]:
        board = self.boards[player_idx]; slots = board.get_available_slots();
        if len(slots) < 2: return []
        actions = []
        for i in range(3):
            discard = hand[i]; cards_place = [hand[j] for j in range(3) if i != j]; c1, c2 = cards_place[0], cards_place[1]
            for s1_info, s2_info in combinations(slots, 2):
                r1, i1 = s1_info; r2, i2 = s2_info
                actions.append(((c1, r1, i1), (c2, r2, i2), discard))
                actions.append(((c2, r1, i1), (c1, r2, i2), discard))
        return actions

    # (get_state_representation, copy, __hash__, __eq__, to_dict, from_dict без изменений)
    def get_state_representation(self) -> tuple: b_t = tuple(b.get_board_state_tuple() for b in self.boards); ch_t = tuple(tuple(sorted(card_to_str(c) for c in self.current_hands.get(i) if c is not None)) if self.current_hands.get(i) else None for i in range(self.NUM_PLAYERS)); fh_t = tuple(tuple(sorted(card_to_str(c) for c in self.fantasyland_hands[i] if c is not None)) if self.fantasyland_hands[i] else None for i in range(self.NUM_PLAYERS)); d_t = tuple(tuple(sorted(card_to_str(c) for c in p_d)) for p_d in self.private_discard); return (b_t, d_t, self.dealer_idx, self._internal_current_player_idx, self.street, tuple(self.fantasyland_status), self.is_fantasyland_round, tuple(self.fantasyland_cards_to_deal), ch_t, fh_t, tuple(self._player_acted_this_street), tuple(self._player_finished_round))
    def copy(self) -> 'GameState': new = GameState.__new__(GameState); new.boards = [b.copy() for b in self.boards]; new.deck = self.deck.copy(); new.private_discard = [list(p) for p in self.private_discard]; new.dealer_idx = self.dealer_idx; new._internal_current_player_idx = self._internal_current_player_idx; new.street = self.street; new.current_hands = {idx: list(h) if h else None for idx, h in self.current_hands.items()}; new.fantasyland_hands = [list(h) if h else None for h in self.fantasyland_hands]; new.fantasyland_status = list(self.fantasyland_status); new.next_fantasyland_status = list(self.next_fantasyland_status); new.fantasyland_cards_to_deal = list(self.fantasyland_cards_to_deal); new.is_fantasyland_round = self.is_fantasyland_round; new._player_acted_this_street = list(self._player_acted_this_street); new._player_finished_round = list(self._player_finished_round); new._last_player_acted = self._last_player_acted; return new
    def __hash__(self): return hash(self.get_state_representation())
    def __eq__(self, other): return isinstance(other, GameState) and self.get_state_representation() == other.get_state_representation()
    def to_dict(self) -> Dict[str, Any]: boards = [{'rows': {r: [card_to_str(c) for c in b.rows[r]] for r in b.ROW_NAMES}, '_cards_placed': b._cards_placed, 'is_foul': b.is_foul, '_is_complete': b._is_complete} for b in self.boards]; return {"boards": boards, "private_discard": [[card_to_str(c) for c in p] for p in self.private_discard], "dealer_idx": self.dealer_idx, "_internal_current_player_idx": self._internal_current_player_idx, "street": self.street, "current_hands": {i: [card_to_str(c) for c in h] if h else None for i, h in self.current_hands.items()}, "fantasyland_status": self.fantasyland_status, "next_fantasyland_status": self.next_fantasyland_status, "fantasyland_cards_to_deal": self.fantasyland_cards_to_deal, "is_fantasyland_round": self.is_fantasyland_round, "fantasyland_hands": [[card_to_str(c) for c in h] if h else None for h in self.fantasyland_hands], "_player_acted_this_street": self._player_acted_this_street, "_player_finished_round": self._player_finished_round, "_last_player_acted": self._last_player_acted}
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameState': num_p = len(data.get("boards", [])); num_p = cls.NUM_PLAYERS if num_p == 0 else num_p; boards = []; known_cards: Set[int] = set();
        for b_data in data.get("boards", [{} for _ in range(num_p)]): b = PlayerBoard(); rows_raw = b_data.get('rows', {}); cards_count = 0;
            for r_name in b.ROW_NAMES: r_cards = []; cap = b.ROW_CAPACITY[r_name];
                for c_str in rows_raw.get(r_name, [CARD_PLACEHOLDER] * cap):
                    if c_str != CARD_PLACEHOLDER: try: c_int = card_from_str(c_str); r_cards.append(c_int); known_cards.add(c_int); cards_count += 1; except ValueError: r_cards.append(None)
                    else: r_cards.append(None)
                b.rows[r_name] = (r_cards + [None] * cap)[:cap]; b._cards_placed = b_data.get('_cards_placed', cards_count); b.is_foul = b_data.get('is_foul', False); b._is_complete = b_data.get('_is_complete', b._cards_placed == 13); b._reset_caches(); boards.append(b)
        p_discard = []; [p_d.append(c) for p_d_strs in data.get("private_discard", [[] for _ in range(num_p)]) if (p_d := []) is not None for cs in p_d_strs if (c := card_from_str(cs)) != INVALID_CARD and known_cards.add(c) or p_discard.append(p_d)]; c_hands = {}; raw_c_hands = data.get("current_hands", {});
        for i in range(num_p): h_strs = raw_c_hands.get(str(i)); h = []; [h.append(c) for cs in h_strs if (c := card_from_str(cs)) != INVALID_CARD and known_cards.add(c)] if h_strs else None; c_hands[i] = h if h_strs else None
        fl_hands = []; raw_fl_hands = data.get("fantasyland_hands", [None] * num_p);
        for i in range(num_p): h_strs = raw_fl_hands[i] if i < len(raw_fl_hands) else None; h = []; [h.append(c) for cs in h_strs if (c := card_from_str(cs)) != INVALID_CARD and known_cards.add(c)] if h_strs else None; fl_hands.append(h if h_strs else None)
        remain_cards = Deck.FULL_DECK_CARDS - known_cards; deck = Deck(cards=remain_cards); def_b = [False] * num_p; def_i = [0] * num_p; d_idx = data.get("dealer_idx", 0); int_cp_idx = data.get("_internal_current_player_idx", (d_idx + 1) % num_p); st = data.get("street", 0); fl_stat = data.get("fantasyland_status", list(def_b)); next_fl_stat = data.get("next_fantasyland_status", list(def_b)); fl_cards = data.get("fantasyland_cards_to_deal", list(def_i)); is_fl = data.get("is_fantasyland_round", any(fl_stat)); acted = data.get("_player_acted_this_street", list(def_b)); finished = data.get("_player_finished_round", list(def_b)); last_acted = data.get("_last_player_acted", None);
        inst = cls.__new__(cls); inst.boards = boards; inst.deck = deck; inst.private_discard = p_discard; inst.dealer_idx = d_idx; inst._internal_current_player_idx = int_cp_idx; inst.street = st; inst.current_hands = c_hands; inst.fantasyland_hands = fl_hands; inst.fantasyland_status = fl_stat; inst.next_fantasyland_status = next_fl_stat; inst.fantasyland_cards_to_deal = fl_cards; inst.is_fantasyland_round = is_fl; inst._player_acted_this_street = acted; inst._player_finished_round = finished; inst._last_player_acted = last_acted; return inst
