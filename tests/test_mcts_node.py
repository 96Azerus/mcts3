# tests/test_mcts_node.py
"""
Unit-тесты для модуля src.mcts_node.
"""
import pytest
import time
from unittest.mock import patch, MagicMock

# Импорты из src пакета
from src.mcts_node import MCTSNode
from src.game_state import GameState
from src.card import card_from_str, hand_to_int, INVALID_CARD
from src.board import PlayerBoard
from src.scoring import calculate_headsup_score

# Хелперы
def hand(card_strs):
    return [card_from_str(s) if s else None for s in card_strs]

def create_simple_state(street=1, dealer=0, p0_hand_str=None, p1_hand_str=None):
    state = GameState(dealer_idx=dealer)
    state.street = street
    if p0_hand_str:
        state.current_hands[0] = hand_to_int(p0_hand_str)
    if p1_hand_str:
        state.current_hands[1] = hand_to_int(p1_hand_str)
    state._internal_current_player_idx = (dealer + 1) % 2
    state._player_finished_round = [False, False]
    return state

# --- Тесты инициализации ---
def test_mcts_node_init():
    state = create_simple_state()
    node = MCTSNode(state)
    assert node.game_state == state
    assert node.parent is None
    assert node.action is None
    assert node.children == {}
    assert node.untried_actions is None
    assert node.visits == 0
    assert node.total_reward == 0.0
    assert node.rave_visits == {}
    assert node.rave_total_reward == {}

# --- Тесты expand ---
def test_mcts_node_expand():
    state = create_simple_state(street=1, dealer=0, p1_hand_str=['Ac','Kc','Qc','Jc','Tc'])
    root = MCTSNode(state)
    assert root._get_player_to_move() == 1
    initial_action_count = len(state.get_legal_actions_for_player(1))
    assert initial_action_count > 1

    child = root.expand()
    assert child is not None
    assert child.parent == root
    assert child.action is not None
    # Проверяем, что действие хешируемое и добавлено в children
    try:
        hash(child.action)
        assert child.action in root.children
    except TypeError:
        pytest.fail("Expanded action is not hashable")
    assert len(root.untried_actions) == initial_action_count - 1
    assert child.visits == 0

    action1 = child.action
    child2 = root.expand()
    assert child2 is not None
    assert child2.action != action1
    assert len(root.untried_actions) == initial_action_count - 2

# --- Тесты rollout ---
@patch('src.mcts_node.MCTSNode._static_heuristic_rollout_policy', return_value=None)
@patch('src.mcts_node.MCTSNode._static_heuristic_fantasyland_placement', return_value=(None, None))
def test_mcts_node_rollout_all_fouls(mock_fl_policy, mock_rollout_policy):
    """Тест роллаута, где оба игрока фолят (политика возвращает None)."""
    state = create_simple_state(street=1, dealer=0,
                                p0_hand_str=['2c','3c','4c','5c','6c'],
                                p1_hand_str=['Ad','Kd','Qd','Jd','Td'])
    node = MCTSNode(state)
    reward_p0, actions_p0 = node.rollout(perspective_player=0)
    reward_p1, actions_p1 = node.rollout(perspective_player=1)

    assert reward_p0 == 0.0
    assert reward_p1 == 0.0
    assert mock_rollout_policy.called

def test_mcts_node_rollout_simple_win():
    """Тест роллаута для уже терминального состояния."""
    state = GameState()
    board0 = PlayerBoard(); board0.set_full_board(hand(['Ah','Ad','Ac']), hand(['7h','8h','9h','Th','Jh']), hand(['As','Ks','Qs','Js','Ts']))
    board1 = PlayerBoard(); board1.set_full_board(hand(['Kh','Qd','2c']), hand(['Ac','Kd','Qh','Js','9d']), hand(['Tc','Td','Th','2s','3s']))
    state.boards = [board0, board1]
    state._player_finished_round = [True, True]
    state.street = 6

    node = MCTSNode(state)
    assert node.is_terminal()
    reward_p0, _ = node.rollout(perspective_player=0)
    reward_p1, _ = node.rollout(perspective_player=1)

    expected_score = calculate_headsup_score(board0, board1)
    assert reward_p0 == float(expected_score)
    assert reward_p1 == float(-expected_score)

# --- Тесты UCT ---
def test_uct_select_child_no_children():
    state = create_simple_state()
    node = MCTSNode(state)
    assert node.uct_select_child(1.4, 500) is None

def test_uct_select_child_unvisited():
    state = create_simple_state(street=1, dealer=0, p1_hand_str=['Ac','Kc','Qc','Jc','Tc'])
    root = MCTSNode(state)
    child1 = root.expand()
    assert child1 is not None
    root.visits = 1

    # Если есть неисследованные действия, uct_select_child вернет None,
    # т.к. выбор происходит из children, а неисследованные еще не там.
    # Сначала нужно исчерпать untried_actions через expand.
    # Поэтому тестируем случай, когда все действия испробованы, но дети не посещены.
    root.untried_actions = [] # Имитируем, что все действия были expand'нуты
    root.children = {child1.action: child1} # Добавляем одного ребенка
    selected_single = root.uct_select_child(1.4, 500)
    # Ожидаем, что будет выбран единственный неисследованный ребенок (score=inf)
    assert selected_single == child1

def test_uct_select_child_visited():
    state = create_simple_state(street=1, dealer=0, p1_hand_str=['Ac','Kc','Qc','Jc','Tc', '9c']) # 6 карт для 2 действий
    root = MCTSNode(state)
    # Создаем двух детей
    action1 = state.get_legal_actions_for_player(1)[0]
    next_state1 = state.apply_action(1, action1)
    child1 = MCTSNode(next_state1, parent=root, action=action1)
    root.children[action1] = child1

    action2 = state.get_legal_actions_for_player(1)[1]
    next_state2 = state.apply_action(1, action2)
    child2 = MCTSNode(next_state2, parent=root, action=action2)
    root.children[action2] = child2

    root.untried_actions = [] # Все действия испробованы
    root.visits = 10
    child1.visits = 5
    child1.total_reward = 3.0 # Q(P0) = 0.6
    child2.visits = 3
    child2.total_reward = -1.0 # Q(P0) = -0.33

    # Мокируем RAVE, чтобы он не влиял на выбор
    root.rave_visits = {}
    root.rave_total_reward = {}

    selected = root.uct_select_child(1.4, 0) # rave_k = 0 отключает RAVE
    # Ожидаем, что будет выбран child1 (выше Q + exploration)
    assert selected == child1

# --- Тесты backpropagate ---
def test_backpropagate_updates_stats():
    state = create_simple_state()
    root = MCTSNode(state)
    # Имитируем путь
    action1 = 'action1_hashable'
    action2 = 'action2_hashable'
    child1 = MCTSNode(state, parent=root, action=action1)
    child2 = MCTSNode(state, parent=child1, action=action2)
    path = [root, child1, child2]
    simulation_actions = {action1, action2, 'other_action_hashable'}

    # Имитируем бэкпропагацию
    root._backpropagate_parallel(path, total_reward=5.0, num_rollouts=2, simulation_actions=simulation_actions)

    assert root.visits == 2
    assert root.total_reward == 5.0
    assert child1.visits == 2
    assert child1.total_reward == 5.0
    assert child2.visits == 2
    assert child2.total_reward == 5.0

    # Проверяем RAVE статистику (только для хешируемых действий)
    assert root.rave_visits.get(action1) == 2
    assert root.rave_total_reward.get(action1) == 5.0
    assert child1.rave_visits.get(action2) == 2
    assert child1.rave_total_reward.get(action2) == 5.0
    assert 'other_action_hashable' not in root.rave_visits # Не было в пути как действие узла
