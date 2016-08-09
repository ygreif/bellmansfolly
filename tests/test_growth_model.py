import math
import numpy

from numba import jit

import simulator
import growth.growth_model as g

params = simulator.SimulationParameters(discount=g.beta)
state_space = g.GrowthEconomyStateSpace(g.tfp_by_state, g.k_precision)
action_space = g.GrowthEconomyActionSpace(state_space)




def test_state_space_indexing():
    # index 0 maps to min capital
    assert state_space.capital_to_index(g.min_capital) == 0
    assert state_space.capital_to_index(g.min_capital + g.k_precision) == 1

    assert state_space.state_to_capital((0, 0)) == g.min_capital
    assert state_space.state_to_capital((0, 1)) == g.min_capital + g.k_precision

    state_max = (0, state_space.dims[1])  # state with max capital
    state_next_to_max = (0, state_space.dims[1] - 1)
    assert state_space.state_to_capital(state_next_to_max) <= g.max_capital <= state_space.state_to_capital(state_max)
    
    assert state_space.dims[0] == len(g.tfp_by_state)
    assert state_space.dims[1] == math.ceil((g.max_capital - g.min_capital) / g.k_precision)


def test_tfp_transition_probabilities():
    for row in g.state_transitions:
        assert sum(row) == 1.0


def test_action_space():
    r = list(action_space.get_actions((0, g.min_capital)))
    n_decimals = round(math.log(1 / g.k_precision, 10))

    # produce minumum capital
    tfp = g.tfp_by_state[0]
    p = g.production(g.min_capital, tfp)
    print("prod",p,"k",g.min_capital,"tfp",tfp)
    numpy.testing.assert_almost_equal(r[0], (g.min_capital, g.utility(p - g.min_capital)), n_decimals)
    # don't consume anything
    numpy.testing.assert_almost_equal(r[-1], (g.max_capital, g.utility(p - g.max_capital)), n_decimals)

    # can go to any state
    assert len(r) == state_space.dim()[1]


def test_concavity():
    for state in state_space.states():
        prev_payoff = float('inf')
        for action, payoff in action_space.get_actions(state):
            print(action, payoff)
            assert payoff < prev_payoff
            prev_payoff = payoff
    assert False

def test_transition_probabilities():
    all_states = frozenset(state_space.states())
    for state in state_space.states():
        for action, _ in action_space.get_actions(state):
            cum_prob = 0.0
            for next_state, prob in action_space.transition_probabilities(action, state):
                assert next_state in state_space.states()
                cum_prob += prob
            assert cum_prob == 1.0
