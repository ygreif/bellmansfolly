import numba
import itertools
import math
import numpy

from model import state_space, action_space

alpha = 1.0 / 3.0  # return on capital
beta = 0.95  # discount factor
delta = 1  # deprecation of capital


def utility(c):
    return (1 - beta) * math.log(c)


def production(k, tfp):
    return tfp * math.pow(k, alpha) + (1 - delta) * k

tfp_by_state = numpy.array([0.9792, 0.9896, 1.0000, 1.0106, 1.0212], float)
state_transitions = numpy.array([[0.9727, 0.0273, 0.0000, 0.0000, 0.0000],
                                  [0.0041, 0.9806, 0.0153, 0.0000, 0.0000],
                                  [0.0000, 0.00815, 0.9837, 0.00815, 0.0000],
                                  [0.0000, 0.0000, 0.0153, 0.9806, 0.0041],
                                  [0.0000, 0.0000, 0.0000, 0.0273, 0.9727]], float)

capital_steady_state = (alpha*beta)**(1/(1-alpha))
k_precision = 0.0001
min_capital = int(capital_steady_state / 2.0 / k_precision) * k_precision
max_capital = int(capital_steady_state * 1.5 / k_precision) * k_precision


class GrowthEconomyStateSpace(state_space.StateSpace):
    def __init__(self, tfp_states, k_precision):
        self.k_precision = k_precision
        self.dims = [int(len(tfp_states)), math.ceil((max_capital - min_capital) / k_precision)]
        self.statess = [(i, j) for i, j in itertools.product(range(self.dims[0]), range(self.dims[1]))]

    def dim(self):
        return self.dims

    def states(self):
        return self.statess

    def state_to_capital(self, state):
        return state[1] * self.k_precision + min_capital

    def capital_to_index(self, capital):
        return int((capital - min_capital) / k_precision)


class GrowthEconomyActionSpace(action_space.ActionSpace):
    def __init__(self, growth_state_space):
        self.state_space = growth_state_space

        # for memoizeing
        self._actions_by_tfp_level = {}
        self._transitions = {}

    def get_actions(self, state):
        if state in self._actions_by_tfp_level:
            return self._actions_by_tfp_level[state]
        tfp = tfp_by_state[state[0]]
        k = self.state_space.state_to_capital(state)
        output = production(k, tfp)
        result = []
        for next_k in numpy.arange(min_capital, min(max_capital, output), k_precision):
            result.append((next_k, utility(production(k, tfp) - next_k)))
        self._actions_by_tfp_level[state] = result
        return result

    def transition_probabilities(self, action, state):
        capital_index = self.state_space.capital_to_index(action)
        if (capital_index, state[0]) in self._transitions:
            return self._transitions[(capital_index, state[0])]
        transitions = []
        for next_tfp_state in range(len(state_transitions)):
            next_state = (next_tfp_state, capital_index)
            transitions.append((next_state, state_transitions[(state[0], next_tfp_state)]))
        self._transitions[(capital_index, state[0])] = transitions
        return transitions
