import numpy

from model import state_space, action_space
import simulator


class SingleDimStateSpace(state_space.StateSpace):
    def __init__(self, len):
        self.d = [1, len]

    def dim(self):
        return self.d

    def states(self):
        return [(0, i) for i in xrange(self.dim()[1])]


class SingleDimActionSpace(action_space.ActionSpace):
    def __init__(self, choices, valuations, len):
        self.choices = choices
        self.valuations = valuations
        self.len = len

    def get_actions(self, state):
        for choice in self.choices:
            potential_action = choice + state[1]
            if 0 <= potential_action < self.len:
                yield (choice, -1 * choice * choice + self.valuations[state])

    def transition_probabilities(self, action, state):
        loc = state[1]
        for delta, prob in [(-1, .5), (0, .25), (1, .25)]:
            next_loc = min(max(loc + action + delta, 0), self.len - 1)
            if 0 <= next_loc < self.len:
                yield (0, next_loc), prob


# TODO: make fixture or abandon py.test
l = 4
discount = .5
valuations = numpy.asarray([[1, 1.5, 4, 6]])
s_s = SingleDimStateSpace(l)
a_s = SingleDimActionSpace([0, 1, 2], valuations, l)
value_func = numpy.asarray([[1.5, 2, 8, 12]])
sim = simulator.Simulator(s_s, a_s, discount)


def test_find_best_policy_for_state():
    best_action, best_value = sim.find_best_policy_for_state((0, 0), value_func)
    print best_action, best_value
    assert best_action == 0
    assert best_value == 1 + .5 * (.75 * 1.5 + .25 * 2)

def test_iterate():
    next_value_func, next_policy_func = sim.iterate(value_func)
    assert (next_policy_func == [0, 1, 1, 0]).all()
    assert (next_value_func == [1.8125, 3.5, 8, 11]).all()

def test_find_best_policy():
    final_payoff, best_policy = sim.find_best_policy()
    next_payoff, next_policy = sim.iterate(final_payoff)
    print best_policy
    print final_payoff
    # check policies are the same and payoffs are within epsilon
    for idx in xrange(s_s.dim()[1]):
        assert best_policy[0][idx] == next_policy[0][idx]
        assert abs(final_payoff[0][idx] - next_payoff[0][idx]) < sim.tolerance
