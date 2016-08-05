import numpy


class Simulator(object):
    def __init__(self, state_space, action_space, discount):
        self.state_space = state_space
        self.action_space = action_space
        self.discount = discount
        self.max_iterations = 1000
        self.tolerance = 0.01

    def find_best_policy_for_state(self, state, value_func):
        best_action = None
        best_payoff = float('-inf')
        for a, payoff in self.action_space.get_actions(state):
            for next_state, prob in self.action_space.transition_probabilities(a, state):
                payoff += value_func[next_state] * prob * self.discount
            if payoff > best_payoff:
                best_action = a
                best_payoff = payoff
        return best_action, best_payoff

    def iterate(self, value_func):
        next_value_func = numpy.ndarray(self.state_space.dim())
        next_policy_func = numpy.ndarray(self.state_space.dim())

        for state in self.state_space.states():
            action, value = self.find_best_policy_for_state(state, value_func)
            next_value_func[state] = value
            next_policy_func[state] = action

        return next_value_func, next_policy_func

    def find_best_policy(self):
        value_func = numpy.ndarray(self.state_space.dim())
        for i in range(0, self.max_iterations):
            next_value_func, next_policy_func = self.iterate(value_func)
#            if numpy.amax(abs(value_func - next_value_func)) < self.tolerance:
 #               break
            value_func = next_value_func
        return next_value_func, next_policy_func
