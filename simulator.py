import numpy


class SimulationParameters(object):
    def __init__(self, discount=.9, max_iterations=1000, tolerance=0.01):
        self.discount = discount
        self.max_iterations = max_iterations
        self.tolerance = tolerance


class Simulator(object):
    def __init__(self, state_space, action_space, parameters=SimulationParameters()):
        self.state_space = state_space
        self.action_space = action_space
        self.parameters = parameters

    @property
    def discount(self):
        return self.parameters.discount

    @property
    def max_iterations(self):
        return self.parameters.max_iterations

    @property
    def tolerance(self):
        return self.parameters.tolerance

    def find_best_policy_for_state(self, state, value_func):
        best_action = None
        best_payoff = float('-inf')
        for a, payoff in self.action_space.get_actions(state):
            acc_payoff = payoff
            for next_state, prob in self.action_space.transition_probabilities(a, state):
                acc_payoff += value_func[next_state] * prob * self.discount
            if acc_payoff > best_payoff:
                best_action = a
                best_payoff = acc_payoff
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
        for i in range(self.max_iterations):
            next_value_func, next_policy_func = self.iterate(value_func)
            if numpy.amax(abs(value_func - next_value_func)) < self.tolerance:
                break
            value_func = next_value_func
        return next_value_func, next_policy_func
