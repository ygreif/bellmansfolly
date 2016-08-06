import abc


class ActionSpace(object):
    @abc.abstractmethod
    def transition_probabilities(self, action, state):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_actions(self, state):
        raise NotImplementedError()

    def is_concave(self):
        return False
 
