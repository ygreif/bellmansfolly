import abc


class StateSpace(object):
    @abc.abstractmethod
    def dim(self):
        return NotImplementedError()

    @abc.abstractmethod
    def states(self):
        return NotImplementedError()
