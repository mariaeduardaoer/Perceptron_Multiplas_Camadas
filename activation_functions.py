from abc import ABC, abstractmethod
import math


class ActivationFunction(ABC):

    @staticmethod
    @abstractmethod
    def g(u):
        pass


class DerivativeFunction(ABC):

    @staticmethod
    @abstractmethod
    def dg(u):  # derivada de g
        pass


class BinaryStep(ActivationFunction):

    def g(u):
        return 1 if u >= 0 else 0


class SignFunction(ActivationFunction):

    def g(u):
        return 1 if u >= 0 else -1


class TanH(ActivationFunction):

    def g(u):
        return (1 - math.e ** (-2 * u)) / ((1 + math.e ** (-2 * u)))


class TanHDerivative(DerivativeFunction):

    def dg(u):
        return 1 - TanH.g(u)


class Logistic(ActivationFunction):

    def g(u):
        return 1 / (1 + math.e ** (-u))


class LogisticDerivative(DerivativeFunction):

    def dg(u):
        return (Logistic.g(u)) * (1 - Logistic.g(u))
