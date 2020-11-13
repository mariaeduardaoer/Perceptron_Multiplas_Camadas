# -*- coding: utf-8 -*-
import numpy as np
from activation_functions import Logistic, LogisticDerivative


class MLP:
    ...

    def __init__(self, input_values, output_values, layers, activation_function=Logistic,
                 derivative_function=LogisticDerivative, learning_rate=0.1, precision=1e-6):
        ones_column = np.ones((len(input_values), 1)) * -1
        self.input_values = np.append(ones_column, input_values, axis=1)
        self.output_values = output_values
        self.learning_rate = learning_rate
        self.precision = precision
        self.activation_function = activation_function
        self.derivative_function = derivative_function

        self.W = []
        neuron_input = self.input_values.shape[1]
        for i in range(len(layers)):
            self.W.append(np.random.rand(layers[i], neuron_input))
            neuron_input = layers[i] + 1

        self.epochs = 0
        self.eqms = []

    def train(self):

        print('Inicializando o processo de treinamento...')
        error = True

        while error:
            print(f'EPOCH: {self.epochs}')
            error = False

            eqm_previous = self.eqm()

            for x, d in zip(self.input_values, self.output_values):

                # FORWARD
                I1 = np.dot(self.W[0], x)
                Y1 = np.zeros(I1.shape)
                for i in range(Y1.shape[0]):
                    Y1[i] = self.activation_function.g(I1[i])
                Y1 = np.append(-1, Y1)

                I2 = np.dot(self.W[1], Y1)
                Y2 = np.zeros(I2.shape)
                for i in range(Y2.shape[0]):
                    Y2[i] = self.activation_function.g(I2[i])

                # TREINAMENTO COM O BACKPROPAGATION

                # BACKWARD
                # ajustando pesos sinapticos da camada de SAIDA:    
                delta2 = (d - Y2) * self.derivative_function.dg(I2)
                delta2_reshape = delta2.reshape(delta2.shape[0], 1)
                self.W[1] = self.W[1] + (self.learning_rate * delta2_reshape * Y1)

                # ajustando pesos sinapticos da camada de INTERMEDIARIA:
                soma = sum(delta2[:, np.newaxis] * self.W[1])
                delta1 = soma[1:] * self.derivative_function.dg(I1)
                delta1_reshape = delta1.reshape(delta1.shape[0], 1)
                self.W[0] = self.W[0] + (self.learning_rate * delta1_reshape * x)

            eqm_actual = self.eqm()
            self.eqms.append(eqm_actual)
            self.epochs += 1
            if abs(eqm_actual - eqm_previous) > self.precision:
                error = True

        print(f'Final W: {self.W}')
        return self.eqms

    def eqm(self):

        eq = 0

        for x, d in zip(self.input_values, self.output_values):

            I1 = np.dot(self.W[0], x)
            Y1 = np.zeros(I1.shape)
            for i in range(Y1.shape[0]):
                Y1[i] = self.activation_function.g(I1[i])
            Y1 = np.append(-1, Y1)

            I2 = np.dot(self.W[1], Y1)
            Y2 = np.zeros(I2.shape)
            for i in range(Y2.shape[0]):
                Y2[i] = self.activation_function.g(I2[i])

            eq += 0.5 * sum((d - Y2) ** 2)

        return eq / len(self.output_values)

    def evaluate(self, x):

        x = np.append(-1, x)

        I1 = np.dot(self.W[0], x)
        Y1 = np.zeros(I1.shape)
        for i in range(Y1.shape[0]):
            Y1[i] = self.activation_function.g(I1[i])
        Y1 = np.append(-1, Y1)

        I2 = np.dot(self.W[1], Y1)
        Y2 = np.zeros(I2.shape)
        for i in range(Y2.shape[0]):
            Y2[i] = self.activation_function.g(I2[i])

        # arredondamento simetrico
        # 1, se yi >= 0.5
        # 0, se yi < 0.5
        return [1 if 100 * (Y2[0] / sum(Y2)) > 50 else 0,
                1 if 100 * (Y2[1] / sum(Y2)) > 50 else 0,
                1 if 100 * (Y2[2] / sum(Y2)) > 50 else 0]
