import math
import random

import numpy as np

WEIGHT_RANGE = [-2, 2]
BIAS_RANGE = [-2, 2]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


class Neuron:
    def __init__(self, incomingNeurons) -> None:
        self.incomingNeurons = incomingNeurons
        self.bias = random.uniform(BIAS_RANGE[0], BIAS_RANGE[1])
        self.weights = []

        # creating a weight for each incoming neuron
        for neuron in range(incomingNeurons):
            self.weights.append(random.uniform(WEIGHT_RANGE[0], WEIGHT_RANGE[1]))

    def getOutput(self, inputs):

        # Weight inputs, add bias, then use the activation function (sigmoid)
        result = np.dot(self.weights, inputs) + self.bias
        return sigmoid(result)


class Layer:
    def __init__(self, incomingNeurons, numOfNeurons):
        self.neurons = []
        for i in range(numOfNeurons):
            newNeuron = Neuron(incomingNeurons)
            self.neurons.append(newNeuron)

    def calcLayerOutputs(self, inputs):
        result = []
        for neuron in self.neurons:
            result.append(neuron.getOutput(inputs))

        return result


class NeuralNetwork:
    def __init__(self, inputNodes, hiddenLayers: list, outputNodes) -> None:
        self.numOfInputs = inputNodes
        self.numOfOutputs = outputNodes
        self.numOfHiddenLayers = hiddenLayers
        self.hiddenLayerList = []

        incomingNeurons = self.numOfInputs

        for anzahlAnNeurons in hiddenLayers:
            l = Layer(incomingNeurons, anzahlAnNeurons)  # einmal wie viele inputs das neuron bekommt, und dann wie viele neuronen das layer bekommen soll
            incomingNeurons = anzahlAnNeurons  # für das nächste layer sind die incoming neurons so viele wie jetzt erstellt werden
            self.hiddenLayerList.append(l)

        self.outPutLayer = Layer(incomingNeurons, self.numOfOutputs)

    def calcOutput(self, inputs):
        for layer in self.hiddenLayerList:
            inputs = layer.calcLayerOutputs(inputs)  # die inputs werden auf den Ouput von dem vorherigen hidden Layer gesetzt
        # wenn alle hidden layer die inptus einmal beeinflusst haben wird der finale input zu dem output layer gegeben welche dann die finale ausgabe berechnet
        output = self.outPutLayer.calcLayerOutputs(inputs)
        return output

    def evaluateLoss(self, outputs, expectedOutputs):
        return ((expectedOutputs - outputs) ** 2).mean()

    def mutate(self, inputs, expectedOutputs=None, learn_rate=0.1):

        print(inputs, expectedOutputs)

        prediction = self.calcOutput(inputs)

        output = self.calcOutput(inputs)

        loss = self.evaluateLoss(output, expectedOutputs)


inputs = np.array([0, 1])

expectedOutputs = np.array([0, 1])

nn = NeuralNetwork(2, [2, 2], 2)
nn.mutate(inputs, expectedOutputs)
