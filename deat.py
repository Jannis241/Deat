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
        self.raw_output = []
        self.inputs = None
        # creating a weight for each incoming neuron
        for neuron in range(incomingNeurons):
            self.weights.append(random.uniform(WEIGHT_RANGE[0], WEIGHT_RANGE[1]))

    def getOutput(self, inputs):
        self.inputs = np.array(inputs)
        self.raw_output = np.dot(self.weights, self.inputs) + self.bias
        return sigmoid(self.raw_output)


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

    def backpropagate(self, output_errors, learn_rate):
        input_errors = np.zeros(self.neurons[0].incomingNeurons)

        for i, neuron in enumerate(self.neurons):
            d_raw_output = deriv_sigmoid(neuron.raw_output)  # Derivative of sigmoid activation
            error_signal = output_errors[i] * d_raw_output  # Error signal for current neuron

            for j in range(len(neuron.weights)):
                input_errors[j] += neuron.weights[j] * error_signal

            # Update weights and bias for current neuron
            for j in range(len(neuron.weights)):
                neuron.weights[j] -= learn_rate * error_signal * neuron.inputs[j]

            neuron.bias -= learn_rate * error_signal

        return input_errors


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
        outputs = np.array(outputs)
        expectedOutputs = np.array(expectedOutputs)
        return ((expectedOutputs - outputs) ** 2).mean()

    def train(self, inputs, expectedOutputs, learn_rate=0.05, trainings=500):
        self.overAllLoss = 0
        expectedOutputs = np.array(expectedOutputs)
        inputs = np.array(inputs)
        for training in range(trainings):

            prediction = self.calcOutput(inputs)

            loss = self.evaluateLoss(prediction, expectedOutputs)

            # Backpropagation
            error = -(expectedOutputs - prediction)  # wie viel so zu sagen noch zur richtigen Lösung fehlt

            """
            zb ist der expected output = [1,1]
            wenn die predition dann [0,0] ist der error = [1,1] - [0,0] = [1,1]
            wenn die predition dann [0,1] ist der error = [1,1] - [0,1] = [1,0]
            das heißt es wird geguckt wie falsch dieser eine wert ist und dann die weights und biases angepasst werden
            """

            # telling each layer (starting from behind) how bad their output was and how much to change them
            for layer in reversed(self.hiddenLayerList + [self.outPutLayer]):
                error = layer.backpropagate(error, learn_rate)
            self.overAllLoss += loss

        self.averageLoss = self.overAllLoss / trainings


def calculateAccuracy(output, expectedOutput):
    accuracy = round(100 - abs(expectedOutput - output) * 100, 3)
