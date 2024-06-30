import random

import deat

generations = 1000

nn = deat.NeuralNetwork(2, [2, 2], 2)


def startGeneration():
    input = random.choice([[0, 1], [1, 0]])
    expectedOutput = input
    nn.mutate(input, expectedOutput)


for i in range(generations):
    startGeneration()
