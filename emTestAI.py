import random

import deat

# input landID1, landID2
# output: tore für land 1, tore für land 2


spanien = 1
deutschland = 2
frankreich = 3
england = 4


def normalize_list(numbers: list):
    max_value = max(numbers)
    normalized_numbers = [x / max_value for x in numbers]
    return normalized_numbers


nn = deat.NeuralNetwork(2, [5], 1)

for generation in range(500):
    expectedOutput = input[0] + input[1]

    nn.train(input, expectedOutput)
    if generation % 10 == 0:
        print(f"generation: {generation}, loss: {round(nn.averageLoss, 3)}")


input = [5, 5]
expectedOutput = [10]
output = nn.calcOutput(input)
loss = nn.evaluateLoss(output, expectedOutput)

for i in range(len(output)):
    output[i] = float(round(output[i], 3))


print("-------------------")
print(expectedOutput)
print("Prediction: ", output)
print()
print("loss: ", round(loss, 3))
print()
print(f"overall Iterations: {500 * generation}")
