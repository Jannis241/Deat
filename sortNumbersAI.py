import random

import deat

nn = deat.NeuralNetwork(10, [5], 10)

for generation in range(500):
    input = [random.random() for i in range(10)]
    expectedOutput = sorted(input)

    nn.train(input, expectedOutput)
    if generation % 10 == 0:
        print(f"generation: {generation}, loss: {round(nn.averageLoss, 3)}")


input = [0.2, 0.1, 0.4, 0.8, 0.55, 0.44, 0.875, 0.6, 0.22, 0.99]
expectedOutput = sorted(input)
output = nn.calcOutput(input)
loss = nn.evaluateLoss(output, expectedOutput)

for i in range(len(output)):
    output[i] = float(round(output[i], 3))


print("-------------------")
for i in range(len(input)):
    accuracy = deat.calculateAccuracy(input[i], expectedOutput[i])
    print(f"input: {input[i]}, expected output: {expectedOutput[i]}, output: {output[i]}, Accuracy: {accuracy}%")
print()
print("loss: ", round(loss, 3))
print()
print(f"overall Iterations: {500 * generation}")
