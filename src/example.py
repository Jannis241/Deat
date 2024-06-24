# programm was bestimmt welche Zahl größer ist
import random
import deat


NN = deat.NeuralNetork(2, [3,3], 2)


networks = 25
generations = 20
tries = 10

mutationRate = 0.3

nnList = [deat.NeuralNetork(2, [2,2], 2)]
for i in range(generations):
    for nn in nnList:
        for t in range(tries):
            zahl1 = random.randrange(1,10) / 10
            zahl2 = 1 - zahl1

            inputs = [zahl1, zahl2]


            NNoutput = nn.calcOutput(inputs)

            NNchoice = NNoutput.index(max(NNoutput))

            correctIndex = inputs.index(max(inputs))

            if NNchoice == correctIndex:
                nn.score += 1

            else:
                nn.score -= 1

            

            # gucken ob es richtig ist und score geben

    
    bestNN = deat.getBestNN(nnList)
    for nn in nnList:
        nn.mutate(mutationRate, bestNN)

    print(f"Gen {i}: {round(bestNN.score,2)} points")