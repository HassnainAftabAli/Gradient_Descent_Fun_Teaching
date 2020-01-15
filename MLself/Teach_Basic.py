import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

weight = 20
bias = 10
inputs = np.random.rand(5000)*100
expectedOuts = inputs*11
itters = 100
length = len(expectedOuts)
inputs = np.array(inputs)
costArr = []
endVals = np.zeros(len(expectedOuts))

def find_outs(values):
    return (values*weight) + bias

def cost(outputs):
    return (np.sum(np.square(outputs-expectedOuts))/(2*length))

def gradient(learningrate, outputs):
    return (np.dot(outputs - expectedOuts, inputs)*(learningrate/length))

for x in range(itters):
    endVals = find_outs(inputs)
    thisCost = cost(endVals)
    costArr.append(thisCost)
    print('Cost of this iteration is ' + str(thisCost))
    grad = gradient(0.0004, endVals)
    weight = weight - grad
    bias = bias - grad
    print('Weight is ' + str(weight))

df = pd.DataFrame(costArr)
df.plot()
plt.show()