import numpy as np


WEIGHTS = np.array([0.1, 0.2, 0])


def neural_network(input_data, weight):
    prediction = input_data.dot(weight)
    return prediction


toes = np.array([8.5, 9.5, 9.9, 9.0])
wlrec = np.array([0.65, 0.8, 0.8, 0.9])
nfans = np.array([1.2, 1.3, 0.5, 1.0])


input_data = np.array([toes[0], wlrec[0], nfans[0]])
pred = neural_network(input_data, WEIGHTS)
print(pred)
