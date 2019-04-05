import tensorflow as tf
import math
import numpy as np
from tensorflow import keras
import random
import copy

from Node import Node

stateSize = 9
maxMoves = 9
batchsize = 32


def printBoardTTT(state):
    print("Board")
    for i in range(3):
        for j in range(3):
            print(state[3*i+j], end=' ')
        print()


def printProbabilities(prob):
    print("print probabilities")

# TODO: finish

def rotateTTT(data):
    newData = data
    for i in range(3):
        for j in range(3):
            for k in range(2):
                newData[k][3*i+j] = data[k][3*(2-j)+i]
    return newData

def AddSymmetriesTTT(data, trainingData):
    trainingData.append(copy.deepcopy(data))
    data = rotateTTT(data)
    trainingData.append(copy.deepcopy(data))
    data = rotateTTT(data)
    trainingData.append(copy.deepcopy(data))
    data = rotateTTT(data)
    trainingData.append(copy.deepcopy(data))
    for i in range(3):
        for j in range(2):
            data[j][i], data[j][i+6] = data[j][i+6], data[j][i]
    trainingData.append(copy.deepcopy(data))
    data = rotateTTT(data)
    trainingData.append(copy.deepcopy(data))
    data = rotateTTT(data)
    trainingData.append(copy.deepcopy(data))
    data = rotateTTT(data)
    trainingData.append(copy.deepcopy(data))



class Net:
    def __init__(self, input_size):
        self.model = keras.Sequential([
            keras.layers.Dense(64, activation='sigmoid',
                               input_shape=(input_size, )),
            keras.layers.Dense(64, activation='sigmoid'),
            keras.layers.Dense(maxMoves+1, activation='softmax')
        ])
        self.model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def simulate(self, start):
        # print("Simulating")
        cur = start
        while not cur.leaf:
            cur = cur.chooseBest()
            if cur == None:
                print("Error in simulate: cur is None")
                return
        # print("leaf reached")
        # print(cur.getState())
        if not cur.end:
            s = np.array(cur.getState()).reshape(1, stateSize)
            s = self.model.predict(s)[0].tolist()
            if abs(sum(s)-1) > 0.0001:
                print("Error in simulate: Invalid probability distribution")
                print(s)
                return
            for i in range(maxMoves+1):
                if math.isnan(s[i]):
                    print("Error in simulate: NN output has nan")
                    print(s)
                    return
            v = s[maxMoves]
            cur.expand(s)
        else:
            v = cur.endVal
        while cur != start:
            v = -v
            cur.parent.update(v, cur)
            cur = cur.parent

    def selfPlay(self, trainingData, sims):
        start = [0]*9  # TTT specific
        cur = Node(start, None)
        while not cur.end:
            for _ in range(sims):
                self.simulate(cur)
            cur = cur.chooseNewState()
        winner = cur.endVal
        cur = cur.parent
        while cur != None:
            prob = cur.getProbDistribution()
            winner = -winner
            prob.append(winner)
            data = [cur.getState(), prob]
            AddSymmetriesTTT(data, trainingData)
            cur = cur.parent

    def learn(self, data):
        inputs = []
        answers = []
        random.shuffle(data)
        n = len(data)
        for i in range(n):
            inputs.append(data[i][0].copy())
            answers.append(data[i][1].copy())
        inputs = np.array(inputs).reshape(n, stateSize)
        answers = np.array(answers).reshape(n, stateSize+1)
        steps = math.ceil(n/batchsize)
        self.model.fit(inputs, answers, epochs=1,
                       steps_per_epoch=steps)

    def train(self, epochs, sims, games):
        print("Start training")
        for i in range(epochs):
            data = []
            for _ in range(games):
                self.selfPlay(data, sims)
                print("Finished game")
                print("data size = " + str(len(data)))
            print("Generated data")
            self.learn(data)
            print("Epoch %d completed" % (i))
