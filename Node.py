import math
import numpy as np
import tensorflow as tf
import copy

stateSize = 9
maxMoves = 9
c_puct = 0.01


def validMovesTTT(state):
    return [(state[i] == 0) for i in range(9)]


def validMovesC4(state):
    return [(state[i+35] == 0) for i in range(7)]


def evaluateStateTTT(s):
    for i in range(3):
        if s[3*i] == s[3*i+1] == s[3*i+2] != 0:
            return (True, s[3*i])
        if s[i] == s[i+3] == s[i+6] != 0:
            return (True, s[i])
    if s[0] == s[4] == s[6] != 0:
        return (True, s[0])
    if s[2] == s[4] == s[6] != 0:
        return (True, s[2])
    for i in range(9):
        if s[i] == 0:
            return (False, 0)
    return (True, 0)


def evaluateStateC4(s):
    # horizontal
    for i in range(6):
        for j in range(4):
            if s[7 * i + j] == s[7 * i + j + 1] == s[7 * i + j + 2] == s[7 * i + j + 3] and s[7 * i + j] != 0:
                return (True, s[7 * i + j])
    # vertical
    for i in range(3):
        for j in range(7):
            if s[7 * i + j] == s[7 * i + j + 7] == s[7 * i + j + 14] == s[7 * i + j + 21] and s[7 * i + j] != 0:
                return (True, s[7 * i + j])
    # diagonal up-right
    for i in range(3):
        for j in range(4):
            if s[7 * i + j] == s[7 * i + j + 8] == s[7 * i + j + 16] == s[7 * i + j + 24] and s[7 * i + j] != 0:
                return (True, s[7 * i + j])
    # diagonal up-left
    for i in range(3):
        for j in range(3, 8):
            if s[7 * i + j] == s[7 * i + j + 6] == s[7 * i + j + 12] == s[7 * i + j + 18] and s[7 * i + j] != 0:
                return (True, s[7 * i + j])
    # there are still moves available
    for i in range(7):
        if s[i + 35] == 0:
            return (False, 0)
    # tie
    return (True, 0)


def nextStateTTT(state, move):
    s = state.copy()
    if s[move] != 0:
        print("Error in Node.py nextStateTTT: Invalid move")
    s[move] = 1
    return s


def nextStateC4(state, move):
    s = state.copy()
    if s[move+35] != 0:
        print("Error in Node.py nextStateC4: Invalid move")
    for i in range(move, 42, 7):
        if s[i] == 0:
            s[i] = 1
            return s


class Node:
    startState = []

    @staticmethod
    def initC4():
        Node.startState = [0]*stateSize

    def __init__(self, state, parent):  # done
        if len(state) != stateSize:  # use tf.shape?
            print("Error in Node.py __init__: node initialized with invalid state")
            print(state)
            return
        self.state = state
        self.parent = parent
        self.valid = validMovesTTT(state)
        self.end, self.endVal = evaluateStateTTT(state)
        self.leaf = True
        self.children = [None]*maxMoves
        self.N = [0]*maxMoves
        self.W = [0]*maxMoves
        self.Q = [0]*maxMoves
        self.P = [0]*maxMoves

    def getState(self):
        return self.state.copy()

    def chooseBest(self):  # done
        if self.leaf:
            print("Error in Node.py chooseBest: Choosing from leaf node")
            return None
        totalVisits = sum(self.N)
        bestVal = [self.Q[i] + c_puct * self.P[i] * math.sqrt(totalVisits + 1) / (self.N[i] + 1)
                   if self.valid[i] else -2 for i in range(maxMoves)]
        # bestMove = [i for i in range(maxMoves) if self.valid[i] and self.Q[i] +
                    # c_puct * self.P[i] * math.sqrt(totalVisits + 1) / (self.N[i] + 1) == max(bestVal)]
        if any(math.isnan(i) for i in bestVal):
            print("Error in Node.py chooseBest: no best move found")
            for i in range(maxMoves):
                if math.isnan(self.P[i]):
                    print("Probabilities are nan!")
                    return None
            return None
        return self.children[np.argmax(bestVal)]

    def expand(self, prob):  # done
        # print("Expanding")
        if not self.leaf:
            print("Error in Node.py expand: tried to expand non-leaf node")
            return
        if len(prob) != maxMoves+1:
            print("Error in Node.py expand: probability vector size does not match -- size = " + str(tf.size(prob)))
            return
        self.leaf = False
        for i in range(maxMoves):
            if self.valid[i]:
                new = [-j for j in nextStateTTT(self.state, i)]
                self.children[i] = Node(new, self)
                self.P[i] = prob[i]

    def update(self, v, child):  # done
        # print("Update")
        for i in range(maxMoves):
            if self.children[i] == child:
                self.N[i] += 1
                self.W[i] += v
                self.Q[i] = self.W[i] / self.N[i]
        # print(sum(self.N))

    def getProbDistribution(self):
        s = sum(self.N)
        prob = [i/s for i in self.N]
        if prob == None:
            print("Error in getProbDistribution")
        return prob

    def chooseMove(self):
        return np.random.choice(maxMoves, 1, p=self.getProbDistribution())[0]

    def chooseNewState(self):
        if self.leaf:
            print("Error in Node.py chooseNewState: choosing from leaf node")
        return self.children[self.chooseMove()]
