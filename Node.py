import math
import numpy as np

stateSize = 42
maxMoves = 7
c_puct = 0.01

def validMovesC4(state):
    return [(state[i+35] == 0) for i in range(7)]

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

def nextStateC4(state, move):
    copy = state.copy()
    if copy[move+35] != 0:
        print("Error in Node.py nextStateC4: Invalid move")
    for i in range(move, 42, 7):
        if copy[i] == 0:
            copy[i] = 1
            return copy


class Node:
    startState = []

    @staticmethod
    def initC4():
        Node.startState = [0]*stateSize

    def __init__(self, state, parent): # done
        if tf.size(state) != stateSize + 1: # use tf.shape?
            print("Error in Node.py __init__: node initialized with invalid state -- state size is " + str(tf.size(state)))
        self.state = state
        self.parent = parent
        self.valid = validMovesC4(state)
        self.end, self.endVal = evaluateStateC4(state)
        self.leaf = True
        self.children = [None]*maxMoves
        self.N = [0]*maxMoves
        self.W = [0]*maxMoves
        self.Q = [0]*maxMoves
        self.P = [0]*maxMoves

    def chooseBest(self): # done
        if self.leaf:
            print("Error in Node.py chooseBest: Choosing from leaf node")
            return None
        totalVisits = sum(self.N)
        bestVal = [self.Q[i] + c_puct * self.P[i] * math.sqrt(totalVisits + 1) / (self.N[i] + 1) for i in range(maxMoves) if valid[i]]
        bestMove = [i for i in range(maxMoves) if self.valid[i] and self.Q[i] + c_puct * self.P[i] * math.sqrt(totalVisits + 1) / (self.N[i] + 1) == bestVal]
        if math.isnan(bestVal):
            print("Error in Node.py chooseBest: no best move found")
            for i in range(maxMoves):
                if math.isnan(self.P[i]):
                    print("Probabilities are nan!")
                    return None
            return None
        return self.children[bestMove]
    
    def expand(self, prob): #done
        if not self.leaf:
            print("Error in Node.py expand: tried to expand non-leaf node")
            return
        if prob.size() != maxMoves+1:
            print("Error in Node.py expand: probability vector size does not match -- size = " + str(tf.size(prob)))
            return
        self.leaf = False
        for i in range(maxMoves):
            if self.valid[i]:
                copy = nextStateC4(self.state, i)
                copy = -copy
                self.children[i] = Node(copy, self)
                self.P[i] = prob[i]

    def update(self, v, child): # done
        for i in range(maxMoves):
            if self.children[i] == child:
                self.N[i] += 1
                self.W[i] += v
                self.Q[i] = self.W[i] / self.N[i]
    
    def chooseMove(self):
        s = sum(self.N)
        prob = [i/s for i in self.N] # probability distribution
        prob.append(0)
        return np.random.choice(maxMoves, 1, p = prob)[0]

    def chooseNewState(self):
        if self.leaf:
            print("Error in Node.py chooseNewState: choosing from leaf node")
        return self.children[self.chooseMove()]
