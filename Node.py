local = True

if local:
    import math
    import numpy as np
    import tensorflow as tf
    import copy
    from Game import *

stateSize = 42
maxMoves = 7
c_puct = 1


class Node:
    def __init__(self, state, parent=None):  # done
        if len(state) != stateSize:  # use tf.shape?
            print("Error in Node.py __init__: node initialized with invalid state")
            print(state)
            return
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

    def getState(self):
        return self.state.copy()

    def chooseBest(self):  # done
        if self.leaf:
            print("Error in Node.py chooseBest: Choosing from leaf node")
            return
        totalVisits = sum(self.N)
        values = [self.Q[i] + c_puct * self.P[i] * math.sqrt(totalVisits + 1) / (self.N[i] + 1)
                  if self.valid[i] else -2 for i in range(maxMoves)]
        bestMoves = [i for i in range(maxMoves) if values[i] == max(values)] # tied for best
        if any(math.isnan(i) for i in values):
            print("Error in Node.py chooseBest: values has nan")
            for i in range(maxMoves):
                if math.isnan(self.P[i]):
                    print("Probabilities are nan!")
                    return
            return
        chosenMove = bestMoves[np.random.randint(len(bestMoves))]
        return self.children[chosenMove]

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
                new = [-j for j in nextStateC4(self.state, i)]
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
