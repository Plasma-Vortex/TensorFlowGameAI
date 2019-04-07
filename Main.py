local = True

if local:
    from NeuralNet import Net
    import numpy as np
    from Game import *

np.random.seed()

n = Net("128-128-128-128", 2400)
# n = Net("1024-1024-1024-1024", 920)

n.train(20, 40)
# n.playHuman(40, 0)
