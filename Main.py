local = True

if local:
    from NeuralNet import Net
    import numpy as np
    from Game import *

np.random.seed()

n = Net("128-128-128-128", 0)
# n = Net("1024-1024-1024-1024", 0)

n.train(20, 50)
# n.playHuman(500, 0)
