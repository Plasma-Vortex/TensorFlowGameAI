local = True

if local:
    from NeuralNet import Net
    import numpy as np
    from Game import *

np.random.seed()

# n = Net("128-128-128-128", 2760)
n = Net("512-512-512-512-512", 0)
# n = Net("256-256", 310)

n.train(20, 50)
# n.playHuman(1, 0)
