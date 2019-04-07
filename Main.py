local = True

if local:
    from NeuralNet import Net
    import numpy as np

np.random.seed()

n = Net("128-128-128-128", 220)

n.train(20, 40)
# n.playHuman(500, 0)