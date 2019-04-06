local = True

if local:
    from Node import Node
    from NeuralNet import Net

stateSize = 9

n = Net(stateSize, "largenet", 0)

n.train(100000, 50, 10)
