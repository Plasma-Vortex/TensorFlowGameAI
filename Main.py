import tensorflow as tf
import math
from Node import Node
from NeuralNet import Net
import copy

print(tf.__version__)

stateSize = 9

n = Net(stateSize)

n.train(1000, 50, 50)