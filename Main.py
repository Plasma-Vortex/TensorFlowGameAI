local = True

if local:
    from NeuralNet import Net
    import numpy as np
    from Game import *
    from Arena import *
    import matplotlib.pyplot as plt

np.random.seed()

# n1 = Net("128-128-128-128", 2500)
# n2 = Net("128-128-128-128", 1200)
# # n2 = Net("Conv v1", 290)

# print(fight(n1, n2, 1, 0))

# nets = []
# # skip = 20
# for i in range(1900, 3100, 20):
#     nets.append(Net('128-128-128-128', i))
#     print('Loaded age %d' % i)

# score = tournament(nets, 1, 1, 1)
# print(score)
# print(np.argmax(score))

s100_1 = [-50, -16, -39, -24, -6, -34, 12, 1, 3, 3,
          9, 1, -6, 13, -7, -4, 6, 6, 14, 10,
          6, 0, 3, 2, 8, 3, 9, 14, 15, 9,
          17, 5, -8, 1, -7, -9, -7, 27, 20]
s100_10 = [-28, -26, -17, -31, -2, -17, 0, -13, 2, 6,
           0, 3, -5, -4, -2, 3, 8, -10, -1, 7,
           8, 17, 14, -2, 0, 12, -5, 9, 17, 25,
           13, 1, 1, -6, 0, -6, 14, 5, 10]
s50_1 = np.array([-95, -100, -46, -58, -50, -69, -24, -58, -33, -19, -41, -63, -25, 19, -16, 8, -8, 11, -3, 5, -4, 0, -31, 0, 3, -28, -31, 25, 0, -15, 3, 4, 0, -3, 9, 5, 19,
         28, 45, 32, 32, 6, 20, 6, 9, -7, 2, -4, 8, 14, 66, 7, 31, 27, 48, 21, 46, 30, 67, 22, 22, 39, 36, 3, -12, -7, -17, -1, -31, -21, -9, -18, -3, 6, 28, 32, 39, 33, 34])
s50_10 = np.array([-75, -64, -37, -46, -54, -47, -26, -40, -15, 22,
          -47, -46, -9, 8, -32, -2, -13, -6, -20, -22,
          11, -9, -29, -39, -4, -2, -19, -9, 5, 7,
          17, -5, 14, -1, -3, 12, -19, 20, -9, 36,
          1, -3, 20, -3, 10, 37, 31, 21, -21, 27,
          32, 15, 22, 2, 47, 10, 20, 13, 47, 54,
          11, 19, 20, 20, 25, -5, -12, 11, -33, -21,
          3, -36, 27, 15, 26, 40, 40, 31, 34])
# (1900, 3100, 20)
s = [-4, -7, -17, -8, 1, -18, -28, 8, 1, -7, -27, -2, 3, 8, -2, -20, -30, 18, 10, -10, -8, 8, 9, 1, 10, 6, -12, -4, -20, -8, 7, 3, 3, -10, -8, -12, 13, 13, 22, 18, 13, 5, 4, 12, -8, 2, -7, -5, -20, 1, -9, 5, 4, 19, 34, 19, 14, -8, 14, 11]

# plt.plot(100*np.arange(1, len(s100_1)+1), s100_1, 'b-')
# plt.plot(100*np.arange(1, len(s100_10)+1), s100_10, 'r-')
# plt.plot(50*np.arange(1, len(s50_1)+1), s50_1/2, 'g-')
# plt.plot(50*np.arange(1, len(s50_10)+1), s50_10/2, 'c-')
# plt.plot(1900+20*np.arange(len(s)), s, 'c-')
# plt.show()

# n = Net("128-128-128-128", 2980)
n = Net("Conv v1", 2540)

n.train(5, 50)
# n.playHuman(500, 0)
