import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import os
from scipy import misc
import numpy
import imageio.v2 as imageio


def dot(x,y):
    s = 0
    for i in range(min(len(x), len(y))):
        s += x[i] * y[i]
    return s

class memory_net:
    def __init__(self, img_size, init_state, memory):
        self.IMG_SIZE = img_size
        self.NET_SIZE = img_size**2
        if len(init_state) != self.NET_SIZE:
            print("STATE SIZE ERROR")
            return None
        self.init_state = init_state
        self.memory = memory
        self.weights = self.set_weights()
        self.state = init_state

    def set_weights(self):
        weights = []
        for i in range(self.NET_SIZE):
            W = list()
            for j in range(self.NET_SIZE):
                w = 0
                if i == j:
                    w = 0
                else:
                    for m in self.memory:
                        w += m[i]*m[j] / len(self.memory)
                W.append(w)
            weights.append(W)

        return weights

    def print_node_states(self):
        plt.matshow(self.get_matrix(), cmap='gray', vmin = -1, vmax = 1)
        plt.show()

    def update_network(self):
        n = random.randrange(self.NET_SIZE)
        W = self.weights[n]
        update = dot(W, self.state)
        if update >= 0:
            self.state[n] = 1
        else:
            self.state[n] = -1

    def get_matrix(self):
        mat = list()
        for i in range(self.IMG_SIZE):
            row = list()
            for j in range(self.IMG_SIZE):
                row.append(self.state[i*self.IMG_SIZE + j])
            mat.append(row)
        return mat

    def run_net(self, N, r):
        matrices = list()
        for i in range(N):
            if i % int(N/r) == 0:
                matrices.append(self.get_matrix())
            self.update_network()
        return matrices

    def animate(self, N, r, dt):
        matrices = self.run_net(N, r)
        num_frames = len(matrices)
        plt.rcParams["figure.figsize"] = [14.00, 7.00]
        plt.rcParams["figure.autolayout"] = True

        fig, ax = plt.subplots()
        time_text = ax.text(-0.15, 0.95,'',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
        im = ax.imshow(matrices[0], cmap='gray', vmin = -1, vmax = 1)

        def update(i):
            im.set_data(matrices[i])
            time_text.set_text('time = %.1d' % i)

        anim = FuncAnimation(fig, update, frames=num_frames, interval=dt)
        plt.show()

# loading the images
path = "font_images/"
nums = [97, 45, 87]
IMG_SIZES = list()
Mem = list()

for num in nums:
    im = imageio.imread(path + str(num) + '.png')

    mem_image = list(map(lambda l : list(map(lambda s : 1 if s[3] > 100 else -1, l)), im))

    IMG_SIZES.append(im.shape[0])
    # flatten and append the list
    Mem.append([item for sublist in mem_image for item in sublist])

IMG_SIZE = IMG_SIZES[0]
NET_SIZE = IMG_SIZE**2
if not all(s == IMG_SIZE for s in IMG_SIZES):
    print('ERROR INCOMPATIBLE IMAGE SIZES')
    print(IMG_SIZES)
    exit()
# make network with random initial state
init_state = list(map(lambda x : 1 if x > 0.5 else -1, numpy.random.rand(NET_SIZE)))
M = memory_net(IMG_SIZE, init_state, Mem)
M.animate(1000, 100, 1)
