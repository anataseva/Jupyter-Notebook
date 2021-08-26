Searching for patterns in Conway’s Game of Life
Мodeling the Conway's Game of Life

John Horton Conway, in the late 60s of last century created a game called life. The name of the game itself derives from the social life of a person. It is a simulation game with unusual patterns. To start the game requires the player to station a figure on some fields from one infinite, one-color chessboard. Two models of the game obtained by means of integer programming will be presented, a task for maximum initial density for a continuously live layout.

print("Proof of the theorem")
Proof of the theorem
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def create_world(n):
    return np.random.choice((1, 0), n * n, p=(0.2, 0.8)).reshape(n, n)


def evolve_world(k, x, img):

    n = np.int(np.sqrt(np.size(x)))
    y = np.copy(x)

    for i in range(n):
        for j in range(n):

            s = x[(i - 1) % n][(j - 1) % n] + x[(i - 1) % n][j] + x[(i - 1) % n][(j + 1) % n] \
              + x[i][(j - 1) % n]                                         + x[i][(j + 1) % n] \
              + x[(i + 1) % n][(j - 1) % n] + x[(i + 1) % n][j] + x[(i + 1) % n][(j + 1) % n]

            if x[i][j] == 1:
                if s == 2 or s == 3:
                    y[i][j] = 1
                else:
                    y[i][j] = 0
            else:
                if s == 3:
                    y[i][j] = 1
                else:
                    y[i][j] = 0

    img.set_data(y)
    x[:] = y[:]
    return img


def main():

    n = 25
    x = create_world(n)

    fig, ax = plt.subplots()
    img = ax.imshow(x)

    evolution = animation.FuncAnimation(fig, evolve_world, interval=30, fargs=(x, img))
    plt.show()


main()

print("1.The first characteristic type is called constant")
1.The first characteristic type is called constant
import matplotlib.pyplot as plt

ax = plt.axes()
ax.arrow(0, 0, 0.5, 0.5, head_width=0.05, head_length=0.1, fc='k', ec='k')
plt.show()

print("2.The second characteristic type is called an oscillator")
2.The second characteristic type is called an oscillator
import numpy as np
import matplotlib.pyplot as plt

# Fixing random state for reproducibility
np.random.seed(19680801)


def color_cycle_example(ax):
    L = 6
    x = np.linspace(0, L)
    ncolors = len(plt.rcParams['axes.prop_cycle'])
    shift = np.linspace(0, L, ncolors, endpoint=False)
    for s in shift:
        ax.plot(x, np.sin(x + s), 'o-')


def image_and_patch_example(ax):
    ax.imshow(np.random.random(size=(20, 20)), interpolation='none')
    c = plt.Circle((5, 5), radius=5, label='patch')
    ax.add_patch(c)


plt.style.use('grayscale')

fig, (ax1, ax2) = plt.subplots(ncols=2)
fig.suptitle("oscilations")

color_cycle_example(ax1)
image_and_patch_example(ax2)

plt.show()

import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0.0, 2.0, 0.01)
s = np.sin(2 * np.pi * t)

upper = 1
lower = -1

supper = np.ma.masked_where(s < upper, s)
slower = np.ma.masked_where(s > lower, s)
smiddle = np.ma.masked_where((s < lower) | (s > upper), s)

fig, ax = plt.subplots()
ax.plot(t, smiddle, t, slower, t, supper)
plt.show()

print("3.The third characteristic type is that a certain period is repeated in a different position")
3.The third characteristic type is that a certain period is repeated in a different position
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

Path = mpath.Path

fig, ax = plt.subplots()
pp1 = mpatches.PathPatch(
    Path([(0, 0), (1, 0), (1, 1), (0, 0)],
         [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]),
    fc="none", transform=ax.transData)

ax.add_patch(pp1)
ax.plot([0.75], [0.25], "ro")
ax.set_title('The red point is figures different position')

plt.show()
