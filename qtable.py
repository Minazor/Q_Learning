from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import os

style.use("ggplot")

if not os.path.exists("qtable_charts"):
    os.makedirs("qtable_charts")


def get_q_color(value, vals):
    if value == max(vals):
        return "green", 1.0
    else:
        return "red", 0.3


fig = plt.figure(figsize=(12, 9))


for i in range(1, 25000, 10):
    print(i)
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    q_table = np.load(f"qtables/{i}-qtable.npy")

    for x, x_vals in enumerate(q_table):
        for y, y_vals in enumerate(x_vals):
            ax1.scatter(
                x,
                y,
                c=get_q_color(y_vals[0], y_vals)[0],
                marker="o",
                alpha=get_q_color(y_vals[0], y_vals)[1],
            )
            ax2.scatter(
                x,
                y,
                c=get_q_color(y_vals[1], y_vals)[0],
                marker="o",
                alpha=get_q_color(y_vals[1], y_vals)[1],
            )
            ax3.scatter(
                x,
                y,
                c=get_q_color(y_vals[2], y_vals)[0],
                marker="o",
                alpha=get_q_color(y_vals[2], y_vals)[1],
            )

            ax1.set_ylabel("Action 0")
            ax2.set_ylabel("Action 1")
            ax3.set_ylabel("Action 2")

    # plt.show()
    plt.savefig(f"qtable_charts/{i}.png")
    plt.clf()
