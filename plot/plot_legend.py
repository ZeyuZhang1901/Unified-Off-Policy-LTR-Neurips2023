import numpy as np
from matplotlib import pyplot as plt

colors = ["#D81B60", "#1E88E5", "#FFC107", "#004D40", "#1EBC38", "#ACFE04", "#9F98BD"]

x = np.linspace(1, 100, 1000)
y = np.log(x)
y1 = np.sin(x)
fig = plt.figure("Line plot")
legendFig = plt.figure("Legend plot", figsize=(8, 0.5))
ax = fig.add_subplot(111)
(line1,) = ax.plot(x, y, c=colors[0], lw=1, marker="^")
(line2,) = ax.plot(x, y1, c=colors[1], lw=1, marker="^")
(line3,) = ax.plot(x, y1, c=colors[2], lw=1, marker="^")
(line4,) = ax.plot(x, y1, c=colors[3], lw=1, marker="^")
(line5,) = ax.plot(x, y1, c=colors[4], lw=1, marker="^")
(line6,) = ax.plot(x, y1, c=colors[5], lw=1, marker="^")
legendFig.legend(
    [line1, line2, line3, line4, line5, line6],
    ["err_3", "err_5", "err_10", "ndcg_3", "ndcg_5", "ndcg_10"],
    loc="lower center",
    ncols=6,
)
legendFig.savefig("./plot/legend.png")
