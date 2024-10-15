import math
import os
import time
import datetime
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.legend_handler import HandlerTuple
from scipy.ndimage import gaussian_filter1d
import matplotlib.colors as colors
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
from matplotlib import rcParams

config = {"font.size": 11}
rcParams.update(config)
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--shown_index",
    type=int,
    default=1,
    help="The index of results that are used to illustrate the figure.",
)
parser.add_argument(
    "--truncate_length",
    type=int,
    default=600,
    help="Truncate trajectories to a specific length to improve readability",
)
args = parser.parse_args()

SHOWN_INDEX = args.shown_index
# Truncate trajectories to a specific length to improve readability
TRUNCATE_LENGTH = args.truncate_length
BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir)
RES_PATH = BASE_PATH + "/results"
# load the file as the format shown
with open(f"{RES_PATH}/traj_{SHOWN_INDEX}.pkl", "rb") as f:
    x_auv, y_auv, x_usv, y_usv, sx, sy, SoPcenter, lda = pickle.load(f)
with open(f"{RES_PATH}/tracking_error_{SHOWN_INDEX}.pkl", "rb") as f:
    _, _, _, parameters = pickle.load(f)
x_max, y_max, _ = parameters

N_AUV = len(x_auv)
# trajectory processing
draw_x_auv = [[x_auv[i][0]] for i in range(N_AUV)]
draw_y_auv = [[y_auv[i][0]] for i in range(N_AUV)]
draw_x_usv = [x_usv[0]]
draw_y_usv = [y_usv[0]]
sidx_t = [0] * N_AUV  # truncated
traj_fineness = 4
for i in range(N_AUV):
    oidx = 0
    # previous point
    for idx in range(1, max(len(x_auv[i]), TRUNCATE_LENGTH)):
        point_d = np.linalg.norm(
            np.array([x_auv[i][idx] - x_auv[i][oidx], y_auv[i][idx] - y_auv[i][oidx]])
        )
        if point_d > traj_fineness:
            draw_x_auv[i].append(x_auv[i][idx])
            draw_y_auv[i].append(y_auv[i][idx])
            oidx = idx
        # check if the point is hover point
        if len(sx[i]) != 0:
            if [x_auv[i][idx], y_auv[i][idx]] == [sx[i][sidx_t[i]], sy[i][sidx_t[i]]]:
                sidx_t[i] += 1 if (len(sx[i]) - 1 > sidx_t[i]) else 0
                if oidx != idx:
                    draw_x_auv[i].append(x_auv[i][idx])
                    draw_y_auv[i].append(y_auv[i][idx])
                    oidx = idx
# Also, asv
oidx = 0
for idx in range(1, max(len(x_usv), TRUNCATE_LENGTH)):
    point_d = np.linalg.norm(
        np.array([x_usv[idx] - x_usv[oidx], y_usv[idx] - y_usv[oidx]])
    )
    if point_d > traj_fineness:
        draw_x_usv.append(x_usv[idx])
        draw_y_usv.append(y_usv[idx])
        oidx = idx


fig = plt.figure(figsize=(4.9, 4.9))
ax = fig.add_subplot(1, 1, 1)
plt.grid(linestyle="--", color="#cccccc")
ax.set_aspect(1)
# assign colors
P_length = len(lda) // 2
X_length = len(lda) - P_length
p_shape = ["P"] * P_length + ["X"] * X_length
p_size = [48] * P_length + [45] * X_length
p_color = ["indianred", "darkorange", "lightseagreen", "darkviolet", "gainsboro"]
if len(lda) > 5:
    p_color += list(cm.rainbow(np.linspace(0.05, 0.95, len(lda) - 5)))
U2_shape = ["d", "s"]  # start_point, end_point shape of AUV/USV
U2_color = ["mediumorchid", "salmon"]
if len(x_auv) > 2:
    U2_color += list(cm.nipy_spectral(np.linspace(0.1, 0.9, len(x_auv) - 2)))
U2_color.append("limegreen")
lw_SNpoint = 0.7
lw_U2point = 1.1  # scatter
lw_line = 1.8

# get the lambda
lmda = np.sort(np.unique(np.array(lda)))
# fig objs
fig_obj_SNs = []
fig_obj_lines = []
fig_obj_startpoint = []
fig_obj_endpoint = []
fig_obj_hoverarea = []


for idx, l in enumerate(lmda):
    SN_xy = SoPcenter[np.array(lda) == l]
    sn_obj = ax.scatter(
        SN_xy[:, 0],
        SN_xy[:, 1],
        marker=p_shape[idx],
        s=p_size[idx],
        color=p_color[idx],
        label=f"SN λ={l}",
        edgecolors="k",
        linewidths=lw_SNpoint,
    )
    fig_obj_SNs.append(sn_obj)

# start point & end point
for i in range(N_AUV):
    (line_obj,) = ax.plot(
        draw_x_auv[i],
        draw_y_auv[i],
        linestyle="--",
        linewidth=lw_line,
        color=U2_color[i],
    )
    # start point
    sp_obj = ax.scatter(
        draw_x_auv[i][0],
        draw_y_auv[i][0],
        marker=U2_shape[0],
        color=U2_color[i],
        edgecolors="k",
        linewidths=lw_U2point,
    )
    # end point
    ep_obj = ax.scatter(
        draw_x_auv[i][-1],
        draw_y_auv[i][-1],
        marker=U2_shape[1],
        color=U2_color[i],
        edgecolors="k",
        linewidths=lw_U2point,
    )
    # hovering area
    for idx in range(sidx_t[i]):
        circle = plt.Circle(
            (sx[i][idx], sy[i][idx]), 6, fill=True, color=U2_color[i], alpha=0.2
        )
        circle2 = plt.Circle(
            (sx[i][idx], sy[i][idx]), 6, fill=False, color=U2_color[i], alpha=0.4
        )
        ax.add_artist(circle)
        ax.add_artist(circle2)
    fig_obj_lines.append(line_obj)
    fig_obj_startpoint.append(sp_obj)
    fig_obj_endpoint.append(ep_obj)
    if sidx_t[i] != 0:
        fig_obj_hoverarea.append(circle2)
# finally, ASV
fig_obj_lines.append(
    ax.plot(
        draw_x_usv, draw_y_usv, linestyle="--", linewidth=lw_line, color=U2_color[-1]
    )[0]
)
fig_obj_startpoint.append(
    ax.scatter(
        draw_x_usv[0],
        draw_y_usv[0],
        marker=U2_shape[0],
        color=U2_color[-1],
        edgecolors="k",
        linewidths=lw_U2point,
    )
)
fig_obj_endpoint.append(
    ax.scatter(
        draw_x_usv[-1],
        draw_y_usv[-1],
        marker=U2_shape[1],
        color=U2_color[-1],
        edgecolors="k",
        linewidths=lw_U2point,
    )
)
plt.rcParams.update({"font.size": 7})  # small text for legend
ax.legend(
    [tuple(fig_obj_SNs)]
    + fig_obj_lines
    + [tuple(fig_obj_startpoint), tuple(fig_obj_endpoint)]
    + fig_obj_hoverarea,
    ["SNs priority from low to high"]
    + [f"AUV{i+1} trajectory" for i in range(N_AUV)]
    + ["USV trajectory", "Start point", "End point"]
    + [f"AUV{i+1} coverage area" for i in range(N_AUV)],
    handler_map={tuple: HandlerTuple(ndivide=None)},
    ncol=2,
    loc="upper right",
    bbox_to_anchor=(0.9, 1.2),
)
plt.rcParams.update({"font.size": 11})
ax.set_xlabel("X-axis (m)")
ax.set_ylabel("Y-axis (m)")
plt.tight_layout()
plt.show()
# plt.savefig("test.png", dpi=300) # for no-gui run
