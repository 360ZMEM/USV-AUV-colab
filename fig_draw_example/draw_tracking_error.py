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
import numpy as np
from matplotlib import rcParams
from tidewave_usbl import TideWave, USBL

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
args = parser.parse_args()
SHOWN_INDEX = args.shown_index
BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir)
RES_PATH = BASE_PATH + "/results"
with open(f"{RES_PATH}/tracking_error_{SHOWN_INDEX}.pkl", "rb") as f:
    tracking_error, x_auv, y_auv, parameters = pickle.load(f)
N_AUV = len(x_auv)
x_max, y_max, H = parameters
# we directly call funcs from tidewave_usbl to calculate error
usbl = USBL()
tidewave = TideWave(H=H, X_max=x_max, Y_max=y_max, T_max=len(tracking_error))
tidewave.calc_tideWave()
# then we get tracking error for (0,0) and midpoint
terror_point_o = [[] for i in range(N_AUV)]  # (0,0)
terror_point_m = [
    [] for i in range(N_AUV)
]  # midpoint, 200*0.5=100m, (100m,100m) by default
for i in range(len(tracking_error[0])):
    if i > 180:
        _ = 5
    tide_h_o = tidewave.get_tideHeight(0, 0, i)
    tide_h_m = tidewave.get_tideHeight(0.5, 0.5, i)
    for j in range(N_AUV):
        real_auv_posit_o = np.array([x_auv[j][i], y_auv[j][i], tide_h_o])
        real_auv_posit_m = np.array(
            [x_auv[j][i] - 0.5 * x_max, y_auv[j][i] - 0.5 * y_max, tide_h_o]
        )
        # calc posit
        pred_auv_posit_o = usbl.calcPosit(real_auv_posit_o)[:2]
        pred_auv_posit_m = usbl.calcPosit(real_auv_posit_m)[:2]
        # get the tracking error
        terror_point_o[j].append(
            np.linalg.norm(real_auv_posit_o[:2] - pred_auv_posit_o)
        )
        terror_point_m[j].append(
            np.linalg.norm(real_auv_posit_m[:2] - pred_auv_posit_m)
        )
# we sum the error by N_AUV
tracking_error = np.sum(tracking_error, axis=0)
terror_point_o = np.sum(np.array(terror_point_o), axis=0)
terror_point_m = np.sum(np.array(terror_point_m), axis=0)
# then we draw the picture
fig2 = plt.figure(figsize=(4.5, 3.9))
ax2 = fig2.add_subplot(1, 1, 1)
ax2.grid(linestyle="--", color="#cccccc")
# the color assignment
# colors = cm.nipy_spectral(np.linspace(0.05,0.95,3))
colors = ["dodgerblue", "orange", "forestgreen"]
lw_front = 1.6
lw_back = 0.9
# draw background
plt.semilogy(tracking_error, color=colors[0], alpha=0.2, linewidth=lw_back)
plt.semilogy(terror_point_o, color=colors[1], alpha=0.2, linewidth=lw_back)
plt.semilogy(terror_point_m, color=colors[2], alpha=0.2, linewidth=lw_back)
# front
smooth_W = 10
smoothed_tracking_error = gaussian_filter1d(tracking_error, smooth_W)
smoothed_terror_point_o = gaussian_filter1d(terror_point_o, smooth_W)
smoothed_terror_point_m = gaussian_filter1d(terror_point_m, smooth_W)
plt.semilogy(
    smoothed_tracking_error, color=colors[0], linewidth=lw_front, label="USV-AUV Colab"
)
plt.semilogy(
    smoothed_terror_point_o, color=colors[1], linewidth=lw_front, label="Fixed (0,0)"
)
plt.semilogy(
    smoothed_terror_point_m, color=colors[2], linewidth=lw_front, label="Fixed midpoint"
)
plt.xlabel("time (s)")
plt.ylabel("error (m)")
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
# plt.savefig("test.png", dpi=300) # for no-gui run
