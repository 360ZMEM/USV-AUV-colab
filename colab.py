import numpy as np
from tidewave_usbl import TideWave
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

X_max = 120
Y_max = 120
H = 100
T_max = 500
tidewave = TideWave(H, X_max, Y_max, T_max)
tidewave.calc_tideWave()


def get_waterHeight(pos_usv):
    # return 100 # horizontal plane
    return tidewave.get_tideHeight(pos_usv[0] / X_max, pos_usv[1] / Y_max, 100)


def calcnegdetJ_USV(pos_usv, pos_auv):  # pos_auv -> 3d, pos_usv -> 2d
    pos_usv_3d = np.zeros(3)
    pos_usv_3d[:2] = pos_usv
    pos_usv_3d[2] = get_waterHeight(pos_usv)
    pos_usv = pos_usv_3d
    S_i = np.zeros(pos_auv.shape[0])
    p_i = np.zeros(pos_auv.shape[0])
    A_i = np.zeros(pos_auv.shape[0])
    # we don't consider coeffs
    for i in range(pos_auv.shape[0]):
        S_i[i] = np.linalg.norm(pos_usv - pos_auv[i])
        p_i[i] = np.linalg.norm(pos_auv[i][:2])
        A_i[i] = (p_i[i] ** 4 - 2 * (S_i[i] ** 2) * (p_i[i] ** 2)) / (2 * (S_i[i] ** 6))
    det_J1 = np.sum(S_i ** (-2))
    det_J2 = np.sum(2 * A_i + S_i ** (-2))
    det_J3 = 0
    for i in range(pos_auv.shape[0]):
        for j in range(i + 1, pos_auv.shape[0]):
            vi = pos_auv[i][:2] - pos_usv[:2]
            vj = pos_auv[j][:2] - pos_usv[:2]
            sinij = np.linalg.norm(np.cross(vi, vj)) / (
                np.linalg.norm(vi) * np.linalg.norm(vj)
            )
            det_J3 += 4 * A_i[i] * A_i[j] * (sinij) ** 2
    # if any value is not reasonable, return 0
    if np.sum(np.isnan(np.array([det_J1, det_J2, det_J3]))) != 0:
        return 0
    else:
        return -(det_J1 * det_J2 + det_J3)


def calcposit_USV(bounds, tol, pos_auv):
    calc_detJ = lambda pos_usv: calcnegdetJ_USV(pos_usv, pos_auv)
    return differential_evolution(calc_detJ, bounds=bounds, tol=tol, maxiter=500).x


# Example
pos_auv = np.array(
    [[36, 93, 0], [55, 44, 0], [15, 19, 0], [72, 26, 0]]
)  # feel free to change this
CALC_F = 1
x_start, x_end = (0, int(X_max * CALC_F))
y_start, y_end = (0, int(Y_max * CALC_F))
x = np.arange(x_start, x_end) / CALC_F
y = np.arange(y_start, y_end) / CALC_F
X, Y = np.meshgrid(x, y)
detJ = np.zeros(X.shape)
for i in range(len(x)):
    for j in range(len(y)):
        detJ[j, i] = -calcnegdetJ_USV(np.array([x[i], y[j]]), pos_auv)
max_ind_flat = np.argmax(detJ)
max_index = np.unravel_index(max_ind_flat, detJ.shape)
print("grid search result [x,y]/ m:", [x[max_index[1]], y[max_index[0]]])
optim_pos_usv = calcposit_USV(
    bounds=[(0, X_max), (0, Y_max)], tol=0.01, pos_auv=pos_auv
)
print("optimize result [x,y]/ m:", optim_pos_usv)
plt.figure(figsize=(7, 5))
plt.scatter(
    optim_pos_usv[0] * CALC_F,
    optim_pos_usv[1] * CALC_F,
    color="grey",
    label="Optimal USV Point",
    zorder=5,
    edgecolors="k",
    linewidth=1.1,
)
plt.scatter(
    pos_auv[:, 0] * CALC_F,
    pos_auv[:, 1] * CALC_F,
    color="white",
    label="AUV Points",
    zorder=5,
    edgecolors="k",
    linewidth=1.1,
)
# draw colormap
c = plt.imshow(detJ, origin="lower", cmap="viridis", interpolation="bicubic")  #
plt.colorbar(c, label="detJ relative value")
plt.legend()
plt.savefig("test.png", dpi=300)
