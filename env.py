import numpy as np
import math
import copy
from tidewave_usbl import TideWave, USBL
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import time


class Env(object):
    def __init__(self, args):
        # ---- paras args ----
        self.N_SNs = args.n_s
        self.N_AUV = args.N_AUV
        self.X_max = args.border_x
        self.Y_max = args.border_y
        self.border = np.array([self.X_max, self.Y_max])
        self.r_dc = args.R_dc
        self.N_POI = args.n_s
        self.epi_len = args.episode_length
        # ---- paras specified here ----
        self.USV_SHIFT_MAX = 4
        self.X_min = 0
        self.Y_min = 0
        self.r_dc = args.R_dc
        self.f = 20  # khz, AUV ~ SNs
        self.b = 1
        self.safe_dist = 10
        self.H = 100  # water depth
        self.V_max = 2.2
        self.V_min = 1.2
        self.S = 60
        self.P_u = 3e-2
        # ---- variables ----
        self.SoPcenter = np.zeros((self.N_POI, 2))  # center of SNs
        self.state_dim = 6 + 2 * (self.N_AUV - 1)
        self.state = [np.zeros(self.state_dim)] * self.N_AUV
        self.rewards = []
        self.xy = np.zeros((self.N_AUV, 2))
        self.obs_xy = np.zeros((self.N_AUV, 2))  # observation
        self.usv_xy = np.zeros(2)
        self.vxy = np.zeros((self.N_AUV, 2))
        self.dis = np.zeros((self.N_AUV, self.N_POI))
        self.dis_hor = np.zeros((self.N_AUV, self.N_POI))  # horizontal distance
        # ---- SNs ----
        self.LDA = [3, 5, 8, 12]  # poisson variables
        CoLDA = np.random.randint(0, len(self.LDA), self.N_POI)
        self.lda = [self.LDA[CoLDA[i]] for i in range(self.N_POI)]  # assign poisson
        self.b_S = np.random.randint(0.0, 1000.0, self.N_POI).astype(np.float32)
        self.Fully_buffer = 5000
        self.H_Data_overflow = [0] * self.N_AUV
        self.Q = np.array(
            [self.lda[i] * self.b_S[i] / self.Fully_buffer for i in range(self.N_POI)]
        )
        self.idx_target = np.argsort(self.Q)[-self.N_AUV :]
        self.updata = self.b_S[self.idx_target] / self.Fully_buffer
        # ---- Metrics ----
        self.FX = np.zeros(self.N_AUV)
        self.ec = np.zeros(self.N_AUV)
        self.TL = np.zeros(self.N_AUV)
        self.N_DO = 0
        self.crash = np.zeros(self.N_AUV)
        # ---- USBL ----
        self.usbl = USBL()
        # ---- TideWave ----
        self.tidewave = TideWave(self.H, self.X_max, self.Y_max, self.epi_len)
        self.tidewave.calc_tideWave()
        self.Ft = 0

    def calcnegdetJ_USV(self, posit_usv):
        S_i = np.zeros(self.xy.shape[0])
        p_i = np.zeros(self.xy.shape[0])
        A_i = np.zeros(self.xy.shape[0])
        # get the tidewave height
        pos_usv_3d = np.zeros(3)
        pos_usv_3d[:2] = posit_usv
        pos_usv_3d[2] = self.tidewave.get_tideHeight(
            posit_usv[0] / self.X_max, posit_usv[1] / self.Y_max, self.Ft
        )
        # we don't consider coeffs
        for i in range(self.xy.shape[0]):
            pos_auv_3d = np.zeros(3)
            pos_auv_3d[:2] = self.xy[i]
            S_i[i] = np.linalg.norm(pos_usv_3d - pos_auv_3d)
            p_i[i] = np.linalg.norm(pos_auv_3d)
            A_i[i] = (p_i[i] ** 4 - 2 * (S_i[i] ** 2) * (p_i[i] ** 2)) / (
                2 * (S_i[i] ** 6)
            )
        det_J1 = np.sum(S_i ** (-2))
        det_J2 = np.sum(2 * A_i + S_i ** (-2))
        det_J3 = 0
        for i in range(self.xy.shape[0]):
            for j in range(i + 1, self.xy.shape[0]):
                vi = self.xy[i] - posit_usv
                vj = self.xy[j] - posit_usv
                sinij = np.linalg.norm(np.cross(vi, vj)) / (
                    np.linalg.norm(vi) * np.linalg.norm(vj)
                )
                det_J3 += 4 * A_i[i] * A_i[j] * (sinij) ** 2
        # if any value is not reasonable, return 0
        if np.sum(np.isnan(np.array([det_J1, det_J2, det_J3]))) != 0:
            return 0
        else:
            return -(det_J1 * det_J2 + det_J3)

    # bonus func: calculate optimal position for USV ()
    def calcposit_USV(self):
        init_guess = np.mean(self.xy, axis=0) if self.Ft == 0 else self.usv_xy
        bounds = (
            [
                (init_guess[0] - self.X_max, init_guess[0] + self.X_max),
                (init_guess[1] - self.Y_max, init_guess[1] + self.Y_max),
            ]
            if self.Ft == 0
            else [
                (
                    init_guess[0] - self.USV_SHIFT_MAX,
                    init_guess[0] + self.USV_SHIFT_MAX,
                ),
                (
                    init_guess[1] - self.USV_SHIFT_MAX,
                    init_guess[1] + self.USV_SHIFT_MAX,
                ),
            ]
        )
        tol = 1e-2 if self.Ft == 0 else 5e-2
        opt_asv_posit = differential_evolution(
            self.calcnegdetJ_USV, bounds=bounds, tol=tol, maxiter=500
        )  # DE performs well in finding global solution
        self.usv_xy = opt_asv_posit.x

        # data rate calculating

    def calcRate(self, f, b, d, dir=0):
        f1 = (f - b / 2) if dir == 0 else (f + b / 2)
        lgNt = 17 - 30 * math.log10(f1)
        lgNs = 40 + 26 * math.log10(f1) - 60 * math.log10(f + 0.03)
        lgNw = 50 + 20 * math.log10(f1) - 40 * math.log10(f + 0.4)
        lgNth = -15 + 20 * math.log10(f1)
        NL = 10 * math.log10(
            1000
            * b
            * (
                10 ** (lgNt / 10)
                + 10 ** (lgNs / 10)
                + 10 ** (lgNw / 10)
                + 10 ** (lgNth / 10)
            )
        )
        alpha = (
            0.11 * ((f1**2) / (1 + f1**2))
            + 44 * ((f1**2) / (4100 + f1**2))
            + (2.75e-4) * (f1**2)
            + 0.003
        )
        TL = 15 * math.log10(d) + alpha * (0.001 * d)
        SL = 10 * math.log10(self.P_u) + 170.77
        R = 0.001 * b * math.log(1 + 10 ** (SL - TL - NL), 2)
        return R

    def get_state(self):  # new func
        for i in range(self.N_AUV):
            state = []
            # we assume that the AUVs cannot communicate directly
            # therefore, we measure the positions of AUVs by the AUV-USV communication
            meas_posit = np.zeros(3)
            meas_posit[:2] = self.xy[i]
            usv_xyz = np.zeros(3)
            usv_xyz[:2] = self.usv_xy
            usv_xyz[2] = self.tidewave.get_tideHeight(
                usv_xyz[0] / self.X_max, usv_xyz[1] / self.Y_max, self.Ft
            )
            usv_auv_diff = usv_xyz - meas_posit  # symmetry
            usv_auv_diff = self.usbl.calcPosit(usv_auv_diff, idx=i)
            meas_posit = usv_xyz - usv_auv_diff
            # input
            self.obs_xy[i][:2] = meas_posit[:2]
        # then get locs
        for i in range(self.N_AUV):
            state = []
            for j in range(self.N_AUV):
                if j == i:
                    continue
                state.append(
                    (self.obs_xy[j] - self.obs_xy[i]).flatten()
                    / np.linalg.norm(self.border)
                )
            # posit Target SNs
            state.append(
                (self.target_Pcenter[i] - self.obs_xy[i]).flatten()
                / np.linalg.norm(self.border)
            )
            state.append((self.obs_xy[i]).flatten() / np.linalg.norm(self.border))
            # finally, FX and N_DO
            state.append([self.FX[i] / self.epi_len, self.N_DO / self.N_POI])
            self.state[i] = np.concatenate(tuple(state))

    # reset
    def reset(self):
        self.FX = np.zeros(self.N_AUV)
        self.ec = np.zeros(self.N_AUV)
        self.TL = np.zeros(self.N_AUV)
        self.N_DO = 0
        self.crash = np.zeros(self.N_AUV)
        # assign x/y to SNs
        self.SoPcenter[:, 0] = np.random.randint(
            self.safe_dist, self.X_max - self.safe_dist, size=self.N_POI
        )
        self.SoPcenter[:, 1] = np.random.randint(
            self.safe_dist, self.Y_max - self.safe_dist, size=self.N_POI
        )
        # assign x/y to AUVs, the distance between AUVs > 2 * safe_dist
        while True:
            dist_ok = True
            self.xy[0] = np.random.randint(
                self.safe_dist, self.X_max - self.safe_dist, size=self.N_AUV
            )
            self.xy[1] = np.random.randint(
                self.safe_dist, self.Y_max - self.safe_dist, size=self.N_AUV
            )
            for i in range(self.N_AUV):
                for j in range(i + 1, self.N_AUV):
                    if np.linalg.norm(self.xy[i] - self.xy[j]) < 2 * self.safe_dist:
                        dist_ok = False
            if dist_ok == True:
                break
        # reset the position of ASV
        self.calcposit_USV()
        self.b_S = np.random.randint(0, 1000, self.N_POI)
        # assign target SNs
        self.Q = np.array(
            [self.lda[i] * self.b_S[i] / self.Fully_buffer for i in range(self.N_POI)]
        )
        self.idx_target = np.argsort(self.Q)[-self.N_AUV :]
        self.updata = self.b_S[self.idx_target] / self.Fully_buffer
        self.target_Pcenter = self.SoPcenter[self.idx_target]

        # states
        self.get_state()
        return self.state

    def posit_change(self, actions, hovers):
        for i in range(self.N_AUV):
            # action mapping
            actions[i][0] = 0.5 * (actions[i][0] + 1)
            detX = (actions[i][0] * (self.V_max - self.V_min) + self.V_min) * math.cos(
                actions[i][1] * math.pi
            )
            detY = (actions[i][0] * (self.V_max - self.V_min) + self.V_min) * math.sin(
                actions[i][1] * math.pi
            )
            self.vxy[i, 0] = detX
            self.vxy[i, 1] = detY
            V = math.sqrt(pow(detX, 2) + pow(detY, 2))
            if hovers[i] == True:
                detX = 0
                detY = 0
            xy_ = copy.deepcopy(self.xy[i])
            xy_[0] += detX
            xy_[1] += detY
            # getting the metric of crossing the border
            Flag = False
            self.FX[i] = (
                np.sum((xy_ - np.array([0, 0])) < 0) + np.sum((self.border - xy_) < 0)
            ) > 0
            Flag = (np.sum((xy_) < 0) + np.sum((self.border - xy_) < 0)) == 0
            if not Flag:  # Flag False -> cross the border
                xy_[0] -= detX
                xy_[1] -= detY
            if Flag and (hovers[i] == False):
                F = (0.7 * self.S * (V**2)) / 2
                self.ec[i] = (F * V) / (
                    -0.081 * (V**3) + 0.215 * (V**2) - 0.01 * V + 0.541
                ) + 15
            else:
                self.ec[i] = 90 + 15
            # assigning positions
            self.xy[i] = xy_
        self.calcposit_USV()

    def step_move(self, hovers):
        self.N_DO = 0
        self.b_S += [np.random.poisson(self.lda[i]) for i in range(self.N_POI)]
        for i in range(self.N_POI):  # check data overflow
            if self.b_S[i] >= self.Fully_buffer:
                self.N_DO += 1
                self.b_S[i] = self.Fully_buffer
        self.updata = self.b_S[self.idx_target] / self.Fully_buffer
        self.crash = np.zeros(self.N_AUV)
        self.TL = np.zeros(self.N_AUV)
        self.rewards = np.zeros(self.N_AUV)
        data_rate = np.zeros(self.N_AUV)
        # get state
        self.get_state()
        # get crash information
        for i in range(self.N_AUV):
            for j in range(self.N_AUV):
                if j == i:
                    continue
                dxy = (self.xy[j] - self.xy[i]).flatten()
                sd = np.linalg.norm(dxy)
                if sd < 5:
                    self.crash[i] += 1
            # then calculating dis AUV ~ target SNs
            self.calc_dist(i)
            if self.dis_hor[i, self.idx_target[i]] < self.r_dc:
                self.TL[i] = True
                data_rate[i] = max(
                    self.calcRate(self.f, self.b, self.dis[i, self.idx_target[i]], 0),
                    self.calcRate(self.f, self.b, self.dis[i, self.idx_target[i]], 1),
                )
                self.b_S[self.idx_target[i]] = 0
            self.rewards = self.compute_reward()
        return self.state, self.rewards, self.TL, data_rate, self.ec, self.crash

    def calc_dist(self, idx):
        # get height
        H = self.tidewave.get_tideHeight(
            self.xy[idx][0] / self.X_max, self.xy[idx][1] / self.Y_max, self.Ft
        )
        for i in range(self.N_POI):
            self.dis[idx][i] = math.sqrt(
                pow(self.SoPcenter[i][0] - self.xy[idx][0], 2)
                + pow(self.SoPcenter[i][1] - self.xy[idx][1], 2)
                + pow(self.H, 2)
            )
            self.dis_hor[idx][i] = math.sqrt(
                pow(self.SoPcenter[i][0] - self.xy[idx][0], 2)
                + pow(self.SoPcenter[i][1] - self.xy[idx][1], 2)
            )

    def CHOOSE_AIM(self, idx=0, lamda=0.05):
        self.calc_dist(idx=idx)
        Q = np.array(
            [
                self.lda[i] * self.b_S[i] / self.Fully_buffer - lamda * self.dis[idx][i]
                for i in range(self.N_POI)
            ]
        )
        idx_target = np.argsort(Q)[-self.N_AUV :]
        inter = np.intersect1d(idx_target, self.idx_target)
        if len(inter) < len(self.idx_target):
            diff = np.setdiff1d(idx_target, inter)
            self.idx_target[idx] = diff[0]
        else:
            idx_target = np.argsort(self.Q)[-(self.N_AUV + 1) :]
            self.idx_target[idx] = idx_target[0]
        self.target_Pcenter = self.SoPcenter[self.idx_target]
        # state[i]
        st_idx = 2 * (self.N_AUV - 1)
        self.state[idx][st_idx : st_idx + 2] = (
            self.target_Pcenter[idx] - self.xy[idx]
        ).flatten() / np.linalg.norm(self.border)
        self.state[idx][-1] = self.N_DO / self.N_POI
        return self.state[idx]

    def compute_reward(self):  # oracle
        reward = np.zeros(self.N_AUV)
        for i in range(self.N_AUV):
            dist_to_target = np.linalg.norm(self.xy[i] - self.target_Pcenter[i])
            reward[i] += -0.6 * dist_to_target - self.FX[i] * 0.1 - self.N_DO * 0.05
            for j in range(i + 1, self.N_AUV):
                dist_between_auvs = np.linalg.norm(self.xy[j] - self.xy[i])
                if dist_between_auvs < 12:
                    reward[i] -= 6 * (12 - dist_between_auvs)
            # rew
            if self.TL[i] > 0:
                reward[i] += 12
            reward[i] -= 0.085 * self.ec[i]  # adjust this factor
        return reward
