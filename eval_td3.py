# This file can generate two results:
# (a) Trajectories of AUVs and USV(fig 4)
# (b) Positioning error of the AUV
# you can use fig_draw_example/* to generate analysis
import math
import os
from env import Env
import numpy as np
import argparse
import copy
import pickle

# pytorch
from td3 import TD3

# args, same as train_ddpg.py, but some are dummy
parser = argparse.ArgumentParser()
# ------ training paras ------
parser.add_argument("--is_train", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--gamma", type=float, default=0.97)
parser.add_argument("--tau", type=float, default=0.001)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--replay_capa", type=int, default=20000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--policy_freq", type=int, default=2)
parser.add_argument("--repeat_num", type=int, default=10)
parser.add_argument(
    "--episode_length", type=int, default=1000, help="the length of an episode (sec)"
)
parser.add_argument("--save_model_freq", type=int, default=25)
parser.add_argument(
    "--load_ep",
    type=int,
    default=575,
    help="Load model ep. Make sure this number is divisible by save_model_freq",
)
# ------ env paras ------
parser.add_argument(
    "--R_dc",
    type=float,
    default=6.0,
    metavar="R_DC",
    help="the radius of data collection",
)
parser.add_argument("--border_x", type=float, default=200.0, help="Area x size")
parser.add_argument("--border_y", type=float, default=200.0, help="Area y size")
parser.add_argument("--n_s", type=int, default=30, help="The number of SNs")
parser.add_argument("--N_AUV", type=int, default=2, help="The number of AUVs")
parser.add_argument("--Q", type=float, default=2, help="Capacity of SNs (Mbits)")
parser.add_argument(
    "--alpha", type=float, default=0.05, help="SNs choosing distance priority"
)
args = parser.parse_args()
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = BASE_PATH + "/models_ddpg/"
RES_PATH = BASE_PATH + "/results"
if not os.path.exists(RES_PATH):
    os.makedirs(RES_PATH)


def eval():
    for ep in range(args.repeat_num):
        state_c = env.reset()
        state = copy.deepcopy(state_c)
        # record trajs
        x_auv = [[env.xy[i][0]] for i in range(N_AUV)]
        y_auv = [[env.xy[i][1]] for i in range(N_AUV)]
        x_usv = [env.usv_xy[0]]
        y_usv = [env.usv_xy[1]]
        tracking_error = [
            [np.linalg.norm(env.obs_xy[i] - env.xy[i])] for i in range(N_AUV)
        ]
        # hovering point
        sx = [[] for i in range(N_AUV)]
        sy = [[] for i in range(N_AUV)]
        ep_r = 0
        idu = 0
        N_DO = 0
        DQ = 0
        FX = [0] * N_AUV
        sum_rate = 0
        Ec = [0] * N_AUV
        Ht = [0] * N_AUV
        Ft = 0
        crash = 0
        mode = [0] * N_AUV
        ht = [0] * N_AUV
        hovers = [False] * N_AUV  # flags
        ep_reward = 0
        while True:
            act = []
            for i in range(N_AUV):
                iact = agents[i].select_action(state[i])
                act.append(iact)
            env.posit_change(act, hovers)
            state_, rewards, Done, data_rate, ec, cs = env.step_move(hovers)
            # add_posits
            crash += cs
            ep_reward += np.sum(rewards) / 1000
            for i in range(N_AUV):
                # append positions to traj
                x_auv[i].append(env.xy[i][0])
                y_auv[i].append(env.xy[i][1])
                x_usv.append(env.usv_xy[0])
                y_usv.append(env.usv_xy[1])
                # add the tracking error
                tracking_error[i].append(np.linalg.norm(env.obs_xy[i] - env.xy[i]))
                if mode[i] == 0:
                    state[i] = copy.deepcopy(state_[i])
                    if Done[i] == True:  # SN serving
                        idu += 1
                        ht[i] = args.Q * env.updata[i] / data_rate[i]
                        mode[i] += math.ceil(ht[i])
                        # add the hovering point
                        sx[i].append(env.xy[i][0])
                        sy[i].append(env.xy[i][1])
                        hovers[i] = True
                        sum_rate += data_rate[i]
                else:
                    mode[i] -= 1
                    Ht[i] += 1
                    if mode[i] == 0:
                        hovers[i] = False
                        Ht[i] -= math.ceil(ht[i]) - ht[i]
                        state[i] = env.CHOOSE_AIM(idx=i, lamda=args.alpha)
            Ft += 1
            env.Ft = Ft
            N_DO += env.N_DO
            FX = np.array(FX) + np.array(env.FX)
            DQ += sum(env.b_S / env.Fully_buffer)
            Ec = np.array(Ec) + np.array(ec)
            if Ft > args.episode_length:
                N_DO /= Ft
                DQ /= Ft
                DQ /= env.N_POI
                Ec = np.sum(np.array(Ec) / (Ft - np.array(Ht))) / N_AUV
                print(
                    "EP:{:.0f} | ep_r {:.0f} | L_data {:.2f} | sum_rate {:.2f} | idu {:.2f} | ec {:.2f} | N_D {:.0f} | CS {} | FX {}".format(
                        ep, ep_reward, DQ, sum_rate, idu, Ec, N_DO, crash, FX
                    )
                )
                # save the file, and set the number index
                traj_start_idx = len(
                    [f for f in os.listdir(RES_PATH) if f.lower().count("traj") != 0]
                )
                terror_start_idx = len(
                    [
                        f
                        for f in os.listdir(RES_PATH)
                        if f.lower().count("tracking_error") != 0
                    ]
                )
                with open(f"{RES_PATH}/traj_{traj_start_idx}.pkl", "wb") as f:
                    pickle.dump(
                        [x_auv, y_auv, x_usv, y_usv, sx, sy, env.SoPcenter, env.lda], f
                    )
                with open(
                    f"{RES_PATH}/tracking_error_{terror_start_idx}.pkl", "wb"
                ) as f:
                    pickle.dump(
                        [tracking_error, x_auv, y_auv, [env.X_max, env.Y_max, env.H]], f
                    )
                break


if __name__ == "__main__":
    env = Env(args)
    N_AUV = args.N_AUV
    state_dim = env.state_dim
    action_dim = 2
    agents = [TD3(state_dim, action_dim) for _ in range(N_AUV)]
    # load models
    for i in range(N_AUV):
        agents[i].load(SAVE_PATH, args.load_ep, idx=i)
    eval()
