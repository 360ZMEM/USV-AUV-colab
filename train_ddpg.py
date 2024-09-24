# ----------------------------------------
#  Training DDPG agents and saving models
# ----------------------------------------
import math
import os
from env import Env
import numpy as np
import argparse
import copy
# pytorch
from td3 import TD3

# args
parser = argparse.ArgumentParser()
# ------ training paras ------
parser.add_argument('--is_train', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gamma', type=float, default=0.97)
parser.add_argument('--tau', type=float, default=0.001)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--replay_capa', type=int, default=20000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--policy_freq', type=int, default=2)
parser.add_argument('--episode_num', type=int, default=3202)
parser.add_argument('--episode_length', type=int, default=1000, help='the length of an episode (sec)')
parser.add_argument('--save_model_freq', type=int, default=25)
# ------ env paras ------
parser.add_argument('--R_dc', type=float, default=6., metavar='R_DC',help='the radius of data collection')
parser.add_argument('--border_x', type=float, default=200.,help='Area x size')
parser.add_argument('--border_y', type=float, default=200.,help='Area y size')
parser.add_argument('--n_s', type=int, default=30, help='The number of SNs')
parser.add_argument('--N_AUV', type=int, default=2, help='The number of AUVs')
parser.add_argument('--Q', type=float, default=2, help='Capacity of SNs (Mbits)')
parser.add_argument('--alpha', type=float, default=0.05, help='SNs choosing distance priority')
args = parser.parse_args()
SAVE_PATH = os.getcwd()
SAVE_PATH = SAVE_PATH + '/models_ddpg/'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
def train():
    # noise var
    noise = 0.8
    for ep in range(args.episode_num):
        state_c = env.reset()
        state = copy.deepcopy(state_c) 
        ep_r = 0
        idu = 0
        N_DO = 0
        DQ = 0 
        FX = [0] * N_AUV
        sum_rate = 0
        Ec = [0] * N_AUV
        TD_error = [0] * N_AUV
        A_Loss = [0] * N_AUV
        Ht = [0] * N_AUV
        Ft = 0
        update_network = [0] * N_AUV
        crash = 0
        mode = [0] * N_AUV
        ht = [0] * N_AUV
        hovers = [False] * N_AUV  # flags
        ep_reward = 0
        while True:
            act = []
            # choose action
            for i in range(N_AUV):
                iact = agents[i].select_action(state[i])
                iact = np.clip(iact + noise * np.random.randn(2), -1, 1)
                act.append(iact)
            env.posit_change(act,hovers) 
            state_,rewards,Done,data_rate,ec,cs=env.step_move(hovers)
            crash += cs
            ep_reward += np.sum(rewards) / 1000
            # store transition
            for i in range(N_AUV):
                if mode[i] == 0:
                    # if abs(act[i][0] - 0) <= 0.005 or abs(act[i][1] - 0) <= 0.005:
                    #     _ = 5
                    agents[i].store_transition(state[i],act[i],rewards[i],state_[i],False)
                    state[i] = copy.deepcopy(state_[i])
                    if Done[i] == True:
                        idu += 1
                        ht[i] = args.Q * env.updata[i] / data_rate[i]  # TO BE 确认
                        mode[i] += math.ceil(ht[i])
                        # 新增部分
                        hovers[i] = True
                        sum_rate +=  data_rate[i]
                else:
                    mode[i] -= 1
                    Ht[i] += 1
                    if mode[i] == 0:
                        hovers[i] = False
                        Ht[i] -= (math.ceil(ht[i]) - ht[i])
                        state[i] = env.CHOOSE_AIM(idx=i,lamda=args.alpha)
                # training
                if len(agents[i].replay_buffer) > 20 * args.batch_size:
                    a_loss,td_error =  agents[i].train()
                    noise = max(noise*0.99998, 0.1)
                    update_network[i] += 1
                    TD_error[i] += td_error
                    A_Loss[i] += a_loss
            Ft += 1
            env.Ft = Ft
            N_DO += env.N_DO
            FX = np.array(FX) + np.array(env.FX)
            DQ += sum(env.b_S/env.Fully_buffer)
            Ec = np.array(Ec) + np.array(ec)
            if Ft > args.episode_length:
                for i in range(N_AUV):
                    if update_network[i] != 0:
                        TD_error[i] /= update_network[i]
                        A_Loss[i] /= update_network[i]
                N_DO /= Ft
                DQ /= Ft
                DQ /= env.N_POI
                Ec = np.sum(np.array(Ec) / (Ft - np.array(Ht))) / N_AUV
                print('EP:{:.0f} | TD Error {} | ALoss {} | ep_r {:.0f} | L_data {:.2f} | sum_rate {:.2f} | idu {:.2f} | ec {:.2f} | N_D {:.0f} | CS {} | FX {}'.format(ep,TD_error,A_Loss,ep_reward,DQ,sum_rate,idu,Ec,N_DO,crash,FX))
                break
        # save models
        if ep % args.save_model_freq == 0 and ep != 0:
            for i in range(N_AUV):
                agents[i].save(SAVE_PATH, ep, idx=i)
            

# main
if __name__ == '__main__':
    env = Env(args)
    N_AUV = args.N_AUV
    state_dim = env.state_dim
    action_dim = 2
    # agents
    agents = [TD3(state_dim, action_dim) for _ in range(N_AUV)]
    train()
