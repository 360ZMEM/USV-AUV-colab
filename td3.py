import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DDPG
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.fc4(x))


class QValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet, self).__init__()
        # ---- Q1 ----
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)
        # ---- Q2 ----
        self.fc3 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1) 
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        Q1 = self.fc_out(x)
        # Q2
        x = F.relu(self.fc3(cat))
        x = F.relu(self.fc4(x))
        Q2 = self.fc_out2(x)
        return Q1, Q2
    # only return Q1 value(https://github.com/sfujim/TD3)
    def Q1(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        Q1 = self.fc_out(x)
        return Q1
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state,action,reward,next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state,action,reward,next_state, done)
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state,action,reward,next_state, done = map(np.stack, zip(*batch))
        return state,action,reward,next_state, done
    
    def __len__(self):
        return len(self.buffer)


    
class TD3(object):
    def __init__(
		self,
		state_dim,
		action_dim,
        hidden_dim=128,
		discount=0.97,
		tau=0.001,
        lr=1e-3,
        batch_size=64
	):
        self.actor = PolicyNet(state_dim,hidden_dim,action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic = QValueNet(state_dim,hidden_dim,action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.discount = discount
        self.tau = tau
        self.replay_buffer = ReplayBuffer(20000)
        self.policy_freq = 2
        self.total_it = 0
        self.batch_size = batch_size

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).detach().cpu().numpy().flatten() # 1-dim
    
    def store_transition(self,state,action,reward,next_state, done=False):
        self.replay_buffer.push(state,action,reward,next_state,done)

    def train(self):
        state,action,reward,next_state,done = self.replay_buffer.sample(batch_size=self.batch_size)
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device).unsqueeze(1)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device).unsqueeze(1)
        # Critic loss
        with torch.no_grad():
            noise = (torch.randn_like(action) * 0.1).clamp(-1.0,1.0)
            next_action = (self.actor_target(next_state) + noise).clamp(-1,1)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1-done) * self.discount * target_Q
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_it += 1
        # loss
        try:
            return actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy()
        except:
            return 0, critic_loss.detach().cpu().numpy()

    # save model
    def save(self, filename, ep, idx):
        ep = '_' + str(ep)
        idx = '_' + str(idx)
        torch.save(self.critic.state_dict(), filename + "TD3" + idx + ep + "_critic.pth")
        torch.save(self.critic_optimizer.state_dict(), filename + "TD3" + idx + ep  + "_critic_optimizer.pth")
        torch.save(self.actor.state_dict(), filename + "TD3" + idx  + ep  + "_actor.pth")
        torch.save(self.actor_optimizer.state_dict(), filename + "TD3" + idx  + ep  + "_actor_optimizer.pth")

    # load model
    def load(self, filename, ep, idx):
        ep = '_' + str(ep)
        self.critic.load_state_dict(torch.load(filename + "TD3" + idx  + ep + "_critic.pth"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "TD3" + idx   + ep  + "_critic_optimizer.pth"))
        self.actor.load_state_dict(torch.load(filename + "TD3" + idx + ep + "_actor.pth"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "TD3"  + idx + ep + "_actor_optimizer.pth"))
        # target networks
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

