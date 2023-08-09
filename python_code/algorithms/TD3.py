from collections import namedtuple
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

state_dim = 12 * 4
action_dim = 1
max_action = 1

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.fc1(state))
        a = F.relu(self.fc2(a))
        a = torch.tanh(self.fc3(a)) * self.max_action
        return a


class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)

        q = F.relu(self.fc1(state_action))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q


class TD3():
    def __init__(self,
                 capacity,
                 directory,
                 batch_size,
                 policy_noise,
                 policy_delay,
                 noise_clip,
                 gamma,
                 tau
                 ):
        self.capacity = capacity
        self.directory = directory
        self.policy_delay = policy_delay
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau

        self.actor = Actor().to(device)
        self.actor_target = Actor().to(device)
        self.critic_1 = Critic().to(device)
        self.critic_1_target = Critic().to(device)
        self.critic_2 = Critic().to(device)
        self.critic_2_target = Critic().to(device)

        self.actor_optimizer = optim.Adam(self.actor.parameters())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters())

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.memory = Replay_buffer(self.capacity)
        self.writer = SummaryWriter(self.directory)

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1)).float().to(device)
        return self.actor(state).cpu().data.numpy()

    def update(self, num_iteration):
        for i in range(num_iteration):
            x, y, u, r, d = self.memory.sample(self.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Select next action according to target policy:
            noise = torch.ones_like(action).data.normal_(0, self.policy_noise).to(device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-max_action, max_action)

            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1 - done) * self.gamma * target_Q).detach()

            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()

            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()

            # Delayed policy updates:
            if i % self.policy_delay == 0:
                # Compute actor loss:
                actor_loss = - self.critic_1(state, self.actor(state)).mean()

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(((1- self.tau) * target_param.data) + self.tau * param.data)

                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(((1 - self.tau) * target_param.data) + self.tau * param.data)

                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(((1 - self.tau) * target_param.data) + self.tau * param.data)

                return actor_loss.item(), loss_Q1.item(), loss_Q2.item()

    def save(self):
        torch.save(self.actor.state_dict(), self.directory+'actor.pth')
        torch.save(self.actor_target.state_dict(), self.directory+'actor_target.pth')
        torch.save(self.critic_1.state_dict(), self.directory+'critic_1.pth')
        torch.save(self.critic_1_target.state_dict(), self.directory+'critic_1_target.pth')
        torch.save(self.critic_2.state_dict(), self.directory+'critic_2.pth')
        torch.save(self.critic_2_target.state_dict(), self.directory+'critic_2_target.pth')

    def load(self):
        self.actor.load_state_dict(torch.load(self.directory + 'actor.pth'))
        self.actor_target.load_state_dict(torch.load(self.directory + 'actor_target.pth'))
        self.critic_1.load_state_dict(torch.load(self.directory + 'critic_1.pth'))
        self.critic_1_target.load_state_dict(torch.load(self.directory + 'critic_1_target.pth'))
        self.critic_2.load_state_dict(torch.load(self.directory + 'critic_2.pth'))
        self.critic_2_target.load_state_dict(torch.load(self.directory + 'critic_2_target.pth'))

    def write_summary(self, mean_score, mean_actor_loss, mean_critic1_loss, mean_critic2_loss, total_step, step, ep):
        self.writer.add_scalar("run/score_ep", mean_score, ep)
        self.writer.add_scalar("run/score_step", mean_score, total_step)
        self.writer.add_scalar("run/step_ep", step, ep)
        self.writer.add_scalar("model/actor_loss", mean_actor_loss, ep)
        self.writer.add_scalar("model/critic1_loss", mean_critic1_loss, ep)
        self.writer.add_scalar("model/critic2_loss", mean_critic2_loss, ep)