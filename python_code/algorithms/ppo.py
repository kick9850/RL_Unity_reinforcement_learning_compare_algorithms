# 라이브러리 불러오기
import numpy as np
import datetime
import platform
import torch
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel \
    import EngineConfigurationChannel

# 파라미터 값 세팅

load_model = False
train_mode = True

discount_factor = 0.99
actor_lr = 1e-5
critic_lr = 1e-5

# 유니티 환경 경로
game = "multi"

# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d")
save_path = f"./saved_models/{game}/PPO/{date_time}"
load_path = f"./saved_models/{game}/PPO/{date_time}"

# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 디렉토리 생성
os.makedirs(save_path, exist_ok=True)

# Actor 클래스
class Actor(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 128)
        self.mu = torch.nn.Linear(128, action_size)
        self.sigma = torch.nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        mu = torch.tanh(self.mu(x))
        sigma = F.softplus(self.sigma(x)) + 1e-5  # Make sure sigma is positive
        return mu, sigma

    def evaluate_actions(self, state, actions):
        mu, sigma = self.forward(state)
        dist = torch.distributions.Normal(mu, sigma)
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return action_log_probs, mu, entropy


# Critic 클래스
class Critic(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = torch.nn.Linear(state_size + action_size, 128)  # Reduce the size to 64 units
        self.q = torch.nn.Linear(128, 1)  # Reduce the size to 1 output unit

    def forward(self, state, action):
        action = action.view(1, -1)
        if action.numel() == 0:
            action = torch.tensor([[0, 0]], device=action.device, dtype=action.dtype)
        if state.numel() == 0:
            state = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], device=state.device, dtype=state.dtype)
        #print(state)
        #print(action)
        state_action = torch.cat([state, action], dim=1)
        #print(state_action)
        q = F.relu(self.fc1(state_action))
        q = self.q(q)
        return q


# PPOAgent 클래스 다양한 함수정의
class PPOAgent:
    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size
        # Initialize the actor and critic
        self.actor = Actor(self.state_size, self.action_size).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = Critic(self.state_size, self.action_size).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Initialize the tensorboard writer
        self.writer = SummaryWriter(save_path)

        # Other parameters
        self.gamma = discount_factor  # discount factor
        self.clip_param = 0.2  # clipping parameter for PPO

    def get_action(self, state):
        # Get the next action
        state = torch.FloatTensor(state).to(device)
        mu, sigma = self.actor(state)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()
        return action.cpu().detach().numpy()

    def train_model(self, state, action, reward, next_state, done):
        # Convert to tensors
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        done = torch.FloatTensor(done).to(device)

        # Compute the critic loss
        #print(next_state, action)
        target_v = self.critic(next_state, action)
        td_target = reward + self.gamma * target_v * (1 - done)
        v = self.critic(state, action)
        critic_loss = F.mse_loss(v, td_target.detach())

        # PPO 계산
        action_log_probs, _, entropy = self.actor.evaluate_actions(state, action)
        old_action_log_probs = action_log_probs.detach()
        ratio = torch.exp(action_log_probs - old_action_log_probs)
        surr1 = ratio * td_target
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * td_target
        actor_loss = -torch.min(surr1, surr2).mean()

        # Backpropagate the losses
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def save_model(self):
        # Save the models
        torch.save(self.actor.state_dict(), save_path + "/actor.pth")
        torch.save(self.critic.state_dict(), save_path + "/critic.pth")

    def write_summary(self, mean_score, mean_actor_loss, mean_critic1_loss, total_step, step, ep):
        self.writer.add_scalar("run/score_ep", mean_score, ep)
        self.writer.add_scalar("run/score_step", mean_score, total_step)
        self.writer.add_scalar("run/step_ep", step, ep)
        self.writer.add_scalar("model/actor_loss", mean_actor_loss, ep)
        self.writer.add_scalar("model/critic1_loss", mean_critic1_loss, ep)