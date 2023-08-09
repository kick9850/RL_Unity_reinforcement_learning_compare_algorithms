import numpy as np
import random
import copy
import datetime
import platform
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque


# DDPG를 위한 파라미터 값 세팅
state_size = 12 * 4
action_size = 1

load_model = False
train_mode = True

batch_size = 128
mem_maxlen = 30000
discount_factor = 0.9
actor_lr = 1e-4
critic_lr = 5e-4
tau = 1e-3

# OU noise 파라미터
mu = 0
theta = 1e-3
sigma = 2e-3

# 유니티 환경 경로
game = "MyKartNormal"

# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d")
save_path = f"./saved_models/{game}/DDPG/{date_time}"
load_path = f"./saved_models/{game}/DDPG/20230719"

# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# OU_noise 클래스 -> ou noise 정의 및 파라미터 결정
class OU_noise:
    def __init__(self):
        self.reset()

    def reset(self):
        self.X = np.ones((1, action_size), dtype=np.float32) * mu

    def sample(self):
        dx = theta * (mu - self.X) + sigma * np.random.randn(len(self.X))
        self.X += dx
        return self.X

# Actor 클래스 -> DDPG Actor 클래스 정의
class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.mu = torch.nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.mu(x))

# Critic 클래스 -> DDPG Critic 클래스 정의
class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.fc1 = torch.nn.Linear(state_size, 128)
        self.fc2 = torch.nn.Linear(128+action_size, 128)
        self.q = torch.nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(state))
        x = torch.cat((x, action), dim=-1)
        x = torch.relu(self.fc2(x))
        return self.q(x)

# DDPGAgent 클래스 -> DDPG 알고리즘을 위한 다양한 함수 정의
class DDPGAgent():
    def __init__(self):
        self.actor = Actor().to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = Critic().to(device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.OU = OU_noise()
        self.memory = deque(maxlen=mem_maxlen)
        self.writer = SummaryWriter(save_path)

        if load_model == True:
            print(f"... Load Model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path+'/ckpt', map_location=device)
            self.actor.load_state_dict(checkpoint["actor"])
            self.target_actor.load_state_dict(checkpoint["actor"])
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.target_critic.load_state_dict(checkpoint["critic"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    # OU noise 기법에 따라 행동 결정
    def get_action(self, state, training=True):
        #  네트워크 모드 설정
        self.actor.train(training)

        action = self.actor(torch.FloatTensor(state).to(device)).cpu().detach().numpy()
        return action + self.OU.sample() if training else action

    # 리플레이 메모리에 데이터 추가 (상태, 행동, 보상, 다음 상태, 게임 종료 여부)
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        batch = random.sample(self.memory, batch_size)
        state      = np.stack([b[0] for b in batch], axis=0)
        action     = np.stack([b[1] for b in batch], axis=0)
        reward     = np.stack([b[2] for b in batch], axis=0)
        next_state = np.stack([b[3] for b in batch], axis=0)
        done       = np.stack([b[4] for b in batch], axis=0)

        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device),
                                                        [state, action, reward, next_state, done])

        # Critic 업데이트
        next_actions = self.target_actor(next_state)
        next_q = self.target_critic(next_state, next_actions)
        target_q = reward + (1 - done) * discount_factor * next_q
        q = self.critic(state, action)
        critic_loss = F.mse_loss(target_q, q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor 업데이트
        action_pred = self.actor(state)
        actor_loss = -self.critic(state, action_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    # 소프트 타겟 업데이트를 위한 함수
    def soft_update_target(self):
        for target_param, local_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    # 네트워크 모델 저장
    def save_model(self):
        print(f"... Save Model to {save_path}/ckpt ...")
        torch.save({
            "actor" : self.actor.state_dict(),
            "actor_optimizer" : self.actor_optimizer.state_dict(),
            "critic" : self.critic.state_dict(),
            "critic_optimizer" : self.critic_optimizer.state_dict(),
        }, save_path+'/ckpt')

    # 학습 기록
    def write_summray(self, score, actor_loss, critic_loss, total_step, step, ep):
        self.writer.add_scalar("run/score_ep", score, ep)
        self.writer.add_scalar("run/score_step", score, total_step)
        self.writer.add_scalar("run/step_ep", step, ep)
        self.writer.add_scalar("model/actor_loss", actor_loss, ep)
        self.writer.add_scalar("model/critic1_loss", critic_loss, ep)