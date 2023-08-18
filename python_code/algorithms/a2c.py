# 라이브러리 불러오기
import numpy as np
import datetime
import platform
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel \
    import EngineConfigurationChannel

# 파라미터 값 세팅
state_size = 12 * 4
action_size = 1

load_model = False
train_mode = True

discount_factor = 0.9
actor_lr = 1e-4
critic_lr = 1e-4

# 유니티 환경 경로
game = "MyKartNormal"
os_name = platform.system()
if os_name == 'Windows':
    env_name = f"../envs/{game}/{game}"


# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d")
save_path = f"./saved_models/{game}/A2C/{date_time}"
load_path = f"./saved_models/{game}/A2C/20230725"

# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# A2C 클래스 -> Actor Network, Critic Network 정의
# Actor 클래스
class Actor(torch.nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 128)
        self.mu = torch.nn.Linear(128, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return torch.tanh(self.mu(x))

# Critic 클래스
class Critic(torch.nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.fc1 = torch.nn.Linear(state_size, 128)
        self.q = torch.nn.Linear(128, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.q(x)


# A2CAgent 클래스 -> A2C 알고리즘을 위한 다양한 함수 정의
class A2CAgent:
    def __init__(self):
        self.actor = Actor().to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = Critic().to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.writer = SummaryWriter(save_path)

        if load_model == True:
            print(f"... best Score Load Model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path + '/ckpt', map_location=device)
            self.actor.load_state_dict(checkpoint["actor"])
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    # 정책을 통해 행동 결정
    def get_action(self, state, training=True):
        #  네트워크 모드 설정
        self.actor.train(training)

        action = self.actor(torch.FloatTensor(state).to(device)).cpu().detach().numpy()
        return action

    # 학습 수행
    def train_model(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).unsqueeze(0).to(device)
        reward = torch.tensor(reward).unsqueeze(0).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = done == 'True'

        # Critic 업데이트
        with torch.no_grad():
            next_value = self.critic(next_state)
            target_q = reward + (1 - done) * discount_factor * next_value
        q = self.critic(state)
        critic_loss = F.mse_loss(target_q, q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor 업데이트
        action_prob = self.actor(state)

        action_log_prob = torch.log(action_prob)
        print(action_log_prob)
        advantage = (next_value - q).detach()
        actor_loss = -(action_log_prob * advantage).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    # 네트워크 모델 저장
    def save_model(self):
        print(f"... best Score Save Model to {save_path}/ckpt ...")
        torch.save({
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }, save_path + '/ckpt')

    # 학습 기록
    def write_summary(self, score, actor_loss, critic_loss, ep):
        self.writer.add_scalar("run/score", score, ep)
        self.writer.add_scalar("model/actor_loss", actor_loss, ep)
        self.writer.add_scalar("model/critic_loss", critic_loss, ep)