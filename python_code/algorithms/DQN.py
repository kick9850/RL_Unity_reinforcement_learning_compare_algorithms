# 라이브러리 불러오기
import numpy as np
import random
import copy
import datetime
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import deque

action_size = 1

load_model = False
train_mode = True

batch_size = 32
mem_maxlen = 10000
discount_factor = 0.95
learning_rate = 1e-4

epsilon_eval = 0.05
epsilon_init = 1.0 if train_mode else epsilon_eval
epsilon_min = 0.1
explore_step = 5000 * 0.8
epsilon_delta = (epsilon_init - epsilon_min) / explore_step if train_mode else 0.


# 유니티 환경 경로
game = "MyKartNormal"

# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d")
save_path = f"./saved_models/{game}/DQN/{date_time}"
load_path = f"./saved_models/{game}/DQN/20230719"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(DQN, self).__init__(**kwargs)
        self.fc1 = torch.nn.Linear(48, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class DQNAgent:
    def __init__(self):
        self.network = DQN().to(device)
        self.target_network = copy.deepcopy(self.network)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=mem_maxlen)
        self.epsilon = epsilon_init
        self.writer = SummaryWriter(save_path, flush_secs=10)

        if load_model == True:
            print(f"... Load Model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path+'/ckpt', map_location=device)
            self.network.load_state_dict(checkpoint["network"])
            self.target_network.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    def get_action(self, state, training=True):
        self.network.train(training)
        epsilon = self.epsilon if training else epsilon_eval
        if epsilon > random.random():
            action = np.random.uniform(-1, 1, size=(state.shape[0], 1))
        else:
            q = self.network(torch.FloatTensor(state).to(device))
            action = q.data.cpu().numpy()
        return action

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

        eye = torch.eye(action_size).to(device)
        one_hot_action = eye[action.view(-1).long()]
        q = (self.network(state) * one_hot_action).sum(1, keepdims=True)

        with torch.no_grad():
            next_q = self.target_network(next_state)
            target_q = reward + next_q.max(1, keepdims=True).values * ((1 - done) * discount_factor)

        loss = F.smooth_l1_loss(q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(epsilon_min, self.epsilon - epsilon_delta)  # 수정된 부분
        return loss.item()

    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())

    def save_model(self):
        print(f"... Save Model to {save_path}/ckpt ...")
        torch.save({
            "network" : self.network.state_dict(),
            "optimizer" : self.optimizer.state_dict(),
        }, save_path+'/ckpt')

    def write_summray(self, score, loss, epsilon, total_step, step, ep):
        self.writer.add_scalar("run/score_ep", score, ep)
        self.writer.add_scalar("run/score_step", score, total_step)
        self.writer.add_scalar("run/step_ep", step, ep)
        self.writer.add_scalar("model/loss", loss, ep)
        self.writer.add_scalar("model/epsilon", epsilon, ep)
