import numpy as np
import datetime
import platform
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Behavioral Cloning을 위한 파라미터 값 세팅
state_size = 12 * 4
action_size = 1

load_model = False
train_mode = True

batch_size = 128
learning_rate = 3e-4

# 유니티 환경 경로
game = "Kart"
os_name = platform.system()
if os_name == 'Windows':
    env_name = f"../envs/{game}/{game}"

# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d")
save_path = f"./saved_models/{game}/BC/{date_time}"
load_path = f"./saved_models/{game}/BC/{date_time}"

# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Actor 클래스 -> Behavioral Cloning Actor 클래스 정의
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


# BCAgent 클래스 -> Behavioral Cloning 알고리즘을 위한 다양한 함수 정의
class BCAgent():
    def __init__(self):
        self.actor = Actor().to(device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.writer = SummaryWriter(save_path)

        if load_model == True:
            print(f"... Load Model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path + '/ckpt', map_location=device)
            self.actor.load_state_dict(checkpoint["actor"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    # 행동 결정
    def get_action(self, state, training=False):
        #  네트워크 모드 설정
        self.actor.train(training)
        action = self.actor(torch.FloatTensor(state).to(device)).cpu().detach().numpy()
        return action

    # 학습 수행
    def train_model(self, state, action):
        losses = []

        rand_idx = torch.randperm(len(state))
        for iter in range(int(np.ceil(len(state) / batch_size))):
            _state = state[rand_idx[iter * batch_size: (iter + 1) * batch_size]]
            _action = action[rand_idx[iter * batch_size: (iter + 1) * batch_size]]

            action_pred = self.actor(_state)
            loss = F.mse_loss(_action, action_pred).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        return np.mean(losses)

    # 네트워크 모델 저장
    def save_model(self):
        print(f"... Save Model to {save_path}/ckpt ...")
        torch.save({
            "actor": self.actor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, save_path + '/ckpt')

    # 학습 기록
    def write_summray(self, loss, epoch):
        self.writer.add_scalar("model/loss", loss, epoch)