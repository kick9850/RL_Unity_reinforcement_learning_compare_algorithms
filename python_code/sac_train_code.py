import numpy as np
import platform

from algorithms.SAC import Agent
import datetime

from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel\
                             import EngineConfigurationChannel

load_model = False
train_mode = True

batch_size = 128

print_interval = 10
save_interval = 100

date_time = datetime.datetime.now().strftime("%Y%m%d")

# 유니티 환경 경로
game = "MyKartNormal"
os_name = platform.system()
if os_name == 'Windows':
    env_name = f"../envs/{game}/{game}"

run_ep = 1000 if train_mode else 0
test_ep = 100
train_start_step = 5000

# Main 함수 TD3
if __name__ == '__main__':
    # 유니티 환경 경로 설정 (file_name)
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_name, side_channels=[engine_configuration_channel])
    env.reset()

    # 유니티 브레인 설정
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=1.0)
    dec, term = env.get_steps(behavior_name)

    # TD3 클래스를 agent로 정의
    agent = Agent(directory=f'./saved_models/{game}/SAC/{date_time}/',
                  state_size=12 * 4,
                  action_size=1,
                  random_seed=0,
                  hidden_size=256,
                  action_prior="uniform", # "normal"
                  gamma=0.99,
                  lr_actor=5e-4,
                  lr_critic=5e-4,
                  buffer_size=int(1e6),
                  batch_size=256,
                  alpha=None,
                  tau=1e-2)

    actor_losses, critic1_losses, critic2_losses, scores, episode, score = [], [], [], [], 0, 0
    total_step = 0
    while episode < run_ep + test_ep:
        for step in range(10000):
            if episode == run_ep:
                if train_mode:
                    agent.save()
                print("TEST START")
                train_mode = False
                engine_configuration_channel.set_configuration_parameters(time_scale=1.0)

            state = dec.obs[0]
            action = agent.act(state)
            action_v = action.numpy()
            action_v = np.clip(action_v * 1, -1, 1)
            action_tuple = ActionTuple()
            action_tuple.add_continuous(action_v)
            env.set_actions(behavior_name, action_tuple)
            env.step()

            dec, term = env.get_steps(behavior_name)
            done = len(term.agent_id) > 0
            reward = term.reward if done else dec.reward
            next_state = term.obs[0] if done else dec.obs[0]
            score += reward[0]
            total_step += 1
            if train_mode:
                agent.memory.add(state, action, reward, next_state, done)

            if train_mode and total_step > max(batch_size, train_start_step):
                # 학습 수행
                experiences = agent.memory.sample()
                actor_loss, critic1_loss, critic2_loss = agent.learn(step, experiences, agent.GAMMA)
                actor_losses.append(actor_loss)
                critic1_losses.append(critic1_loss)
                critic2_losses.append(critic2_loss)

            if done:
                episode += 1
                scores.append(score)
                score = 0

                # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록
                if episode % print_interval == 0:
                    mean_score = np.mean(scores)
                    mean_actor_loss = np.mean(actor_losses)
                    mean_critic1_loss = np.mean(critic1_losses)
                    mean_critic2_loss = np.mean(critic2_losses)
                    agent.write_summary(mean_score, mean_actor_loss, mean_critic1_loss, mean_critic2_loss, total_step, step, episode)
                    actor_losses, critic1_losses, critic2_losses, scores = [], [], [], []

                    print(f"{episode} Episode / Step: {total_step} / Score: {mean_score:.2f} / " +\
                          f"Actor loss: {mean_actor_loss:.4f} / Critic1 loss: {mean_critic1_loss:.4f}" +\
                          f"/ Critic2 loss: {mean_critic2_loss:.4f}")

                # 네트워크 모델 저장
                if train_mode and episode % save_interval == 0:
                    agent.save()
                break
    env.close()