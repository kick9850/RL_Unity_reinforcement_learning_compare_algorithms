# 라이브러리 불러오기
import numpy as np
import platform
import torch
from algorithms.DQN import DQNAgent

from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel\
                             import EngineConfigurationChannel

action_size = 1

load_model = False
train_mode = True

batch_size = 32

run_ep = 1000 if train_mode else 0
test_ep = 100
train_start_step = 5000
target_update_step = 1000

print_interval = 10
save_interval = 100

# 유니티 환경 경로
game = "MyKartNormal"
os_name = platform.system()
if os_name == 'Windows':
    env_name = f"../envs/{game}/{game}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_name,
                           side_channels=[engine_configuration_channel])
    env.reset()
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=1.0)
    dec, term = env.get_steps(behavior_name)
    agent = DQNAgent(step=1000000,
                     epsilon_eval=0.05,
                     epsilon_init=1.0,
                     epsilon_min=0.1,
                     explore_step=0.8,
                     batch_size=32,
                     mem_maxlen=10000,
                     discount_factor=0.95,
                     learning_rate=1e-4)

    losses, scores, episode, score = [], [], 0, 0
    total_step = 0
    while episode < run_ep + test_ep:
        for step in range(10000):
            if episode == run_ep:
                if train_mode:
                    agent.save_model()
                print("TEST START")
                train_mode = False
                engine_configuration_channel.set_configuration_parameters(time_scale=1.0)

            state = dec.obs[0]
            action = agent.get_action(state, train_mode)
            action_tuple = ActionTuple()
            action_tuple.add_continuous(action)
            env.set_actions(behavior_name, action_tuple)
            env.step()

            dec, term = env.get_steps(behavior_name)
            done = len(term.agent_id) > 0
            reward = term.reward if done else dec.reward
            next_state = term.obs[0] if done else dec.obs[0]
            score += reward[0]
            total_step += 1
            if train_mode:
                agent.append_sample(state[0], action[0], reward, next_state[0], [done])

            if train_mode and total_step > max(batch_size, train_start_step):
                loss = agent.train_model()
                losses.append(loss)
                if total_step % target_update_step == 0:
                    agent.update_target()

            if done:
                episode += 1
                scores.append(score)
                score = 0
                if episode % print_interval == 0:
                    mean_score = np.mean(scores)
                    mean_loss = np.mean(losses)
                    agent.write_summray(mean_score, mean_loss, agent.epsilon, total_step, step, episode)
                    losses, scores = [], []

                    print(f"{episode} Episode / Step: {total_step} / Score: {mean_score:.2f} / " + \
                          f"Loss: {mean_loss:.4f} / Epsilon: {agent.epsilon:.4f}")
                if train_mode and episode % save_interval == 0:
                    agent.save_model()
                break
    env.close()