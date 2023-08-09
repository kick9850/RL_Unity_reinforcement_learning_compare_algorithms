import numpy as np
import platform
import torch
from algorithms.ppo import PPOAgent

from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel \
    import EngineConfigurationChannel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_model = False
train_mode = True

# 파라미터 값 세팅
run_ep = 1500 if train_mode else 0
test_ep = 100

print_interval = 10
save_interval = 100

# 유니티 환경 경로
game = "MyKartNormal"
os_name = platform.system()
if os_name == 'Windows':
    env_name = f"../envs/{game}/{game}"

# Main 함수 -> 전체적으로 PPO 알고리즘 진행
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

    # PPOAgent 클래스를 agent로 정의
    agent = PPOAgent(state_size=48, action_size=1)
    actor_losses, critic_losses, scores, episode, score = [], [], [], 0, 0

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
            action = agent.get_action(state)
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
                actor_loss, critic_loss = agent.train_model(state, action, reward, next_state, np.float64([done]))
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

            if done:
                episode += 1
                scores.append(score)
                score = 0

                if episode % print_interval == 0:
                    mean_score = np.mean(scores)
                    mean_actor_loss = np.mean(actor_losses) if len(actor_losses) > 0 else 0
                    mean_critic1_loss = np.mean(critic_losses) if len(critic_losses) > 0 else 0

                    # TensorBoard에 값을 추가합니다.
                    agent.write_summary(mean_score, mean_actor_loss, mean_critic1_loss, total_step, step, episode)

                    actor_losses, critic_losses, scores = [], [], []

                    print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " +
                          f"Actor loss: {mean_actor_loss:.4f} / Critic loss: {mean_critic1_loss:.4f}")
                if train_mode and episode % save_interval == 0:
                    agent.save_model()
                break
    env.close()
