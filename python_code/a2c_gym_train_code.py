import numpy as np
import platform
import gym
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper  # UnityToGymWrapper 파일을 import 해주세요.

from algorithms.a2c import A2CAgent

from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

load_model = False
train_mode = True
print_interval = 10
save_interval = 100

game = "MyKartNormal"
os_name = platform.system()
if os_name == 'Windows':
    env_name = f"../envs/{game}/{game}"

run_ep = 500 if train_mode else 0
test_ep = 100

# Main 함수
if __name__ == '__main__':
    # Unity 환경 설정
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_name, side_channels=[engine_configuration_channel])
    env.reset()

    # UnityToGymWrapper를 사용하여 Unity 환경을 Gym 환경으로 래핑
    gym_env = UnityToGymWrapper(env)

    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=1.0)
    dec, term = env.get_steps(behavior_name)

    agent = A2CAgent()
    actor_losses, critic_losses, scores, episode, score = [], [], [], 0, 0

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
            step += 1
            if train_mode:
                actor_loss, critic_loss = agent.train_model(state, action, reward, next_state, done)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

            if done:
                episode += 1
                scores.append(score)
                score = 0

                if episode % print_interval == 0:
                    mean_score = np.mean(scores)
                    mean_actor_loss = np.mean(actor_losses) if len(actor_losses) > 0 else 0
                    mean_critic_loss = np.mean(critic_losses) if len(critic_losses) > 0 else 0
                    agent.write_summary(mean_score, mean_actor_loss, mean_critic_loss, episode)
                    actor_losses, critic_losses, scores = [], [], []

                    print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " +
                          f"Actor loss: {mean_actor_loss:.4f} / Critic loss: {mean_critic_loss:.4f}")

                if train_mode and episode % save_interval == 0:
                    agent.save_model()
                break
    env.close()
