from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from stable_baselines3.a2c import A2C
from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC
env_name = "/home/smai2/Downloads/RL_Unity/envs/Kart_normal/Kart_linux"


unity_env = UnityEnvironment(file_name=env_name, seed=1, side_channels=[])
env = UnityToGymWrapper(unity_env, uint8_visual=False)
model = SAC.load("/home/smai2/Downloads/RL_Unity/unity_log/20230804122213/sac_best_model.zip")
num_episodes = 10
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        env.render()
    print(f"Episode {episode + 1}: Total reward = {episode_reward}")
env.close()
