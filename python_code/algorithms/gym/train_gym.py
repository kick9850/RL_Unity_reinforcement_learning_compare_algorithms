from mlagents_envs.environment import UnityEnvironment 
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from stable_baselines3.ddpg import DDPG
from sb3_contrib.trpo import TRPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from datetime import datetime


time = datetime.today().strftime("%Y%m%d%H%M%S") 
env_name = "/home/smai2/Downloads/RL_Unity/envs/Kart_linux_basic/Kart_linux"


unity_env = UnityEnvironment(file_name=env_name, seed=1, side_channels=[])
env = UnityToGymWrapper(unity_env, uint8_visual=False)
log_dir = 'unity_log/'

eval_callback = EvalCallback(env, best_model_save_path=log_dir+time,
                                 log_path=log_dir, eval_freq=10000,
                                 deterministic=False, render=True)

model = DDPG('MlpPolicy', env=env, verbose=1, tensorboard_log=log_dir)
model.learn(100000, callback=eval_callback, tb_log_name="DDPG") 