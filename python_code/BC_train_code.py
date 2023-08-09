import torch
import numpy as np
import platform

from algorithms.BC import BCAgent
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel \
    import EngineConfigurationChannel
from mlagents.trainers.demo_loader import demo_to_buffer
from mlagents.trainers.buffer import BufferKey, ObservationKeyPrefix

load_model = False
train_mode = True

train_epoch = 500
test_step = 10000
discount_factor = 0.9

print_interval = 10
save_interval = 100

# Demonstration 경로
demo_path = "../demo/KartAgent.demo"

# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

game = "Kart"
os_name = platform.system()
if os_name == 'Windows':
    env_name = f"../envs/{game}/{game}"

# Main 함수 -> 전체적으로 BC 알고리즘을 진행
if __name__ == '__main__':
    # BCAgent 클래스를 agent로 정의
    agent = BCAgent()

    if train_mode:
        # Demonstration 정보 가져오기
        behavior_spec, demo_buffer = demo_to_buffer(demo_path, 1)
        print(demo_buffer._fields.keys())

        demo_to_tensor = lambda key: torch.FloatTensor(demo_buffer[key]).to(device)
        state = demo_to_tensor((ObservationKeyPrefix.OBSERVATION, 0))
        action = demo_to_tensor(BufferKey.CONTINUOUS_ACTION)
        reward = demo_to_tensor(BufferKey.ENVIRONMENT_REWARDS)
        done = demo_to_tensor(BufferKey.DONE)

        ret = reward.clone()
        for t in reversed(range(len(ret) - 1)):
            ret[t] += (1. - done[t]) * (discount_factor * ret[t + 1])

        # return이 0보다 큰 (state, action) pair만 학습에 사용.
        state, action = map(lambda x: x[ret > 0], [state, action])

        losses = []
        for epoch in range(1, train_epoch + 1):
            loss = agent.train_model(state, action)
            losses.append(loss)

            # 텐서 보드에 손실함수 값 기록
            if epoch % print_interval == 0:
                mean_loss = np.mean(losses)
                print(f"{epoch} Epoch / Loss: {mean_loss:.8f}")
                agent.write_summray(mean_loss, epoch)
                losses = []

            if epoch % save_interval == 0:
                agent.save_model()

    # 빌드 환경에서 Play 시작
    print("PLAY START")

    # 유니티 환경 경로 설정 (file_name)
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(file_name=env_name,
                           side_channels=[engine_configuration_channel])
    env.reset()

    # 유니티 브레인 설정
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    engine_configuration_channel.set_configuration_parameters(time_scale=1.0)
    dec, term = env.get_steps(behavior_name)

    # TEST 시작
    episode, score = 0, 0
    for step in range(test_step):
        state = dec.obs[0]
        action = agent.get_action(state, False)
        action_tuple = ActionTuple()
        action_tuple.add_continuous(action)
        env.set_actions(behavior_name, action_tuple)
        env.step()

        dec, term = env.get_steps(behavior_name)
        done = len(term.agent_id) > 0
        reward = term.reward if done else dec.reward
        next_state = term.obs[0] if done else dec.obs[0]
        score += reward[0]

        if done:
            episode += 1

            # 게임 진행 상황 출력
            print(f"{episode} Episode / Step: {step} / Score: {score:.2f} ")
            score = 0

    env.close()