from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

unity_env = UnityEnvironment("C:\\PROJECTS\\ml-agents\\Project\\example_envs\\UnityEnvironment.exe")
env = UnityToGymWrapper(unity_env, uint8_visual=True, flatten_branched=True)

try:
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, info = env.step(action)

    ## === ##
        if terminated:         
            observation = env.reset()
            break

except KeyboardInterrupt:
    print("Run ended")
    env.close()
