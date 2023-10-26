import gymnasium as gym
import qlearning
import Evaluation_tools

env = gym.make('Acrobot-v1', render_mode=None)
env.action_space.seed(42)

Qtable = qlearning.QTable(0.8, 0.01, 0.1, _action_size=3, _state_size= 1000000, _tile_coding = True, resume_last=False)
#Qtable = qlearning.QTable(0.7, tiles_per_dim, lims, tilings, 4)
Qtable.change_name("BöörjeSalmiing")
evaluator = Evaluation_tools.Evaluator()

observation, info = env.reset(seed=42)
action = env.action_space.sample()
print("Starting")
terminated_i = 0
try:
    while True:
        observation, reward, terminated, truncated, info = env.step(action)
        action = Qtable.update_Q(observation, reward, terminated)

        if terminated or truncated:
            terminated_i = terminated_i + 1
            evaluator.cumulative_reward(reward, terminated_i)

            if reward == 0:
                print(reward, terminated_i)
                #if reward >= 100:
                    #Qtable.save_Q_table_to_file()

            observation, info = env.reset()
except KeyboardInterrupt:
    Qtable.save_Q_table_to_file()
    evaluator.save_log()
    print("Run ended")
    env.close()

env_test = gym.make('Acrobot-v1', render_mode='human')
observation, info = env_test.reset(seed=42)
try:
    while True:
        observation, reward, terminated, truncated, info = env_test.step(action)
        action = Qtable.get_best_action(observation)

        if terminated or truncated:
            observation, info = env_test.reset()
except KeyboardInterrupt:
    print("Test ended")
    env_test.close()