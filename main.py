import gym
import qlearning
import concurrent.futures
import Evaluation_tools

env = gym.make("LunarLander-v2", render_mode='human')
env.action_space.seed(42)

tiles_per_dim = [5, 5, 5, 5, 5, 5, 2, 2]
lims = [(-1.5, 1.5), (-1.5, 1.5), (-5, 5), (-5, 5), (-3.14, 3.14), (-5, 5), (-0, 1), (-0, 1)]
tilings = 5

Qtable = qlearning.QTable(0.9, tiles_per_dim, lims, tilings, 4, resume_last=True)
#Qtable = qlearning.QTable(0.7, tiles_per_dim, lims, tilings, 4)
Qtable.change_name("Böörje Salmiing")
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

            if reward > 100:
                print(reward, terminated_i)
                #if reward >= 100:
                    #Qtable.save_Q_table_to_file()

            observation, info = env.reset()
except KeyboardInterrupt:
    Qtable.save_Q_table_to_file()
    evaluator.save_log()
    print("Run ended")
    env.close()
