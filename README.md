# Max_ReinforcementLearning_sandbox
Dependencies: openAI gymnasium, numpy

Code creates an Q-learning agent in the gymnasium lunar lander v2 environment. The observation space is continous and is discretized by tile coding it (used library https://github.com/MeepMoop/tilecoding).
Each coded state is then hashed and its index is used as a lookup for the actions, thus the state-action pair is connected. 
The Q-learning algorithm is take from from "Artificial Intelligence A Modern Approach" by Stuart Russell and Peter Norvig.
Q-Learning-Agent, page 844, figure 21.8

![BÖRJE](https://user-images.githubusercontent.com/40268765/218102728-ce867708-3ce1-40e7-b3da-be9da417f4c0.gif)
