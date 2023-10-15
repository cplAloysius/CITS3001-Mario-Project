# CITS3001-Mario-Project

In this project, we compare and analyse the performance of 2 agents in the [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) environment. The first agent is a Rule-Based agent and the second agent is a PPO agent using the [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) library.


Rule-Based Agent:
- `python mario_objects.py` to run the rule-based agent
  
- 2 windows will appear, one showing the game and another showing the game with what the agent detects in bounding boxes

PPO Agent:
- `python train.py` to train a model (you can train a new model or load a previously trained model)

- `python test.py` to test the trained model

- Models are stored in the 'train' folder
