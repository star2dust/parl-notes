# A Tutorial to Learn RL from Examples

## Install Requirements

```shell
# or try: pip install -r requirements.txt
pip install paddlepaddle==1.6.3
pip install parl==1.3.1
pip install gym
pip install atari-py
pip install rlschool==0.3.1
```

## PARL Basics

[PARL](https://github.com/PaddlePaddle/PARL) is a flexible and high-efficient reinforcement learning framework. PARL aims to build an agent for training algorithms to perform complex tasks. The main abstractions introduced by PARL that are used to build an agent recursively are the following:

- **Model** is abstracted to construct the forward network which defines a policy network or critic network given state as input.

- **Algorithm** describes the mechanism to update parameters in Model and often contains at least one model.

- **Agent**, a data bridge between the environment and the algorithm, is responsible for data I/O with the outside environment and describes data preprocessing before feeding data into the training process.

## Sarsa and Q-learning

To demonstrate Sarsa and Q-learning in CliffWalking environment.
```python
# sarsa
cd .\tutorials\lesson2\sarsa
python .\train.py
# qlearing
cd .\tutorials\lesson2\q_learning
python .\train.py
```

Results are put here. The first one is Sarsa, the other one is Q-learning.

<img src="docs\figures\sarsa.gif" alt="sarsa" width="250" />

<img src="docs\figures\q_learning.gif" alt="qlearning" width="250" />