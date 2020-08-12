# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:42:40 2020

@author: Woody
"""

from model import Model
from agent import Agent
from algorithm import DQN
from parl.utils import logger
import numpy as np
import gym


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)  # 预测动作，只选最优动作
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

LEARN_FREQ = 5 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 20000    # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再开启训练
BATCH_SIZE = 32   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.005 # 学习率
GAMMA = 0.99 # reward 的衰减因子，一般取 0.9 到 0.999 不等

# 环境选择cp或mc
cp = 0;
if cp:
    env = gym.make('CartPole-v0') 
else:
    env = gym.make('MountainCar-v0') 
    
action_dim = env.action_space.n  # CartPole-v0: 2 | MountainCar-v0: 3
obs_shape = env.observation_space.shape  # CartPole-v0: (4,) | MountainCar-v0: (2,)

# 根据parl框架构建agent
model = Model(act_dim=action_dim)
algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
agent = Agent(
    algorithm,
    obs_dim=obs_shape[0],
    act_dim=action_dim,
    e_greed=0.1,  # 有一定概率随机选取动作，探索
    e_greed_decrement=1e-6)  # 随着训练逐步收敛，探索的程度慢慢降低

# 加载模型
my_test = 0 # 测试自己的训练结果
if my_test:
    save_path = './dqn_model.ckpt'
else:
    if cp:
        save_path = './dqn_model_cp.ckpt'
    else:
        save_path = './dqn_model_mc.ckpt'
agent.restore(save_path)

# test part
eval_reward = evaluate(env, agent, render=True)  # render=True 查看显示效果
logger.info('e_greed:{}   test_reward:{}'.format(
    agent.e_greed, eval_reward))