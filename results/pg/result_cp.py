# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 22:30:51 2020

@author: Woody
"""

from model import Model
from agent import Agent
from algorithm import PolicyGradient
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

LEARNING_RATE = 1e-3

# 环境选择
env = gym.make('CartPole-v0') 
    
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

# 根据parl框架构建agent
model = Model(act_dim=act_dim)
alg = PolicyGradient(model, lr=LEARNING_RATE)
agent = Agent(alg, obs_dim=obs_dim, act_dim=act_dim)

# 加载模型
save_path = './model_cp.ckpt'
agent.restore(save_path)

# test part
total_reward = evaluate(env, agent, render=True) # render=True 查看渲染效果，需要在本地运行，AIStudio无法显示
logger.info('Test reward: {}'.format(total_reward))