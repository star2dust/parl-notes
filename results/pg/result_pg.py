# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 13:51:53 2020

@author: Woody
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 22:30:51 2020

@author: Woody
"""

from model import Model
from agent import Agent
from algorithm import PolicyGradient

import gym
import numpy as np
import os

from parl.utils import logger


# 评估 agent, 跑 5 个episode，求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            obs = preprocess(obs) # from shape (210, 160, 3) to (100800,)
            action = agent.predict(obs) # 选取最优动作
            obs, reward, isOver, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if isOver:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


# Pong 图片预处理
def preprocess(image):
    """ 预处理 210x160x3 uint8 frame into 6400 (80x80) 1维 float vector """
    image = image[35:195] # 裁剪
    image = image[::2,::2,0] # 下采样，缩放2倍
    image[image == 144] = 0 # 擦除背景 (background type 1)
    image[image == 109] = 0 # 擦除背景 (background type 2)
    image[image != 0] = 1 # 转为灰度图，除了黑色外其他都是白色
    return image.astype(np.float).ravel()

LEARNING_RATE = 1e-3

# 环境选择
env = gym.make('Pong-v0')
obs_dim = 80 * 80
act_dim = env.action_space.n
logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

# 根据parl框架构建agent
model = Model(act_dim=act_dim)
alg = PolicyGradient(model, lr=LEARNING_RATE)
agent = Agent(alg, obs_dim=obs_dim, act_dim=act_dim)

# 加载模型
if os.path.exists('./model_pg_3000.ckpt'):
    agent.restore('./model_pg_3000.ckpt')

# test part
total_reward = evaluate(env, agent, render=True) # render=True 查看渲染效果，需要在本地运行，AIStudio无法显示
logger.info('Test reward: {}'.format(total_reward))