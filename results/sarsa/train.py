# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 23:14:03 2020

@author: Woody
"""

from agent import SarsaAgent, QLearningAgent
from gridworld import CliffWalkingWapper, FrozenLakeWapper
import gym
import time
import argparse
import pdb

def run_episode(env, agent, render=False, sarsa=True):
    total_steps = 0 # 记录每个episode走了多少step
    total_reward = 0

    obs = env.reset() # 重置环境, 重新开一局（即开始新的一个episode）
    action = agent.sample(obs) # 根据算法选择一个动作

    while True:
        next_obs, reward, done, _ = env.step(action) # 与环境进行一个交互
        next_action = agent.sample(next_obs) # 根据算法选择一个动作
        
        if sarsa:
            # 训练 Sarsa 算法
            agent.learn(obs, action, reward, next_obs, next_action, done)
        else:
            # 训练 Q-learning算法
            agent.learn(obs, action, reward, next_obs, done)

        action = next_action
        obs = next_obs  # 存储上一个观察值
        total_reward += reward
        total_steps += 1 # 计算step数
        if render:
            env.render() #渲染新的一帧图形
        if done:
            break
    return total_reward, total_steps


def test_episode(env, agent):
    total_reward = 0
    obs = env.reset()
    while True:
        action = agent.predict(obs) # greedy
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        obs = next_obs
        time.sleep(0.5)
        env.render()
        if done:
            break
    return total_reward


def main():

    parser = argparse.ArgumentParser(description='基于表格型方法求解RL.')
    parser.add_argument('-a', '--agent', metavar='(ss|qn)', help='select agent type (sarsa or qlearning)', default='ss',
                    type=str)
    parser.add_argument('-e', '--env', metavar='(fl|cw)', help='select environment (frozenlake or cliffwalking)', default='fl', 
                    type=str)
    
    args = parser.parse_args()
    
    if args.env=='fl':
        # 环境1：FrozenLake, 可以配置冰面是否是滑的
        # 0 left, 1 down, 2 right, 3 up
        env = gym.make("FrozenLake-v0", is_slippery=False)
        env = FrozenLakeWapper(env)
    else:
        # 环境2：CliffWalking, 悬崖环境
        env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
        env = CliffWalkingWapper(env)

    # 环境3：自定义格子世界，可以配置地图, S为出发点Start, F为平地Floor, H为洞Hole, G为出口目标Goal
    # gridmap = [
    #         'SFFF',
    #         'FHFF',
    #         'FFFF',
    #         'HFGF' ]
    # env = GridWorld(gridmap)


# 创建一个agent实例，输入超参数
    if args.agent=='ss':
        agent = SarsaAgent(
    				obs_n=env.observation_space.n,
    				act_n=env.action_space.n,
    				learning_rate=0.1,
    				gamma=0.9,
    				e_greed=0.1)
        sarsa = True
    else:
        agent = QLearningAgent(
    				obs_n=env.observation_space.n,
    				act_n=env.action_space.n,
    				learning_rate=0.1,
    				gamma=0.9,
    				e_greed=0.1)    
        sarsa = False

    # 训练500个episode，打印每个episode的分数
    for episode in range(500):
        ep_reward, ep_steps = run_episode(env, agent, False, sarsa)
        print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps, ep_reward))

    # 全部训练结束，查看算法效果
    test_reward = test_episode(env, agent)
    print('test reward = %.1f' % (test_reward))
    


if __name__ == '__main__':
    main()