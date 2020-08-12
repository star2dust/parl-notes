# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 22:01:43 2020

@author: Woody
"""
import paddle.fluid as fluid
import parl
from parl import layers
# from parl.algorithms import PolicyGradient # 也可以直接从parl库中导入PolicyGradient算法，无需重复写算法

class PolicyGradient(parl.Algorithm):
    def __init__(self, model, lr=None):
        """ Policy Gradient algorithm
        
        Args:
            model (parl.Model): policy的前向网络.
            lr (float): 学习率.
        """

        self.model = model
        assert isinstance(lr, float)
        self.lr = lr

    def predict(self, obs):
        """ 使用policy model预测输出的动作概率
        """
        return self.model(obs)

    def learn(self, obs, action, reward):
        """ 用policy gradient 算法更新policy model
        """
        act_prob = self.model(obs)  # 获取输出动作概率
        # log_prob = layers.cross_entropy(act_prob, action) # 交叉熵
        log_prob = layers.reduce_sum(
            -1.0 * layers.log(act_prob) * layers.one_hot(
                action, act_prob.shape[1]),
            dim=1)
        cost = log_prob * reward
        cost = layers.reduce_mean(cost)

        optimizer = fluid.optimizer.Adam(self.lr)
        optimizer.minimize(cost)
        return cost