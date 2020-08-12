# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 23:28:28 2020

@author: Woody
"""
import parl
from parl import layers

# 神经网络模型
class Model(parl.Model):
    def __init__(self, act_dim):
        hid1_size = 128
        hid2_size = 128
        # 3层全连接网络
        self.fc1 = layers.fc(size=hid1_size, act='relu') # 隐藏层1 卷积
        self.fc2 = layers.fc(size=hid2_size, act='relu') # 隐藏层2 卷积
        self.fc3 = layers.fc(size=act_dim, act=None) # 输出层

    def value(self, obs):
        # 定义网络
        # 输入state，输出所有action对应的Q，[Q(s,a1), Q(s,a2), Q(s,a3)...]
        h1 = self.fc1(obs)
        h2 = self.fc2(h1)
        Q = self.fc3(h2)
        return Q