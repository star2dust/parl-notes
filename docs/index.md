# PARL框架从零入门强化学习

[TOC]

## 写在前面

> 不要重复造轮子，学会使用轮子。

本文源于百度AI平台飞桨学院《[世界冠军带你从零实践强化学习](https://aistudio.baidu.com/aistudio/education/group/info/1335)》课程的总结，感谢科科老师这几天精彩的讲解。本文旨在提供给读者PARL框架的使用方法，并从模型的理解和代码的构建角度来整理五次课所学内容，不求详尽但求简洁明了。我认为强化学习中对算法每一个概念的理解很重要，你可以不懂公式的推导，但是只要你理解了算法框图中的每一个步骤，那你就能够灵活的应用PARL框架去解决自己的问题。

开始之前我假设大家都已经对强化学习有了最基本的认识，同时具有python编程基础，一定程度上掌握[paddlepaddle](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.6/beginners_guide/quick_start_cn.html)和[numpy库](https://www.paddlepaddle.org.cn/tutorials/projectdetail/557543)的用法。如果完全是小白，理论部分可以看看大神的笔记，推荐[Mr.郑先生_](https://blog.csdn.net/zbp_12138/article/details/106800911)、[Tiny Tony](https://www.zhihu.com/people/tiny-tony-5/posts)、[hhhsy](https://www.zhihu.com/people/hhhsy-9/posts)、[叶强](https://www.zhihu.com/people/qqiang00/posts)大佬们的总结。

## 强化学习——从尝试到创造

> 每个人都是过去经验的总和，你过去的经验造成了现在的你。

### 初识强化学习

强化学习（Reinforcement learning，简称RL）是机器学习中的一个领域，强调如何基于环境而行动，以取得最大化的预期利益。核心思想是：智能体（agent）在环境（environment）中学习，根据环境的状态（state）或观测（observation），执行动作（action），并根据环境的反馈 （reward）来指导更好的动作。

<img src="figures/rl_basic.png" alt="rl_basic" width="450"/>

作为机器学习三巨头之一，强化学习和监督学习以及非监督学习关系如下图。

<img src="figures/ml_big3.png" alt="ml_big3" width="350"/>

**监督学习关注的是认知，而强化学习关注的是决策**。简单的说，前者学习经验，后者运用经验。同样都是一张小熊的图片，监督学习输出的是灰熊、熊猫还是老虎，强化学习输出的是装死、逃跑还是干一架。

<img src="figures/bear_how_to_do.png" alt="bear_how_to_do" width="450"/>

### 强化学习的分类和方法

<img src="figures/rl_categories.png" alt="rl_categories" width="450"/>

强化学习主要关注的是无模型的问题，在未知的环境中进行探索学习。其方案有二：

- 基于价值的方法（Q表格）
  给每个状态都赋予一个价值的概念,来代表这个状态是好还是坏,这是一个相对的概念,让智能体往价值最高的方向行进。基于价值是确定性的。
- 基于策略的方法（Policy）
  制定出多个策略,策略里的每个动作都有一定的概率,并让每一条策略走到底,最后查看哪个策略是最优的。基于策略是随机性的。

<img src="figures/rl_methods.png" alt="rl_methods" width="450"/>

### PRAL框架和GYM环境

- 强化学习经典**环境库GYM**将环境（Env）交互接口规范化为：重置环境reset()、交互step()、渲染render()。
- 强化学习**框架库PARL**将强化学习框架抽象为: Model、Algorithm、Agent三层，使得强化学习算法的实现和调试更方便和灵活。（前两者有神经网络才用得上）

Agent的训练（Train）和测试（Test）过程直接上图展示如下。

<img src="figures/gym.png" alt="gym" width="450"/>

本文所需全部依赖库代码如下，paddlepaddle默认使用CPU版本，可自行选用GPU版本，这里不再赘述。

```shell
# 可以直接 pip install -r requirements.txt
pip install paddlepaddle==1.6.3
pip install parl==1.3.1
pip install gym
pip install atari-py
pip install rlschool==0.3.1
```

## 基于表格型方法求解RL

> 生活中遇到问题可以查看生活手册，强化学习亦然。

### 序列决策的经典表达

某一状态信息包含了所有相关的历史，只要当前状态可知，所有的历史信息都不再需要，当前状态就可以决定未来，则认为该状态具有**马尔科夫性**。**马尔可夫决策过程**（MDP）是序列决策的数学模型，它是一个无记忆的随机过程，可以用一个元组<S,P>表示，其中S是有限数量的状态集，P是状态转移概率矩阵。

强化学习中我们引入奖励R和动作A来描述环境，构成MDP五元组<S,A,P,R,$\gamma$>，其中P函数表示环境的随机性，R函数其实是P函数的一部分，表示获得的收益，$\gamma$是衰减因子以适当的减少对未来收益的考虑。

<img src="figures/bear_tree.png" alt="bear_tree" width="450"/>

强化学习主要解决的是Model-free的情况，即P函数和R函数都未知的情况。这时我们用价值V代表某一状态的好坏，用Q函数来代表某个状态下哪个动作更好，即状态动作价值。

<img src="figures/model_free.png" alt="model_free" width="450"/>

现实世界中，奖励R往往是延迟的，所以一般会从当前时间点开始，对后续可能得到的收益累加，以此来计算当前的价值。但是有时候目光不要放得太长远，**对远一些的东西当作近视看不见就好**。适当地引入一个衰减因子$\gamma$，再去计算未来的总收益，$\gamma$的值在0-1之间，时间点越久远，对当前的影响也就越小。

### 状态动作价值的求解

假设人走在树林里，先看到树上有熊爪后看到熊，接着就看到熊发怒了，经过很多次之后，原来要见到熊才瑟瑟发抖的，后来只要见到树上有熊爪就会有晕眩和害怕的感觉。也就是说，在不断地训练之后，下一个状态的价值可以不断地强化、影响上一个状态的价值。

这样的迭代状态价值的强化方式被称为时序差分（Temporal Difference）。单步求解Q函数，用$Q(S_{t+1},A_{t+1})$来近似$G_{t+1}$，以迭代的方式简化数学公式，最终使得$Q(S_t,A_t)$逼近目标值$G_t$。这里的目标值Target就是前面提到的未来收益的累加。

<img src="figures/q_td.png" alt="q_td" width="400"/>

### Sarsa和Qlearning

**Sarsa**全称是state-action-reward-state’-action’，目的是学习特定的state下，特定action的价值Q，最终建立和优化一个Q表格，以state为行，action为列，根据与环境交互得到的reward来更新Q表格，更新公式即为上面的迭代公式。Sarsa在训练中为了更好的探索环境，采用ε-greedy方式（如下图）来训练，有一定概率随机选择动作输出。

<img src="figures/e_greedy.png" alt="e_greedy" width="350"/>

**Q-learning**也是采用Q表格的方式存储Q值，探索部分与Sarsa是一样的，采用ε-greedy方式增加探索。

- Q-learning跟Sarsa不一样的地方是更新Q表格的方式，即learn()函数。
- Sarsa是on-policy，先做出动作再learn，Q-learning是off-policy，learn时无需获取下一步动作

二者更新Q表格的方式分别为：

<img src="figures/qlearning_sarsa_learn.png" alt="qlearning_sarsa_learn" width="400"/>

二者算法对比如下图所示，有三处不同点。

<img src="figures/qlearning_sarsa.png" alt="qlearning_sarsa" width="500"/>

on-policy优化的是目标策略，用下一步一定会执行的动作来优化Q表格；off-policy实际上有两种不同的策略，期望得到的目标策略和大胆探索的行为策略，在目标策略的基础上用行为策略获得更多的经验。

<img src="figures/on_off_policy.png" alt="on_off_policy" width="450"/>

### 代码构建与演示

Sarsa Agent构建

```python
class SarsaAgent(object):
    def __init__(self,
                 obs_n,
                 act_n,
                 learning_rate=0.01,
                 gamma=0.9,
                 e_greed=0.1):
        self.act_n = act_n  # 动作维度，有几个动作可选
        self.lr = learning_rate  # 学习率
        self.gamma = gamma  # reward的衰减率
        self.epsilon = e_greed  # 按一定概率随机选动作
        self.Q = np.zeros((obs_n, act_n))

    # 根据输入观察值，采样输出的动作值，带探索
    def sample(self, obs):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):  #根据table的Q值选动作
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)  #有一定概率随机探索选取一个动作
        return action

    # 根据输入观察值，预测输出的动作值
    def predict(self, obs):
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]  # maxQ可能对应多个action
        action = np.random.choice(action_list)
        return action

    # 学习方法，也就是更新Q-table的方法
    def learn(self, obs, action, reward, next_obs, next_action, done):
        """ on-policy
            obs: 交互前的obs, s_t
            action: 本次交互选择的action, a_t
            reward: 本次动作获得的奖励r
            next_obs: 本次交互后的obs, s_t+1
            next_action: 根据当前Q表格, 针对next_obs会选择的动作, a_t+1
            done: episode是否结束
        """
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward  # 没有下一个状态了
        else:
            target_Q = reward + self.gamma * self.Q[next_obs,
                                                    next_action]  # Sarsa
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)  # 修正q
```

Qlearning Agent构建

```python
class QLearningAgent(object):
    def __init__(self,
                 obs_n,
                 act_n,
                 learning_rate=0.01,
                 gamma=0.9,
                 e_greed=0.1):
        self.act_n = act_n  # 动作维度，有几个动作可选
        self.lr = learning_rate  # 学习率
        self.gamma = gamma  # reward的衰减率
        self.epsilon = e_greed  # 按一定概率随机选动作
        self.Q = np.zeros((obs_n, act_n))

    # 根据输入观察值，采样输出的动作值，带探索
    def sample(self, obs):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):  #根据table的Q值选动作
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)  #有一定概率随机探索选取一个动作
        return action

    # 根据输入观察值，预测输出的动作值
    def predict(self, obs):
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]  # maxQ可能对应多个action
        action = np.random.choice(action_list)
        return action

    # 学习方法，也就是更新Q-table的方法
    def learn(self, obs, action, reward, next_obs, done):
        """ off-policy
            obs: 交互前的obs, s_t
            action: 本次交互选择的action, a_t
            reward: 本次动作获得的奖励r
            next_obs: 本次交互后的obs, s_t+1
            done: episode是否结束
        """
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward  # 没有下一个状态了
        else:
            target_Q = reward + self.gamma * np.max(
                self.Q[next_obs, :])  # Q-learning
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)  # 修正q
```

大家可以用命令行运行以下代码尝试一下。

```shell
# sarsa 演示
cd .\tutorials\lesson2\sarsa
python .\train.py
# qlearing 演示
cd .\tutorials\lesson2\q_learning
python .\train.py
```



