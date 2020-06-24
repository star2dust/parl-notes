# 从零实现强化学习入门小结

课程源于百度AI平台，飞桨学院，世界冠军带你从零实践强化学习课程（[点击查看](https://aistudio.baidu.com/aistudio/education/group/info/1335)）。这个小结我计划从代码和实践的角度来整理这五次课所学内容，直接切入主题，希望能做到干货满满。每一小节先讲概念，再介绍理论，最后讲解代码。

开始之前我假设大家都已经对强化学习有了最基本的认识，同时具有python编程基础，一定程度上掌握[paddlepaddle](https://www.paddlepaddle.org.cn/documentation/docs/zh/1.6/beginners_guide/quick_start_cn.html)和[numpy库](https://www.paddlepaddle.org.cn/tutorials/projectdetail/557543)的用法。如果完全是小白，可以看看班上大神的笔记，特别推荐[三岁学编程](https://blog.csdn.net/weixin_45623093/article/details/106799387)、[Mr.郑先生_](https://blog.csdn.net/zbp_12138/article/details/106800911)、[Tiny Tony](https://www.zhihu.com/people/tiny-tony-5)大佬们的总结。

## 强化学习——从尝试到创新

> 每个人都是过去经验的总和，你过去的经验造成了现在的你。

强化学习（Reinforcement learning，简称RL）是机器学习中的一个领域，强调如何基于环境而行动，以取得最大化的预期利益。核心思想是：智能体（agent）在环境（environment）中学习，根据环境的状态（state）或观测（observation），执行动作（action），并根据环境的反馈 （reward）来指导更好的动作。

<img src="figures/rl_basic.png" alt="rl_start" width="450"/>

作为机器学习三巨头之一，强化学习和监督学习以及非监督学习关系如下图。

<img src="figures/ml_big3.png" alt="rl_start" width="350"/>

**监督学习关注的是认知,而强化学习关注的是决策**。简单的说，前者学习经验，后者运用经验。同样都是一张小熊的图片，监督学习输出的是灰熊、熊猫还是老虎，强化学习输出的是装死、逃跑还是干一架。

<img src="figures/bear_how_to_do.png" alt="rl_start" width="450"/>

### 强化学习的分类和方法

<img src="figures/rl_categories.png" alt="rl_start" width="450"/>

通常强化学习关注的是无模型的问题，在未知的环境中进行探索学习。其方案有二：

- 基于价值的方法（Q表格）
  给每个状态都赋予一个价值的概念,来代表这个状态是好还是坏,这是一个相对的概念,让智能体往价值最高的方向行进。基于价值是确定性的。
- 基于策略的方法（Policy）
  制定出多个策略,策略里的每个动作都有一定的概率,并让每一条策略走到底,最后查看哪个策略是最优的。基于策略是随机性的。

<img src="figures/rl_methods.png" alt="rl_start" width="450"/>

### PRAL框架和GYM环境

- 强化学习经典**环境库GYM**将环境（Env）交互接口规范化为：重置环境reset()、交互step()、渲染render()。
- 强化学习**框架库PARL**将强化学习框架抽象为: Model、Algorithm、Agent三层，使得强化学习算法的实现和调试更方便和灵活。（前两者有神经网络才用得上）

直接上图展示Agent的训练（Train）和测试（Test）过程。

<img src="figures/gym.png" alt="rl_start" width="450"/>

安装依赖库

```shell
pip install paddlepaddle==1.6.3
pip install parl==1.3.1
pip install gym
```

