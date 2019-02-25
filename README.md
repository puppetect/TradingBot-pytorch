
# TradingBot Demo

## 目的


给定某只股票某年的分钟级数据，通过强化学习训练机器得到最佳操盘模型，最后用同只股票其他年份的分钟级数据进行回测，考察其表现


## 原理
1. 数据处理（data.py），应用Ta-lib从原始分钟级数据（Open, High, Low, Close, Volume)提取出若干训练因子（RSI, SAR, WILLR, MACD等）
2. 决策模型（models.py)，经过处理的数据作为观察值state导入深度学习模型得出policy
3. 环境互动（environ.py)，将policy通过Agent（如epsilon-greedy或Probability selection）获得对应的动作action（持有或空仓），并和环境互动后得到下一分钟的观察值next_state、盈亏比例reward、回合完成的指令done、和其他信息info
4. 训练模型(train.py)，得到若干(next_state, reward, done, info）后，根据所选的强化学习类型（DQN或Actor-critic）计算loss并回溯优化模型参数，保存最佳参数并通过tensorboard监测模型表现



## 参考
1. Book [Deep Reinforcement Learning Hands-On](https://www.packtpub.com/big-data-and-business-intelligence/deep-reinforcement-learning-hands)
2. Repo [Deep-Reinforcement-Learning-Hands-On](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On)
