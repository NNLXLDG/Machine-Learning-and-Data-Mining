# Bilingual Glossary — ML/DL/RL 术语汇总

## Machine Learning (机器学习)

| English | 中文 | 简洁解释 |
|---------|------|--------|
| **Supervised Learning** | 监督学习 | 利用标注数据（特征-标签对）进行模型训练 |
| **Unsupervised Learning** | 无监督学习 | 在无标签数据上进行聚类、降维等探索性学习 |
| **Semi-supervised Learning** | 半监督学习 | 结合少量标注数据和大量无标签数据进行学习 |
| **Reinforcement Learning** | 强化学习 | 通过交互环境和奖励信号进行学习，追求累积收益最大化 |
| **Feature Engineering** | 特征工程 | 从原始数据中提取、选择、构造有效特征的过程 |
| **Overfitting** | 过拟合 | 模型过度拟合训练数据，泛化性能下降 |
| **Underfitting** | 欠拟合 | 模型学习不足，训练和测试性能都差 |
| **Regularization** | 正则化 | 通过 L1/L2 惩罚等限制模型复杂度，防止过拟合 |
| **Cross-Validation** | 交叉验证 | 将数据分段轮流用于训练和验证，评估模型泛化能力 |
| **Gradient Descent** | 梯度下降 | 沿着损失函数梯度反方向迭代更新参数的优化算法 |
| **Backpropagation** | 反向传播 | 从输出层逐层计算梯度并传递回输入层的算法 |
| **Batch Normalization** | 批归一化 | 对每一层的输入进行归一化，加速训练和增强稳定性 |
| **Dropout** | 随机丢弃 | 训练时随机关闭部分神经元，防止共同适应和过拟合 |
| **Activation Function** | 激活函数 | 向神经网络引入非线性的函数（如 ReLU、Sigmoid、Tanh） |
| **Hyperparameter** | 超参数 | 人工设置的参数（学习率、隐层数、批大小等），区别于模型学习的参数 |
| **Learning Rate** | 学习率 | 梯度下降中参数更新的步长，影响收敛速度和稳定性 |
| **Epoch** | 轮次 | 使用全部训练数据进行一遍完整训练 |
| **Batch Size** | 批大小 | 每次梯度更新使用的样本数量 |
| **Loss Function** | 损失函数 | 衡量预测值与真实值差异的函数（如交叉熵、MSE） |
| **Confusion Matrix** | 混淆矩阵 | 展示分类模型真正例、假正例、真反例、假反例数量的矩阵 |
| **Precision** | 精准率 | 预测为正类的样本中实际为正类的比例 |
| **Recall / Sensitivity** | 召回率/灵敏度 | 实际正类中被正确预测的比例 |
| **F1 Score** | F1 分数 | 精准率与召回率的调和平均数 |
| **ROC Curve** | ROC 曲线 | 以假正例率为 x 轴、真正例率为 y 轴的曲线，评估分类器性能 |
| **AUC** | 曲线下面积 | ROC 曲线下的面积，值越高分类器越优秀 |

## Deep Learning (深度学习)

| English | 中文 | 简洁解释 |
|---------|------|--------|
| **Neural Network** | 神经网络 | 由输入层、隐层、输出层组成的连接结构，模拟生物神经系统 |
| **Convolutional Neural Network (CNN)** | 卷积神经网络 | 通过卷积操作提取空间特征，广泛用于图像处理 |
| **Convolutional Layer** | 卷积层 | 使用卷积核对输入进行卷积操作，提取局部特征 |
| **Pooling Layer** | 池化层 | 对卷积输出进行下采样（如最大池化），减少参数和计算量 |
| **Recurrent Neural Network (RNN)** | 循环神经网络 | 具有反馈连接的网络，能处理序列数据和时间依赖关系 |
| **Long Short-Term Memory (LSTM)** | 长短期记忆网络 | 改进的 RNN，通过门机制解决梯度消失问题，适合长序列 |
| **Gated Recurrent Unit (GRU)** | 门控循环单元 | LSTM 的简化版本，参数更少，计算更高效 |
| **Attention Mechanism** | 注意力机制 | 动态分配权重关注输入的重要部分，提升模型性能 |
| **Transformer** | Transformer 架构 | 基于自注意力的架构，抛弃循环结构，更适合并行计算和长距离依赖 |
| **Self-Attention** | 自注意力 | 序列内部元素之间的注意力计算，无需外部上下文 |
| **Embedding** | 嵌入 | 将离散的类别（如单词）映射到连续向量空间 |
| **Word2Vec** | Word2Vec | 将单词映射为向量的技术，保留语义信息（Skip-gram、CBOW） |
| **Generative Adversarial Network (GAN)** | 生成对抗网络 | 生成器与判别器博弈产生高质量样本的架构 |
| **Encoder** | 编码器 | 将输入压缩为低维表示的网络部分 |
| **Decoder** | 解码器 | 将低维表示还原为高维输出的网络部分 |
| **Autoencoder** | 自编码器 | 编码器-解码器架构，用于无监督学习和降维 |
| **Variational Autoencoder (VAE)** | 变分自编码器 | 在自编码器基础上加入概率模型，可生成新样本 |
| **Transfer Learning** | 迁移学习 | 利用预训练模型在新任务上微调，减少所需数据量 |
| **Fine-tuning** | 微调 | 在预训练模型基础上用新数据继续训练，适应新任务 |
| **Knowledge Distillation** | 知识蒸馏 | 用大模型的知识指导小模型学习，压缩模型规模 |
| **Data Augmentation** | 数据增强 | 通过变换原有样本生成新样本，增加数据多样性 |
| **Batch Normalization** | 批归一化 | 对每层输入进行标准化，加速收敛和提高稳定性 |
| **Layer Normalization** | 层归一化 | 对单个样本的特征维度进行归一化，与批大小无关 |
| **Weight Initialization** | 权重初始化 | 初始化网络参数的策略（如Xavier、He初始化），影响收敛速度 |
| **Vanishing Gradient** | 梯度消失 | 反向传播中梯度逐层衰减至接近零，深层难以更新 |
| **Exploding Gradient** | 梯度爆炸 | 反向传播中梯度逐层增长至过大，导致不稳定 |
| **Gradient Clipping** | 梯度裁剪 | 限制梯度范数防止爆炸，常用于 RNN 训练 |
| **Momentum** | 动量 | 优化器中引入之前更新方向的惯性，加速收敛 |
| **Adam Optimizer** | Adam 优化器 | 结合动量和适应性学习率的优化算法，应用广泛 |
| **Inception Network** | Inception 网络 | 使用多尺度平行卷积的高效深度网络 |
| **ResNet** | 残差网络 | 通过跳跃连接解决深层网络退化问题，支持极深模型 |
| **DenseNet** | 稠密连接网络 | 每层与前面所有层连接，提高特征重用和梯度流动 |
| **VGG** | VGG 网络 | 简单深度卷积网络，用堆叠小卷积核替代大核 |
| **AlexNet** | AlexNet | 首个深度学习架构在 ImageNet 上取得突破性成果 |
| **MobileNet** | MobileNet | 轻量级网络，通过深度可分离卷积减少参数和计算 |
| **EfficientNet** | EfficientNet | 通过复合系数平衡模型深度、宽度和分辨率 |
| **BERT** | BERT | 双向编码表示，预训练语言模型，推动 NLP 发展 |
| **GPT** | GPT | 生成式预训练模型，通过因果语言建模训练 |
| **Object Detection** | 目标检测 | 定位和识别图像中多个对象的任务 |
| **YOLO** | YOLO | 实时目标检测算法，一次预测输出所有目标 |
| **R-CNN** | R-CNN | 两阶段目标检测，先提议再分类，精度高 |
| **Semantic Segmentation** | 语义分割 | 对图像每个像素进行分类，实现像素级分类 |
| **Instance Segmentation** | 实例分割 | 区分不同个体对象的分割，结合检测和分割 |
| **Panoptic Segmentation** | 全景分割 | 同时进行语义和实例分割，包括背景和前景 |
| **U-Net** | U-Net | 编码-解码架构，跳跃连接，广泛用于医学图像分割 |
| **Dilated Convolution** | 扩张卷积 | 增大感受野而不增加参数的卷积变体 |
| **Depthwise Separable Convolution** | 深度可分离卷积 | 分离通道卷积和点卷积，大幅减少计算 |
| **Quantization** | 量化 | 将浮点权重转为整数或低比特，压缩模型大小 |
| **Pruning** | 剪枝 | 移除不重要的权重或通道，减少模型复杂度 |

## Reinforcement Learning (强化学习)

| English | 中文 | 简洁解释 |
|---------|------|--------|
| **Agent** | 智能体 | 与环境交互的学习主体，采取行动并获得反馈 |
| **Environment** | 环境 | 智能体交互的外部系统，提供状态和奖励 |
| **State** | 状态 | 当前时刻环境的配置或描述 |
| **Action** | 行动 | 智能体在某状态下可执行的选择 |
| **Reward** | 奖励 | 环境对智能体行为的即时反馈信号 |
| **Policy** | 策略 | 将状态映射到行动的决策规则，可为确定性或随机性 |
| **Value Function** | 价值函数 | 估计状态或状态-行动对的长期累积奖励 |
| **State Value** | 状态价值 | 从某状态开始的期望累积奖励 |
| **Action Value / Q-Value** | 行动价值/Q值 | 在某状态执行某行动后的期望累积奖励 |
| **Temporal Difference (TD)** | 时序差分 | 利用相邻时步价值估计差异进行学习的方法 |
| **Q-Learning** | Q-学习 | 无模型强化学习，通过 Q 值迭代更新学习最优策略 |
| **Deep Q-Network (DQN)** | 深度 Q 网络 | 用神经网络近似 Q 函数的深度强化学习算法 |
| **Policy Gradient** | 策略梯度 | 直接参数化策略，沿梯度方向优化累积奖励 |
| **Actor-Critic** | 演员-评论家 | 结合价值和策略的方法，演员更新策略，评论家评估价值 |
| **Proximal Policy Optimization (PPO)** | 近端策略优化 | 稳定高效的策略梯度算法，限制策略更新幅度 |
| **Trust Region Policy Optimization (TRPO)** | 信任域策略优化 | 保证单调策略改进的策略梯度方法 |
| **Model-free RL** | 无模型强化学习 | 不依赖环境动态模型的学习，直接从经验学习 |
| **Model-based RL** | 有模型强化学习 | 学习环境模型（转移概率和奖励），用于规划 |
| **Off-policy Learning** | 离策略学习 | 用一个策略生成经验，用另一策略学习（如 Q-learning） |
| **On-policy Learning** | 在策略学习 | 用同一策略生成经验和学习（如 SARSA、策略梯度） |
| **Experience Replay** | 经验回放 | 存储过往经验，随机采样训练，打破时序相关性 |
| **Epsilon-Greedy** | ε-贪心 | 以概率 ε 进行探索，以概率 1-ε 利用最优行动 |
| **Exploration vs Exploitation** | 探索与利用 | 平衡发现新好策略（探索）和重复已知好策略（利用） |
| **Discount Factor** | 折扣因子 | 权衡即时奖励和未来奖励的参数，γ ∈ [0,1] |
| **Return** | 回报 | 从某时刻开始的累积折扣奖励 |
| **Trajectory** | 轨迹 | 智能体与环境的一次完整交互序列 |
| **Markov Decision Process (MDP)** | 马尔可夫决策过程 | 形式化强化学习问题的框架，满足马尔可夫性质 |
| **Bellman Equation** | 贝尔曼方程 | 递归定义价值函数的方程，强化学习的理论基础 |
| **Multi-armed Bandit** | 多臂老虎机 | 简化强化学习模型，只有一个状态但多个行动 |
| **Inverse Reinforcement Learning** | 逆强化学习 | 从最优行为推断潜在奖励函数 |
| **Meta Reinforcement Learning** | 元强化学习 | 学习如何快速学习新任务的强化学习 |
| **Imitation Learning** | 行为克隆 | 从专家演示中学习策略，无需明确奖励 |
| **Hierarchical RL** | 分层强化学习 | 多层次政策，高层制定目标，低层执行基础行动 |
| **Multi-agent RL** | 多智能体强化学习 | 多个互动智能体的强化学习，引入合作与竞争 |
| **Curiosity-driven Learning** | 好奇心驱动学习 | 通过内在动机（新奇度）探索环境，减少对外部奖励的依赖 |

## Common Metrics & Concepts (常见评估指标与概念)

| English | 中文 | 简洁解释 |
|---------|------|--------|
| **Mean Squared Error (MSE)** | 均方误差 | 预测值与真实值平方差的平均，回归任务常用 |
| **Root Mean Squared Error (RMSE)** | 均方根误差 | MSE 的平方根，与目标变量量纲一致 |
| **Mean Absolute Error (MAE)** | 平均绝对误差 | 绝对误差的平均，对异常值更鲁棒 |
| **Cross Entropy Loss** | 交叉熵损失 | 衡量两个概率分布差异的函数，分类任务常用 |
| **Accuracy** | 准确率 | 正确预测样本占总样本的比例 |
| **Macro Average** | 宏平均 | 计算各类别指标的算术平均，适合类别不平衡 |
| **Weighted Average** | 加权平均 | 按类别样本数加权平均指标 |
| **Micro Average** | 微平均 | 汇总所有样本后计算全局指标 |
| **Baseline** | 基线 | 简单模型的性能作为参考，评估复杂模型改进幅度 |
| **Ensemble Learning** | 集成学习 | 结合多个模型预测，通过投票或加权提升性能 |
| **Bagging** | 自助采样聚合 | 并行训练多个模型，每个用自助采样得到的数据 |
| **Boosting** | 梯度提升 | 顺序训练多个模型，后续模型关注前面的错误 |
| **Random Forest** | 随机森林 | 多个决策树的集成，树之间随机选择特征和样本 |
| **Gradient Boosting** | 梯度提升 | 顺序添加决策树，每棵树拟合前面模型的残差 |
| **XGBoost** | XGBoost | 优化的梯度提升框架，速度快、性能强 |
| **LightGBM** | LightGBM | 轻量级梯度提升机，基于叶增长，训练高效 |
| **CatBoost** | CatBoost | 处理分类特征的梯度提升框架，性能稳定 |
| **Support Vector Machine (SVM)** | 支持向量机 | 寻找最大间隔超平面的分类算法，适合高维数据 |
| **K-Nearest Neighbors (KNN)** | K 最近邻 | 基于邻近样本的非参数分类方法，简单有效 |
| **Decision Tree** | 决策树 | 树形结构的分类模型，易解释 |
| **Naive Bayes** | 朴素贝叶斯 | 基于贝叶斯定理的概率分类器，假设特征独立 |
| **Logistic Regression** | 逻辑回归 | 基于 sigmoid 函数的线性分类模型 |
| **Principal Component Analysis (PCA)** | 主成分分析 | 线性降维方法，找数据方差最大的方向 |
| **t-SNE** | t-SNE | 非线性降维可视化方法，保留局部相似性 |
| **K-Means** | K-均值聚类 | 无监督聚类算法，将样本分为 K 个簇 |
| **DBSCAN** | 密度聚类 | 基于密度的聚类，适合任意形状簇 |
| **Hierarchical Clustering** | 层次聚类 | 自下而上或自上而下逐步聚类，生成树形结构 |
| **Dimensionality Reduction** | 降维 | 减少特征数量，保留重要信息，加速训练 |
| **Anomaly Detection** | 异常检测 | 识别与正常数据显著不同的样本 |
| **Time Series Forecasting** | 时间序列预测 | 根据历史数据预测未来值，如股价、天气预测 |
| **Natural Language Processing (NLP)** | 自然语言处理 | 计算机处理和理解人类语言的技术 |
| **Computer Vision (CV)** | 计算机视觉 | 使计算机能理解图像和视频内容的技术 |
| **Sequence-to-Sequence (Seq2Seq)** | 序列到序列 | 将可变长输入序列映射到可变长输出序列的模型 |
| **Machine Translation** | 机器翻译 | 自动将一种语言翻译为另一种语言 |
| **Named Entity Recognition (NER)** | 命名实体识别 | 识别文本中人名、地名、机构等实体 |
| **Sentiment Analysis** | 情感分析 | 判断文本表达的情感倾向（正面/负面/中立） |
| **Recommendation System** | 推荐系统 | 根据用户行为和偏好推荐项目的系统 |
| **Collaborative Filtering** | 协同过滤 | 基于用户-项目交互矩阵的推荐方法 |
| **Content-based Filtering** | 基于内容的推荐 | 根据项目特征和用户偏好进行推荐 |

