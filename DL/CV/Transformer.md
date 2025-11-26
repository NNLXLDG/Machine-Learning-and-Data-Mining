# Transformer in Computer Vision 
(计算机视觉中的Transformer)

## 一、Transformer基础

### Transformer概述
- **提出背景**：2017年，Google研究员在论文"Attention is all you need"中提出Transformer模型，使用Self-Attention结构取代了在NLP任务中常用的RNN网络结构
- **核心特点**：基于注意力的Transformer网络被广泛用于序列建模任务，包括语言建模和机器翻译
- **扩展方式**：通过增加隐藏层的维度或堆叠更多的Transformer块来提高性能
- **局限性**：需要大量计算，不适合硬件资源和电池有限的移动场景

### 序列建模的挑战
- **RNN的局限性**：
  - 序列处理难以并行化
  - 长序列依赖关系建模困难

### 替代方案比较
1. **CNN替代RNN**：
   - 优势：可以并行计算
   - 局限：高层滤波器考虑更长序列的能力有限
   
2. **Self-Attention替代RNN**：
   - 优势：可并行计算，考虑整个输入序列
   - 特点：$A_1, A_2, A_3, A_4$可以并行计算，每个$A_i$基于整个输入序列获得

### Self-Attention机制
- **核心概念**：
  - Query ($Q$)：用于匹配其他元素
  - Key ($K$)：被匹配的元素
  - Value ($V$)：要提取的信息
  
- **计算过程**：
  1. 计算Query和Key的点积：$A_{i,j} = Q_i \cdot K_j$
  2. 缩放点积：$A_{i,j} = \frac{Q_i \cdot K_j}{\sqrt{d}}$，其中$d$是$Q$和$K$的维度
  3. 应用Softmax：$A_{i,j} = \text{softmax}(A_{i,j})$
  4. 加权求和：$Z_i = \sum_j A_{i,j} V_j$

- **特点**：并行计算，适合GPU加速

### Multi-Head Self-Attention
- **结构**：
  - 将输入投影到多个"头"上
  - 每个头执行独立的Self-Attention计算
  - 拼接各头的结果并通过线性层输出

- **计算过程**：
  1. 输入X分别传递到h个不同的Self-Attention中，计算得到h个输出矩阵$Z_1$到$Z_h$
  2. 将h个输出矩阵拼接(Concat)
  3. 通过Linear层得到最终输出Z

- **优势**：允许模型在不同表示子空间中关注不同位置

### Positional Encoding
- **问题**：Self-Attention本身不包含位置信息
- **解决方案**：
  - 为每个位置添加唯一的位置向量$P_i$
  - 位置向量可以通过训练得到，也可以使用公式计算
  - 在Transformer中使用公式计算：$PE_{(pos,2i)} = \sin(pos/10000^{2i/d_{model}})$
  - $PE_{(pos,2i+1)} = \cos(pos/10000^{2i/d_{model}})$
  
- **实现**：将位置向量与词向量相加：$X' = X + PE$

### Transformer网络架构
- **整体结构**：
  - 左侧为Encoder block，右侧为Decoder block
  - 黄色圈中的部分为Multi-Head Attention，包含多个Self-Attention
  - Encoder block包含一个Multi-Head Attention
  - Decoder block包含两个Multi-Head Attention(其中一个是Masked)

- **核心组件**：
  1. Add & Norm层：
     - Add：残差连接(Residual Connection)，防止网络退化
     - Norm：Layer Normalization，对每一层的激活值进行归一化
  
  2. Feed Forward层：
     - 两层全连接网络
     - 第一层激活函数为Relu，第二层无激活函数
     - 公式：$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$

- **Encoder block**：
  - 接收输入矩阵$X(n×d)$，输出矩阵$O(n×d)$
  - 多个Encoder block叠加组成Encoder

- **Decoder block**：
  - 第一个Multi-Head Attention采用Masked操作，防止模型看到未来信息
  - 第二个Multi-Head Attention使用Encoder的编码信息矩阵计算K,V

## 二、Transformer计算机视觉应用

### ViT (Vision Transformer)
- **提出背景**：Google在2020年提出，直接将Transformer应用在图像分类任务
- **性能**：在ImageNet 1K上达到88.55%的分类准确率（在Google自家的JFT数据集上预训练后）
- **意义**：证明Transformer在计算机视觉领域有效且效果惊人

### ViT网络结构
- **三大模块**：
  1. Patch + Position的Embedding层
  2. Transformer Encoder
  3. MLP Head（分类层）

- **Patch处理**：
  - 将224×224的原始图像分成16×16的小patch
  - patch数量为$224^2/16^2=196$，排列为序列
  - 将每个patch线性映射至768维向量（$16×16×3=768$）
  - 最终token维度为[196, 768]

- **位置编码与分类标记**：
  - 添加位置编码（Add方式融合，不改变维度）
  - 添加class token（768维向量），通过concat方式融合
  - 最终token维度为[197, 768]

### Transformer Encoder分析
- **组成**：
  1. Layer Normalization (LN)：
     - 相比Batch Normalization，LN针对单个样本计算均值和方差
     - 避免BN中受mini-batch数据分布影响的问题
     - 不需要存储每个节点的均值和方差
  
  2. Multi-Head Attention：
     - 与Transformer中的结构相同
  
  3. MLP Block：
     - 全连接层 + GELU激活函数 + DropOut
     - 采用倒瓶颈结构：
       - 输入特征层经过全连接后，通道膨胀为原来的4倍
       - 后一次全连接层再恢复成原来的数目

### ViT的优势与挑战
- **优势**：
  - 全局感受野：能捕捉图像的全局依赖关系
  - 并行计算：适合现代GPU硬件
  - 可扩展性：通过增加模型大小和训练数据可进一步提升性能

- **挑战**：
  - 数据饥饿：需要大量数据训练才能达到最佳性能
  - 计算成本：相比CNN需要更多计算资源
  - 位置信息：需要显式添加位置编码，不如CNN的归纳偏置强大

### ViT的变体与应用
- **DeiT (Data-efficient Image Transformer)**：
  - 通过知识蒸馏减少对预训练数据的依赖
  
- **Swin Transformer**：
  - 引入层次结构和滑动窗口，提高计算效率
  - 适合各种计算机视觉任务
  
- **应用扩展**：
  - 图像分类
  - 目标检测
  - 语义分割
  - 图像生成
