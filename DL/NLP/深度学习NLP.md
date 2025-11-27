# 深度学习 NLP（进入神经网络与 Transformer）

1. 序列模型（RNN 家族）

理解 Transformer 为什么能击败 LSTM，必须对 LSTM 有基本掌握。
	•	RNN、GRU、LSTM
		- **是什么**：
			- RNN（循环神经网络）：一种专门处理序列数据的神经网络，具有“记忆”功能，能把上一步的信息传到下一步。
			- LSTM（长短期记忆网络）/ GRU（门控循环单元）：RNN 的改进版，引入了“门”机制（如遗忘门、输入门）来控制信息的保留与丢弃。
		- **解决什么问题**：
			- RNN 解决了传统神经网络无法处理变长序列（如不同长度的句子）的问题。
			- LSTM/GRU 解决了 RNN 的“梯度消失”问题，使其能记住更长距离的信息（比如文章开头提到的名字，结尾还能记得）。
	•	Attention 的引入（Bahdanau / Luong Attention）
		- **是什么**：一种让模型在解码时能“聚焦”到输入序列中特定部分的机制。就像人翻译句子时，翻译到哪个词，眼睛就会看原文对应的那个词。
		- **解决什么问题**：解决了传统 Seq2Seq 模型必须把整个句子压缩成一个固定长度向量的“瓶颈”问题，大幅提升了长句翻译的效果。
	•	Seq2Seq
		- **是什么**：序列到序列模型，由一个编码器（Encoder）和一个解码器（Decoder）组成。
		- **解决什么问题**：解决了输入和输出序列长度不一致的问题，是机器翻译、文本摘要、对话系统的基础框架。
	•	Encoder–Decoder 框架
		- **是什么**：一种通用的深度学习架构设计模式。Encoder 负责理解输入，Decoder 负责生成输出。
		- **解决什么问题**：实现了“理解”与“生成”的解耦，使得我们可以用不同的模型组合来处理不同的任务（如用 CNN 做 Encoder 处理图像，用 RNN 做 Decoder 生成描述，就是“看图说话”）。

实践：
	•	用 PyTorch 实现 LSTM 文本分类
	•	用 Seq2Seq 搭建一个简单机器翻译模型

⸻

2. Transformer 机制（重点）

Transformer 是现代 NLP/Large Language Model 的核心模块，需要深入理解：
	•	Self-Attention
		- **是什么**：自注意力机制，计算句子中每个词与其他所有词的关联程度。例如在“The animal didn't cross the street because it was too tired”中，Self-Attention 能算出“it”与“animal”关联度最高。
		- **解决什么问题**：让模型能够直接捕捉句子中任意两个词之间的关系，无论它们距离多远，解决了 RNN 难以并行计算和长距离依赖弱的问题。
	•	Multi-Head Attention
		- **是什么**：多头注意力，相当于搞了多个 Self-Attention 模块并行工作，每个“头”关注不同的特征（比如有的关注语法，有的关注语义）。
		- **解决什么问题**：增强了模型的表达能力，让模型能从多个不同的角度理解文本。
	•	Position Embedding
		- **是什么**：位置编码，给每个词加上一个表示其位置的向量。
		- **解决什么问题**：因为 Transformer 是并行处理的，不像 RNN 那样天然有顺序，所以必须手动注入位置信息，否则模型会把“我爱你”和“你爱我”当成一样的。
	•	Residual + LayerNorm
		- **是什么**：残差连接（跳连）和层归一化。
		- **解决什么问题**：这是训练深层神经网络的“稳定器”，防止网络过深导致无法训练（梯度消失/爆炸），保证模型能收敛。
	•	Masked LM vs Auto-Regressive LM
		- **是什么**：
			- Masked LM (如 BERT)：像完形填空一样，盖住中间一个词让模型猜，能同时看到上下文（双向）。
			- Auto-Regressive LM (如 GPT)：像成语接龙一样，根据上文预测下一个词（单向）。
		- **解决什么问题**：定义了模型“学习”的方式。Masked LM 擅长理解任务（分类、阅读理解）；Auto-Regressive LM 擅长生成任务（写文章、对话）。
	•	Encoder vs Decoder vs Encoder-Decoder
		- **是什么**：Transformer 的三种架构变体。
			- Encoder-only (BERT)：只用编码器，擅长理解。
			- Decoder-only (GPT)：只用解码器，擅长生成。
			- Encoder-Decoder (T5/BART)：全套都有，擅长翻译和摘要。
		- **解决什么问题**：针对不同的 NLP 任务类型，选择最合适的架构。

必学论文：
	•	Attention is All You Need

实践：
	•	手写一个最小 Transformer Block
	•	训练一个小型 toy 语言模型（1–5M 参数）
