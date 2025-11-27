# 深度学习 NLP（进入神经网络与 Transformer）

1. 序列模型（RNN 家族）

理解 Transformer 为什么能击败 LSTM，必须对 LSTM 有基本掌握。
	•	RNN、GRU、LSTM
	•	Attention 的引入（Bahdanau / Luong Attention）
	•	Seq2Seq
	•	Encoder–Decoder 框架

实践：
	•	用 PyTorch 实现 LSTM 文本分类
	•	用 Seq2Seq 搭建一个简单机器翻译模型

⸻

2. Transformer 机制（重点）

Transformer 是现代 NLP/Large Language Model 的核心模块，需要深入理解：
	•	Self-Attention
	•	Multi-Head Attention
	•	Position Embedding
	•	Residual + LayerNorm
	•	Masked LM vs Auto-Regressive LM
	•	Encoder vs Decoder vs Encoder-Decoder

必学论文：
	•	Attention is All You Need

实践：
	•	手写一个最小 Transformer Block
	•	训练一个小型 toy 语言模型（1–5M 参数）



