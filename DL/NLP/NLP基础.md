# 基础 NLP 能力（传统 NLP 必备）

目标：掌握 NLP 的基本概念、任务与传统方法，为后续深度学习打底。

1. NLP 基础知识
	•	Token / Type / Vocabulary
	•	语言模型（LM）的概念
	•	词法分析、句法分析
	•	N-Gram、统计语言模型
	•	常见 NLP 任务分类（分类、生成、结构化预测…）

推荐学习方式：
	•	《Speech and Language Processing（3rd）》前几章
	•	Stanford NLP 基础课程 CS224N（前两讲）

⸻

2. 文本特征表示（Embedding 演化线）

从最传统到现代嵌入表征，理解演进脉络非常重要：
	•	Bag-of-Words （BoW）
	•	TF-IDF
	•	Word2Vec（CBOW、SkipGram）
	•	GloVe
	•	FastText
	•	Contextual Embedding（ELMo、ULMFiT → Transformer）

建议实践：
	•	用 sklearn + gensim 实现 TF-IDF、Word2Vec
	•	小规模文本分类任务（如 IMDB）


