# Compiler Theory 编译原理

## 1 编译原理基础（Core Compiler Theory）

### 1.1 编译的整体流程（Compiler Pipeline）
	•	编译与解释的区别、混合模式（Python 就是混合）
	•	前端 vs 后端 vs 优化器
	•	源码 → AST → IR → 代码生成

### 1.2 词法分析（Lexical Analysis）
	•	正则表达式机制
	•	有限自动机 DFA/NFA
	•	Python 的 tokenize 模块如何工作

### 1.3 语法分析（Parsing）
	•	上下文无关文法（CFG）
	•	LL / LR 语法
	•	递归下降解析器（Python 实现解析器常用的方式）
	•	Python 的 ast.parse 背后的流程

### 1.4 抽象语法树（AST）
	•	AST 的作用 & why it matters in AI frameworks
	•	Python AST 的结构与节点类型
	•	修改 AST 的方法（如做一个 mini AST 转换器）

### 1.5 中间表示（IR）
	•	SSA（静态单赋值形式）
	•	三地址码
	•	图表示 IR（常见于深度学习框架）

### 1.6 代码生成（Code Generation）
	•	编译到汇编
	•	Python 的字节码（Bytecode）
	•	虚拟机结构（Python VM, stack-based VM）


## 2 结合 Python 的解释器与底层机制（Interpreter Internals）


### 2.1 Python 的执行模型
	•	CPython 的编译流程
源码 → Token → AST → Bytecode → 执行（栈机）
	•	PyPy、Cython、Numba 的差别

### 2.2 Python 字节码（Bytecode）
	•	dis 模块分析 bytecode
	•	栈式虚拟机（Stack-based VM）
	•	示例：将一段 Python 代码转成你的“手写字节码解释器”

### 2.3 全局解释器锁（GIL）
	•	为什么 Python 有 GIL
	•	GIL 与多线程的关系（与 AI 推理加速结合：PyTorch/TensorFlow 如何绕过 GIL）

### 2.4 JIT（Just-in-time 编译）
	•	PyPy 的 JIT
	•	Numba 的 LLVM JIT
	•	TorchScript 的 JIT
	•	比较静态编译 vs JIT 的优势

### 2.5 Python C API / 扩展
	•	你可以写自己的算子（Operator）
	•	深度学习框架中的 kernel（CUDA/C++）是怎么被 Python 调用的



## AI 相关的编译技术

这是现在做 AI 在”模型底层“、”推理加速“、”算子优化“ 都要用到的知识。

1. 计算图（Computation Graph）
	•	静态图（TensorFlow 1.x）
	•	动态图（PyTorch）
	•	PyTorch 的 TorchDynamo & FX tracer

2. 自动微分（Automatic Differentiation）
	•	forward-mode AD
	•	reverse-mode AD（现代深度学习框架主要使用）
	•	PyTorch 的 autograd 底层机制（Tape + Function graph）

3. 深度学习编译器（DL Compiler）

理解这一部分，你就会知道：
	•	为什么 TensorRT/FasterTransformer/TVM 能提升推理速度？
	•	为什么模型量化后跑得更快？
	•	ONNX 是什么？

（1）TVM 编译器
	•	Relay IR
	•	算子融合
	•	Auto-tuning

（2）XLA 编译器（JAX / TensorFlow）
	•	HLO 图
	•	HLO IR 优化 passes
	•	图融合（Fusion）
	•	编译到 GPU/TPU

（3）TorchInductor（PyTorch 2.x）
	•	AOT Autograd
	•	Triton kernel 生成
	•	通用算子融合
	•	GPU kernel 编译（Triton → PTX → CUDA）

4. 模型量化与编译
	•	Post-training quantization (PTQ)
	•	Quantization-aware training (QAT)
	•	INT8 / BF16 / FP8 在编译器中的处理机制
	•	ONNX Runtime / TensorRT 的量化优化







## 四、工程实践路线


以下是 建议配套的动手项目，每个都很实用。

Level A：理解编译过程
	1.	手写一个 Python 子集解释器
	•	词法 → 语法 → AST → 字节码 → 虚拟机执行
	2.	写个简单静态类型检查器（类似 mypy 子集）

Level B：深入 Python 底层
	3.	写一个 Python 字节码优化器
	4.	用 Cython/Numpy/C API 写一个高性能算子并测速度

Level C：进入 AI 编译器领域
	5.	用 PyTorch FX 做一个简单图优化器（算子融合）
	6.	重写一个 PyTorch 算子的 CUDA kernel
	7.	用 TVM 把一个 ResNet block 编译优化并测性能
	8.	写一个小的自动微分系统（backprop engine）

Level D：构建自己的“小型深度学习框架”
	9.	你可以写一个 mini-framework，包括：
	•	Tensor 类
	•	自动微分
	•	计算图构建
	•	优化器（SGD/Adam）
	•	简易 JIT






