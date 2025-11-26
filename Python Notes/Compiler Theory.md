# Compiler Theory 编译原理

## 编译原理基础（Core Compiler Theory）
1. 编译的整体流程（Compiler Pipeline）
	•	编译与解释的区别、混合模式（Python 就是混合）
	•	前端 vs 后端 vs 优化器
	•	源码 → AST → IR → 代码生成

2. 词法分析（Lexical Analysis）
	•	正则表达式机制
	•	有限自动机 DFA/NFA
	•	Python 的 tokenize 模块如何工作

3. 语法分析（Parsing）
	•	上下文无关文法（CFG）
	•	LL / LR 语法
	•	递归下降解析器（Python 实现解析器常用的方式）
	•	Python 的 ast.parse 背后的流程

4. 抽象语法树（AST）
	•	AST 的作用 & why it matters in AI frameworks
	•	Python AST 的结构与节点类型
	•	修改 AST 的方法（如做一个 mini AST 转换器）

5. 中间表示（IR）
	•	SSA（静态单赋值形式）
	•	三地址码
	•	图表示 IR（常见于深度学习框架）

6. 代码生成（Code Generation）
	•	编译到汇编
	•	Python 的字节码（Bytecode）
	•	虚拟机结构（Python VM, stack-based VM）















