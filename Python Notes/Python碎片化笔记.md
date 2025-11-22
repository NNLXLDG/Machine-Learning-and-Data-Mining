# Python碎片化笔记

##  自动安装库函数
尝试导入指定的包，如果导入失败（包未安装），自动用pip安装；pypi or pkg：如果提供了PyPI名称就用它，否则用包名
```python
def ensure(pkg, pypi=None):
    try:
        __import__(pkg)  # 尝试导入包
    except Exception:  # 如果导入失败
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pypi or pkg])


ensure('sklearn','scikit-learn')
#包名为scikit-learn，但导入时用sklearn，所以安装时用PyPI名称，导入用短名称。
```

## sys库
sys 是 Python 标准库中的一个模块，提供了与 Python 解释器及其环境交互的功能。通过 sys 库，你可以访问与 Python 解释器相关的变量和函数，例如命令行参数、标准输入输出、程序退出等。


## subprocess库
subprocess 是 Python 标准库（无需额外安装）中专门用于 创建新进程、执行外部命令（如系统命令、其他程序）并与之交互 的模块。

它的核心价值是：让 Python 程序跳出自身的运行环境，去调用**操作系统级别的命令**或其他外部程序（比如 ls、cd、git，甚至是 python 脚本、exe 程序等），并能获取这些外部命令的输出、错误信息或执行状态。

在 subprocess 出现之前，Python 常用 os.system() 等函数调用外部命令，但 subprocess 提供了更灵活、更安全的控制能力（比如精确捕获输出、处理错误、设置超时等），因此现在官方推荐优先使用 subprocess。


主要函数有：
- subprocess.run(): 运行一个命令，等待其完成，并返回一个 CompletedProcess 实例，包含有关进程的信息。
- subprocess.Popen(): 启动一个新进程，并提供对其输入/输出/错误管道的访问。
- subprocess.check_call(): 运行一个命令，如果命令返回非零退出状态则引发 CalledProcessError。
- subprocess.check_output(): 运行一个命令，并返回其输出，如果命令返回非零退出状态则引发CalledProcessError。   
 

**subprocess.run()**
```python
import subprocess

# 基本用法
result = subprocess.run(['ls', '-l'])
# 使用列表传递参数避免歧义
# subprocess.run('ls -l', shell=True)   效果相同，但有安全风险


# 捕获输出
result = subprocess.run(['ls', '-l'], capture_output=True, text=True)

print(result.stdout)
print(result.returncode)
```
命令执行完成后，subprocess.run() 会返回一个 CompletedProcess 类型的对象，存储在 result 变量中。这个对象包含了命令执行的关键信息:

+ result.returncode：命令的退出码（整数）。
+ 0 表示命令执行成功（如 ls -l 成功列出内容）；
+ 非 0 表示失败（如 ls 不存在的文件 会返回非 0）。
+ result.stdout：命令的标准输出（stdout）内容。默认情况下为 None（因为命令的输出会直接打印到控制台，和手动执行 ls -l 一样）。
+ result.stderr：命令的错误输出（stderr）内容。默认情况下为 None（错误信息也会直接打印到控制台）。


**更多参数示例：**
```python
result = subprocess.run(
    ['python', '--version'],
    capture_output=True,
    text=True,
    check=True,  # 如果非零退出码则抛出异常
    timeout=30,  # 超时设置
    cwd='/path/to/dir'  # 工作目录
)
```
+ `capture_output=True`用于捕获命令的输出和错误信息（替代手动设置 stdout 和 stderr）。
等价于 `stdout=subprocess.PIPE, stderr=subprocess.PIPE`（PIPE 表示 “管道”，用于暂存输出）。效果：命令的标准输出（stdout，如正常结果） 会被存到 `result.stdout`，错误输出（stderr，如报错信息） 会被存到 `result.stderr`，而不是直接打印到控制台。
+ `text=True`用于将捕获的输出转为字符串类型（默认是字节流 bytes 类型）。
+ `result.stdout` 和 `result.stderr` 的内容会从原始的字节流（如 b'Python 3.9.7'）转换为字符串（如 'Python 3.9.7'），方便后续用字符串方法（如 split()、strip()）处理。
+ `timeout=30`用于设置命令执行的超时时间（单位：秒）。如果命令在指定时间内未完成，会抛出 `subprocess.TimeoutExpired` 异常，避免程序无限等待。
+ `cwd='/path/to/dir'`用于指定命令执行的工作目录。如果不指定，默认使用当前 Python 脚本所在目录。指定后，命令会在该目录下执行，影响相对路径的解析。

![alt text](image-21.png)



## tarfile库
tarfile 是 Python 标准库中的一个模块，用于创建、读取和解压缩 tar 归档文件。tar 文件是一种常见的文件打包格式，通常用于在类 Unix 系统中打包和分发文件。 tarfile 模块提供了方便的接口来处理这些归档文件，使得你可以轻松地创建和提取 tar 文件。




## rcParams设置
rcParams 是 matplotlib 中一个非常重要的概念，是 matplotlib 的运行时配置参数字典，用于控制和自定义 matplotlib 的几乎所有默认行为。

+ rc = runcommand（运行命令）或 runtime configuration（运行时配置）
+ arams = parameters（参数）

rcParams 允许你一次性设置全局的绘图样式，避免在每个图表中重复设置相同的参数。

**字体相关**
```python
plt.rcParams['font.family'] = 'sans-serif'    # 字体族
plt.rcParams['font.size'] = 12                # 字体大小
plt.rcParams['font.sans-serif'] = ['Arial']   # 无衬线字体优先顺序
```
**图形尺寸相关**
```python
plt.rcParams['figure.figsize'] = (8, 6)       # 图形默认尺寸
plt.rcParams['figure.dpi'] = 100              # 图形分辨率
```
**线条样式相关**
```python      
plt.rcParams['axes.grid'] = True              # 显示网格
plt.rcParams['axes.labelsize'] = 14           # 坐标轴标签大小
```

**设置专业报告样式**
```python
# 一次性设置所有样式
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'savefig.bbox': 'tight'
})

# 之后的所有图表都会应用这些设置
plt.plot([1, 2, 3, 4])
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.show()
```
**重置默认设置**
```python
# 重置所有参数到matplotlib默认值
plt.rcParams.update(plt.rcParamsDefault)
```



## serif和sans-serif字体
在字体分类中，serif（衬线字体）和 sans-serif（无衬线字体）是两种主要的字体风格。

- **Serif字体**：这种字体在字母的末端有小的装饰线条，称为“衬线”。衬线字体通常被认为更传统、更正式，适合用于印刷材料，如书籍和报纸。常见的serif字体包括Times New Roman、Georgia等。
- **Sans-serif字体**：这种字体没有衬线，字母的末端是平滑的。无衬线字体通常被认为更现代、更简洁，适合用于数字屏幕显示，如网页和应用程序界面。常见的sans-serif字体包括Arial、Helvetica等。

```python
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
```


DejaVu Sans是 英文字体（支持多语言字符），开源字体，跨平台兼容性好，常用于数据可视化和图形设计中。这种格式由Microsoft和Apple共同开发，具有跨平台使用的特点。
SimHei是中文黑体字体，常用于中文排版和设计中，具有良好的可读性和视觉效果。在matplotlib中，DejaVu Sans通常用于显示英文字符，而SimHei则用于显示中文字符。

```python
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei']
```

这样设置后，matplotlib会优先使用DejaVu Sans来显示英文字符，如果遇到中文字符，则会自动切换到SimHei字体进行显示，从而确保图表中的中英文字符都能正确显示。


## sys.executable
sys.executable 是 Python 解释器的绝对路径。它指向当前正在运行的 Python 解释器的可执行文件的位置。


```python
import sys
import subprocess


# 1. 确保使用相同的Python环境
print(f"当前Python: {sys.executable}")

# 2. 在当前Python环境中安装包，'-m'表示作为模块运行pip
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy'])

# 3. 运行其他Python脚本（使用相同环境）
subprocess.check_call([sys.executable, 'other_script.py'])

# 4. 检查Python版本
subprocess.check_call([sys.executable, '--version'])
```



## try...except Exception 语句
try...except 语句是 Python 中用于异常处理的结构。它允许你在代码中捕获和处理可能发生的错误，从而防止程序崩溃，并提供优雅的错误处理机制。
```python
try:
    # 可能出错的代码
    risky_operation()
except Exception:
    # 出错时的处理
    handle_error()
```


**异常值的层次结构**
在 Python 中，异常（Exceptions）是用于处理程序运行时错误的机制。Python 提供了一个层次结构来组织各种异常类型，使得开发者可以根据需要捕获和处理不同级别的错误。以下是 Python 异常层次结构的简要概述：
```
BaseException
 ├── KeyboardInterrupt    # Command+C
 ├── SystemExit          # sys.exit()
 └── Exception
      ├── ValueError     # 值错误
      ├── TypeError      # 类型错误
      ├── IOError        # I/O错误
      ├── ImportError    # 导入错误
      └── ...等等
```

**各种用法示例**
1. 捕获特定异常
```python
try:
    x = int("不是数字")
except ValueError as e:
    print(f"值错误: {e}")
```
2. 捕获多个特定异常
```python
try:
    file = open("不存在.txt")
    data = json.load(file)
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"文件或JSON错误: {e}")
```
3. 捕获所有异常（不推荐）
```python
try:
    risky_code()
except:  # 捕获所有异常，包括KeyboardInterrupt
    print("出错了")
```
4. 捕获所有Exception（推荐）
```python
try:
    risky_code()
except Exception as e:  # 只捕获Exception及其子类
    print(f"错误类型: {type(e).__name__}")
    print(f"错误信息: {e}")
```
5. 完整的异常处理结构
```python
try:
    # 尝试执行的代码
    result = 10 / int(input("输入数字: "))
    
except ValueError:
    # 处理值错误
    print("请输入有效数字!")
    
except ZeroDivisionError:
    # 处理除零错误  
    print("不能除以零!")
    
except Exception as e:
    # 处理其他所有异常
    print(f"未知错误: {e}")
    
else:
    # 如果没有异常发生
    print(f"结果是: {result}")
    
finally:
    # 无论是否异常都会执行
    print("程序执行完毕")
```


## networkx库
networkx 是一个用于创建、操作和研究复杂网络结构的 Python 库。它提供了丰富的数据结构和算法，用于处理图（Graph）、有向图（DiGraph）和多重图（MultiGraph）等各种类型的网络。networkx 广泛应用于社会网络分析、生物信息学、交通网络等领域。




