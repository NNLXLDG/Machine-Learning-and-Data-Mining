# Python自整理笔记

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













