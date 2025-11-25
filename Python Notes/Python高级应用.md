# Python Advanced Techniques for AI Deployment

> 本文档面向刚接触 AI 模型部署的学生，旨在通过深入 Python 高级特性，帮助你写出高效、可维护的 AI 工程代码。

---

## 1. Python 的执行模型

### 1.1 Python 的运行机制

**CPython 解释器如何执行代码**

你知道吗？Python 代码在执行前会经过以下步骤：

1. **源代码** → **字节码**（编译）
2. **字节码** → **机器码**（解释执行）

```python
import dis

def hello():
    x = 5
    y = 10
    return x + y

# 反编译查看字节码
dis.dis(hello)
```

这很重要，因为：当你用 PyTorch DataLoader 时，每个 worker 进程都需要独立执行字节码，这就涉及到下面的 GIL 问题。

**GIL（全局解释器锁）的本质**

CPython 为了简化内存管理，用一把全局锁（GIL）来保护内存。这意味着：
- **多线程不能真正并行执行 Python 字节码**（只能轮流执行）
- **多进程才能真正并行**（每个进程有独立的 GIL）

```python
import threading
import time

def cpu_intensive():
    """CPU 密集型任务"""
    total = 0
    for i in range(10**8):
        total += i
    return total

# ❌ 多线程版本 - 反而更慢（因为 GIL）
start = time.time()
t1 = threading.Thread(target=cpu_intensive)
t2 = threading.Thread(target=cpu_intensive)
t1.start()
t2.start()
t1.join()
t2.join()
print(f"多线程耗时: {time.time() - start:.2f}s")  # 约 8秒

# ✅ 多进程版本 - 真正并行
from multiprocessing import Process
start = time.time()
p1 = Process(target=cpu_intensive)
p2 = Process(target=cpu_intensive)
p1.start()
p2.start()
p1.join()
p2.join()
print(f"多进程耗时: {time.time() - start:.2f}s")  # 约 4秒
```

**为什么 AI 代码中常常用多进程而不是多线程？**

因为：
- **数据加载** = CPU 密集型（数据增强、预处理都是 CPU 操作）
- **多线程被 GIL 阻挡**，无法真正并行
- **PyTorch DataLoader** 默认用 `num_workers > 0` 就是多进程的原因

**💡 对 AI 的重要性**：理解这一点，你就知道为什么设置 `DataLoader(num_workers=4)` 能真正加速数据加载。

---

### 1.2 深入理解变量、作用域与内存管理

**Python 的引用计数机制**

Python 用引用计数来管理内存：
- 每个对象都有一个 `refcount`（引用计数）
- 当 refcount = 0 时，垃圾回收器立即释放内存

```python
import sys

x = [1, 2, 3]
print(sys.getrefcount(x))  # 至少 2（x 的引用 + getrefcount 参数的引用）

y = x  # 引用计数 +1
print(sys.getrefcount(x))  # 现在是 3

del y  # 引用计数 -1
print(sys.getrefcount(x))  # 回到 2
```

**局部变量 vs 闭包变量**

```python
def outer():
    captured = [1, 2, 3]  # 闭包中被捕获的变量
    
    def inner():
        print(captured)  # 可访问外层变量
        local_var = 99   # 仅在 inner 内有效
    
    return inner

func = outer()
# captured 仍然在内存中，因为被 inner 引用（闭包）
```

这在 AI 代码中很关键：
```python
def create_data_loader(data_list):
    """工厂函数"""
    def load_batch():
        # data_list 被捕获在闭包中，不会被释放
        return data_list[:10]
    return load_batch

loader = create_data_loader([1,2,3,...,1000000])
# 即使数据集很大，只要 loader 被引用，data_list 就一直在内存中
```

**id() 与对象驻留机制**

```python
# 小整数驻留（CPython 优化）
a = 256
b = 256
print(a is b)  # True（同一对象）

c = 257
d = 257
print(c is d)  # False（不同对象）

# 字符串驻留
s1 = "hello_world"
s2 = "hello_world"
print(s1 is s2)  # True（驻留）
```

这看似无关，但在处理数据增强时，如果你不小心多次复制了数据，引用计数和驻留机制可能导致内存泄漏。

**💡 对 AI 的重要性**：

避免数据增强/加载时出现隐性复制 → 降低显存/内存消耗。例如：

```python
# ❌ 错误方式：多次复制数据
def bad_augment(image):
    img1 = image.copy()      # 多余复制
    img2 = img1.copy()       # 又复制了
    return img2

# ✅ 正确方式：原地操作或少复制
import numpy as np
def good_augment(image):
    # 直接修改，减少内存开销
    image = image.astype(np.float32) / 255.0
    return image
```


---

## 2. 迭代器与生成器（数据加载的基础）

### 2.1 迭代器协议

任何实现了 `__iter__()` 和 `__next__()` 的对象都是**迭代器**。

```python
# 自定义迭代器
class CountUp:
    def __init__(self, max):
        self.max = max
        self.current = 0
    
    def __iter__(self):
        return self  # 返回自己
    
    def __next__(self):
        if self.current < self.max:
            self.current += 1
            return self.current
        else:
            raise StopIteration  # 迭代结束

# 使用
for num in CountUp(3):
    print(num)  # 输出 1, 2, 3
```

**可迭代对象 vs 迭代器对象**

- **可迭代对象**：实现了 `__iter__()` 的对象（如列表、字符串、集合）
- **迭代器对象**：实现了 `__iter__()` 和 `__next__()` 的对象

```python
# 列表是可迭代的，但不是迭代器
lst = [1, 2, 3]
print(hasattr(lst, '__iter__'))  # True
print(hasattr(lst, '__next__'))  # False

# iter() 将可迭代对象转为迭代器
iterator = iter(lst)
print(hasattr(iterator, '__next__'))  # True
print(next(iterator))  # 1
print(next(iterator))  # 2
```

### 2.2 生成器基础：从列表到流

**yield 的来历和本质**

在 Python 2.2 之前，如果你要遍历大数据集，必须一次性加载到内存中。这对 AI 来说是灾难性的——想象 ImageNet（数百GB），你无法全部加载。

生成器通过 `yield` 提供了解决方案：**函数可以在中途暂停，保存状态，等待下一次唤醒**。这种"延迟计算"思想是现代数据处理的基础。

具体工作原理：
- 首次 `next()`：函数执行至 `yield`，返回值，然后暂停
- 再次 `next()`：从暂停处继续执行，直到下一个 `yield`
- 函数返回或抛出 `StopIteration`：迭代结束

这使得生成器是**有状态的迭代器**——它记住上次执行到哪里。

```python
def simple_generator():
    print("开始")
    yield 1
    print("继续")
    yield 2
    print("结束")
    yield 3

gen = simple_generator()
print(next(gen))  # "开始", 返回 1
print(next(gen))  # "继续", 返回 2
print(next(gen))  # "结束", 返回 3
```

**为什么生成器对 AI 至关重要？**

对比两种方式：
```python
# 方式 1：列表 - 一次性创建
big_list = [x**2 for x in range(10**6)]  # 立即占用 ~40MB

# 方式 2：生成器 - 按需计算
big_gen = (x**2 for x in range(10**6))  # 只占用几 KB
```

在训练中，生成器允许你：
- 处理超大数据集（只在内存中放一小批）
- 实时数据增强（每个 epoch 看到不同增强）
- 无限数据流（Online Learning）

PyTorch 的 `DataLoader` 和 `IterableDataset` 都基于生成器思想。

### 2.3 面向 AI 的生成器应用场景

**自定义数据集加载器**

在 PyTorch 中，继承 `IterableDataset` 并实现 `__iter__()` 方法。它返回一个生成器，比 `Map-style Dataset` 更灵活——可以动态决定生成什么数据，无需预先定义大小。

```python
class CustomDataset:
    def __iter__(self):
        for i in range(1000):
            yield (f"sample_{i}", i)
```

优势：
- **内存高效**：不需要预先加载所有数据
- **灵活性**：根据需要动态生成（数据增强、采样）
- **无限流**：可以无限 yield

**流式读取大标注文件（如 COCO annotations）**

COCO annotations 可能数GB 大。用列表一次加载会 OOM。但生成器可逐行读取，**无论文件多大，内存中同时只有一个批次**：

```python
def read_annotations_stream(filepath, batch_size=32):
    import json
    batch = []
    with open(filepath) as f:
        for line in f:
            batch.append(json.loads(line))
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch
```

这比 `json.load(open(file))` 一次性加载节省 95% 以上内存。

**数据增强与在线学习**

有些场景需要实时生成训练样本（对抗训练、在线学习）。生成器完美适应：

```python
def online_augment_generator(base_samples):
    while True:  # 无限循环
        for sample in base_samples:
            augmented = apply_random_augmentation(sample)
            yield augmented
```

这让模型每个 epoch 看到不同增强版本，提升泛化。PyTorch DataLoader + num_workers 就是在后台多进程运行这样的生成器。

**💡 为什么生成器是 AI 数据处理的核心**

现代框架（PyTorch、TensorFlow）的数据加载机制都围绕生成器设计：
1. **内存效率**：无需一次性加载
2. **CPU 预加载**：后台生成下一批，GPU 训练当前批，CPU-GPU 并行
3. **灵活性**：支持动态增强、采样、在线学习
4. **可扩展**：处理 TB 级数据集

**💡 对 AI 的重要性**：

大型数据集 & 数据流处理的核心技能。例如在图像分类中：

```python
# PyTorch Dataset 的核心就是迭代器
import torch.utils.data as data

class CustomImageDataset(data.Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 这是一个 "延迟加载" 模式
        # 只在需要时加载图像，而不是全部预加载
        image = load_image(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]
```

---

## 3. 装饰器：函数的增强工厂

装饰器是 Python 最强大的特性之一，但也是最容易被误用的。在 AI 工程中，装饰器用于记录训练日志、自动重试、性能监控等。掌握它，能让你的代码更简洁、可维护。

### 3.1 装饰器的核心思想

**来历：函数作为一等对象**

Python 中函数是一等对象（First-class object），意味着函数可以像数据一样被赋值、传递、返回。这是装饰器的基础：

```python
def greet(name):
    return f"Hello, {name}!"

say_hello = greet          # 赋值
result = apply_func(greet) # 作为参数传递
```

**闭包与 *args / **kwargs**

装饰器的技术基础是**闭包**（closure）——一个函数"记住"它外层作用域的变量。另外，`*args` 和 `**kwargs` 允许装饰器接受任意参数的函数：

```python
def multiplier(factor):        # 返回一个"记住" factor 的函数
    def multiply(x):
        return x * factor
    return multiply

times_3 = multiplier(3)
print(times_3(10))  # 30
```

`*args` 接收任意位置参数，`**kwargs` 接收任意关键字参数。这让装饰器能适用于任何函数签名。

**无参数装饰器：最简单的形式**

装饰器本质上是一个函数，接受另一个函数，返回一个增强版的函数：

```python
def timer_decorator(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} 耗时: {time.time() - start:.4f}s")
        return result
    return wrapper

@timer_decorator
def train_epoch():
    import time
    time.sleep(1)
    return "完成"

train_epoch()  # 打印耗时，然后返回结果
```

`@timer_decorator` 这个语法糖等价于 `train_epoch = timer_decorator(train_epoch)`。

**有参数装饰器：多层嵌套**

有时你需要给装饰器传参。这需要多一层函数嵌套：

```python
def repeat_decorator(times):     # 第一层：装饰器工厂
    def decorator(func):         # 第二层：装饰器
        def wrapper(*args, **kwargs):  # 第三层：包装器
            for _ in range(times):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat_decorator(times=3)
def predict():
    print("预测中...")

predict()  # 打印三遍"预测中..."
```

理解这三层嵌套很关键——它允许你自定义装饰器的行为。

### 3.2 装饰器的工程实践

**functools.wraps：保持元信息**

当你装饰一个函数时，被包装函数的 `__name__`、`__doc__` 等元信息会丢失，变成 `wrapper`。这在调试时很困扰。`functools.wraps` 解决了这个问题：

```python
from functools import wraps

def good_decorator(func):
    @wraps(func)  # 复制原函数的元信息
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@good_decorator
def important_function():
    """这是重要函数"""
    pass

print(important_function.__name__)  # 保留原名
print(important_function.__doc__)   # 保留文档
```

**缓存装饰器：避免重复计算**

如果某个函数在相同输入下总是返回相同结果，可以缓存结果：

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_function(n):
    import time
    time.sleep(1)  # 模拟昂贵计算
    return n * n

result1 = expensive_function(5)  # 耗时 1 秒，计算并缓存
result2 = expensive_function(5)  # 立即返回（从缓存）
```

在 AI 推理中，这很有用——避免重复推理相同输入。

### 3.3 AI 工程中的装饰器应用

**训练过程的计时与监控**

在深度学习中，你需要监控各部分的耗时（加载数据、前向传播、反向传播等）。装饰器可以自动化这个过程，而无需在函数内部散布计时代码：

```python
def training_timer(func):
    from functools import wraps
    import time
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[{func.__name__}] 耗时: {elapsed:.3f}s")
        return result
    return wrapper

@training_timer
def load_batch(dataloader, batch_idx):
    # 加载数据逻辑
    return batch

@training_timer
def train_step(model, batch):
    # 训练逻辑
    return loss
```

这让你可以轻松添加/移除监控，而无需修改函数本身。

**自动重试装饰器：容错能力**

网络请求、数据加载等操作可能失败。装饰器可以自动重试：

```python
def retry(max_attempts=3, delay=1):
    def decorator(func):
        from functools import wraps
        import time
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    print(f"Attempt {attempt} failed, retrying in {delay}s...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=2)
def download_dataset(url):
    # 下载逻辑，可能失败
    import requests
    return requests.get(url).content
```

**日志记录装饰器**

自动记录函数调用、参数和返回值，对调试很有帮助：

```python
def log_execution(func):
    from functools import wraps
    import logging
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        logging.info(f"{func.__name__} returned {type(result)}")
        return result
    return wrapper

@log_execution
def predict(model, image):
    return model(image)
```

**💡 装饰器的最大优势**

装饰器让你在不修改原函数代码的情况下，为其添加新功能。在大型 AI 项目中，这意味着：
- 代码复用：一次定义，到处使用
- 关注点分离：分离业务逻辑和监控/日志
- 易维护：修改装饰器时，所有使用它的函数自动更新

---

## 4. 并行与并发：加速数据处理

**训练计时器 decorator**

```python
from functools import wraps
import time

def training_timer(func):
    """记录训练过程中各部分的耗时"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"【{func.__name__}】开始执行...")
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"【{func.__name__}】完成！耗时 {elapsed:.2f}s")
        return result
    return wrapper

@training_timer
def train_epoch():
    import time
    time.sleep(0.5)
    return "epoch完成"

train_epoch()
# 【train_epoch】开始执行...
# 【train_epoch】完成！耗时 0.50s
```

**自动日志记录 decorator**

```python
import logging
from functools import wraps

def auto_logger(func):
    """自动记录函数的输入和输出"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"调用 {func.__name__}，参数: {args}, {kwargs}")
        try:
            result = func(*args, **kwargs)
            logging.info(f"{func.__name__} 返回: {result}")
            return result
        except Exception as e:
            logging.error(f"{func.__name__} 异常: {e}")
            raise
    return wrapper

@auto_logger
def forward_pass(input_tensor):
    return "模型输出"

forward_pass(torch.randn(1, 3, 224, 224))
```

**自动重试（容错训练任务）decorator**

```python
from functools import wraps
import time

def retry(max_attempts=3, delay=1):
    """失败后自动重试的装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    print(f"第 {attempt+1} 次失败: {e}，{delay}秒后重试...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=2)
def download_model():
    """下载模型权重"""
    import random
    if random.random() < 0.7:
        raise ConnectionError("网络错误")
    return "模型加载成功"

# 会自动重试，直到成功或达到最大次数
```

**缓存模型推理结果（避免重复计算）**

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def model_inference(input_hash):
    """缓存推理结果，避免重复计算"""
    # input_hash 是输入的哈希值
    # 这样可以避免对同一输入的重复推理
    return model(input_hash)

# 使用示例
image_hash = hash(str(image_array.tostring()))
result = model_inference(image_hash)  # 首次计算
result = model_inference(image_hash)  # 直接从缓存返回
```

**💡 对 AI 的重要性**：

训练流程可控、自动化、模块化写法的基础。装饰器让你的代码更简洁、更易维护。





---

## 4. 并行 / 并发（AI 数据处理的核心技能）

### 4.1 Threading（多线程）

多线程适合 **IO 密集型** 任务（如网络请求、文件 IO）。但由于 GIL，不适合 CPU 密集型任务。

```python
import threading
import time

def io_intensive_task(task_id):
    """模拟 IO 操作（如下载数据集）"""
    print(f"任务 {task_id} 开始")
    time.sleep(2)  # 模拟网络延迟
    print(f"任务 {task_id} 完成")

# 单线程：2个任务耗时 4秒
start = time.time()
io_intensive_task(1)
io_intensive_task(2)
print(f"单线程耗时: {time.time() - start:.2f}s")  # 4秒

# 多线程：2个任务耗时 2秒（真正并发）
start = time.time()
t1 = threading.Thread(target=io_intensive_task, args=(1,))
t2 = threading.Thread(target=io_intensive_task, args=(2,))
t1.start()
t2.start()
t1.join()  # 等待线程完成
t2.join()
print(f"多线程耗时: {time.time() - start:.2f}s")  # 2秒
```

### 4.2 Multiprocessing（多进程）

多进程突破 GIL 限制，适合 **CPU 密集型** 任务。每个进程有独立的解释器和内存。

```python
from multiprocessing import Process, Pool
import time

def cpu_intensive_task(n):
    """CPU 密集型任务"""
    total = sum(i*i for i in range(n))
    return total

# 单进程：耗时约 8秒
start = time.time()
cpu_intensive_task(10**8)
cpu_intensive_task(10**8)
print(f"单进程耗时: {time.time() - start:.2f}s")  # 8秒

# 多进程：耗时约 4秒（真正并行）
start = time.time()
p1 = Process(target=cpu_intensive_task, args=(10**8,))
p2 = Process(target=cpu_intensive_task, args=(10**8,))
p1.start()
p2.start()
p1.join()
p2.join()
print(f"多进程耗时: {time.time() - start:.2f}s")  # 4秒
```

**进程池（Pool）**

```python
from multiprocessing import Pool

def square(x):
    return x * x

# 使用进程池批量处理
with Pool(4) as pool:
    # map 相当于内置 map，但分布在 4 个进程上
    results = pool.map(square, range(100))

print(results[:10])  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

**共享内存与数据复制**

```python
from multiprocessing import Process, Queue
import numpy as np

def worker(queue, data):
    """子进程接收数据"""
    # ⚠️ data 被复制到子进程！这很耗时
    processed = data * 2
    queue.put(processed)

# 用队列（Queue）传递数据
q = Queue()
large_data = np.random.randn(1000, 1000)
p = Process(target=worker, args=(q, large_data))
p.start()
result = q.get()  # 从队列获取结果
p.join()
```

👉 **为什么 PyTorch DataLoader 默认用多进程？**

因为：
1. **数据加载 = CPU 密集型**（图像解码、数据增强都很耗 CPU）
2. **多线程被 GIL 卡住**，无法真正并行
3. **多进程能真正并行**，充分利用多核 CPU

```python
# PyTorch DataLoader 内部使用多进程
from torch.utils.data import DataLoader, TensorDataset
import torch

dataset = TensorDataset(torch.randn(1000, 3, 224, 224))
# num_workers=4 意味着 4 个独立进程加载数据
loader = DataLoader(dataset, batch_size=32, num_workers=4)

for batch in loader:
    # 背后：4 个进程同时加载和预处理数据
    pass
```

### 4.3 concurrent.futures（高级并行 API）

提供更简洁的接口来管理线程和进程池。

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

def task(n):
    time.sleep(1)
    return n * n

# 多线程池
with ThreadPoolExecutor(max_workers=4) as executor:
    # submit 提交单个任务
    future = executor.submit(task, 5)
    result = future.result()  # 等待结果
    print(f"结果: {result}")

# 进程池 + map 批量处理
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(task, range(10)))
    print(results)

# Future 对象用于异步编程
futures = []
with ThreadPoolExecutor(max_workers=4) as executor:
    for i in range(10):
        future = executor.submit(task, i)
        futures.append(future)

# 等待所有任务完成
for future in futures:
    print(future.result())
```

### 4.4 Asyncio（协程）

```python
import asyncio

async def async_task(n):
    """异步函数"""
    print(f"任务 {n} 开始")
    await asyncio.sleep(1)  # 非阻塞等待
    print(f"任务 {n} 完成")
    return n * n

# 运行协程
async def main():
    # 并发运行多个协程
    tasks = [async_task(i) for i in range(5)]
    results = await asyncio.gather(*tasks)
    return results

results = asyncio.run(main())
print(results)
```

**💡 对 AI 的重要性**：

异步编程适合 **IO 密集且需要高吞吐** 的场景（如异步数据预处理、异步推理服务）。

---

## 5. 性能优化（AI 工程非常关键）

### 5.1 内存优化

**深拷贝 vs 浅拷贝**

```python
import copy

# 浅拷贝：只复制一层
original = [[1, 2], [3, 4]]
shallow = copy.copy(original)
shallow[0][0] = 999
print(original)  # [[999, 2], [3, 4]] 原列表被修改了！

# 深拷贝：完全独立复制
original = [[1, 2], [3, 4]]
deep = copy.deepcopy(original)
deep[0][0] = 999
print(original)  # [[1, 2], [3, 4]] 原列表未被修改
```

在数据增强中的应用：

```python
import numpy as np

def bad_augment(images):
    """浅拷贝导致原数据被修改"""
    batch = images.copy()  # 浅拷贝
    batch[:, 0] = 0  # 设置第一列为 0
    return batch

images = np.random.randn(32, 3, 224, 224)
augmented = bad_augment(images)
# 如果实际工作中需要原数据，这会导致问题
```

**避免 numpy → python 列表大量转换**

```python
# ❌ 低效：转换为 Python 列表
import numpy as np
data = np.random.randn(1000000)
python_list = data.tolist()  # 转换，很慢
for x in python_list:
    process(x)

# ✅ 高效：直接迭代 numpy
for x in data:
    process(x)

# ✅ 更高效：向量化操作
result = np.vectorize(process)(data)
```

**使用生成器替代列表**

```python
# ❌ 低效：一次性加载整个数据集
def load_images_bad():
    images = []
    for file in file_list:
        images.append(load_image(file))
    return images  # 返回大列表，占用大量内存

# ✅ 高效：延迟加载
def load_images_good():
    for file in file_list:
        yield load_image(file)  # 用生成器，内存高效
```

### 5.2 加速技巧

**向量化操作**

```python
import numpy as np
import time

# ❌ 循环版本：慢
def slow_sum(arr):
    total = 0
    for x in arr:
        total += x
    return total

# ✅ 向量化版本：快
def fast_sum(arr):
    return np.sum(arr)

arr = np.random.randn(10**7)

start = time.time()
slow_sum(arr)
print(f"循环版本耗时: {time.time()-start:.4f}s")  # ~0.5s

start = time.time()
fast_sum(arr)
print(f"向量化版本耗时: {time.time()-start:.4f}s")  # ~0.001s
```

**NumPy 的广播优化**

```python
import numpy as np

# 广播自动扩展数组维度，避免显式循环
a = np.random.randn(1000, 1)     # (1000, 1)
b = np.random.randn(1, 100)      # (1, 100)

# 自动广播到 (1000, 100)
result = a + b  # 高效、简洁

# 等价的低效写法：
result_slow = np.zeros((1000, 100))
for i in range(1000):
    for j in range(100):
        result_slow[i, j] = a[i, 0] + b[0, j]
```

**Memory Pinning（加速 GPU 复制）**

```python
import torch

# ❌ 普通内存 → GPU：涉及 DMA 转移
data = torch.randn(1000, 1000)  # CPU 内存
gpu_data = data.cuda()  # 复制到 GPU（较慢）

# ✅ 锁定内存 → GPU：更快
pinned_data = torch.randn(1000, 1000, pin_memory=True)
gpu_data = pinned_data.cuda()  # 复制更快（DMA 优化）

# DataLoader 中使用 pin_memory
from torch.utils.data import DataLoader
loader = DataLoader(dataset, pin_memory=True)  # 加速数据转移
```

**高效文件 IO（mmap）**

```python
import numpy as np

# ❌ 普通方式：一次性加载
large_data = np.load('huge_file.npy')  # 占用大量内存

# ✅ mmap 方式：虚拟映射，按需加载
large_data = np.load('huge_file.npy', mmap_mode='r')
print(large_data[0:100])  # 只加载前 100 行到内存
```

### 5.3 代码剖析（profiler）

**timeit**

```python
import timeit

def function_to_profile():
    return sum(range(1000))

# 测量执行时间
time_taken = timeit.timeit(function_to_profile, number=100000)
print(f"平均时间: {time_taken/100000*1e6:.2f} μs")
```

**cProfile**

```python
import cProfile

def slow_function():
    total = 0
    for i in range(10**6):
        total += i
    return total

# 分析函数性能
cProfile.run('slow_function()')
# 输出每个函数的调用次数、执行时间等
```

**PyTorch 的 profiler**

```python
import torch
from torch.profiler import profile, record_function

# 分析模型前向传播
model = YourModel()
x = torch.randn(1, 3, 224, 224)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with record_function("forward"):
        output = model(x)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

**💡 对 AI 的重要性**：

训练速度提升 20%–50% 是常见的收益。通过分析，你能发现真正的性能瓶颈。

---

## 6. Python 设计模式（写可维护 AI 代码的关键）

### 6.1 工厂模式

用工厂函数/类来创建对象，避免硬编码。

```python
# ❌ 硬编码创建不同模型
def train(model_name):
    if model_name == "resnet":
        model = ResNet()
    elif model_name == "vgg":
        model = VGG()
    elif model_name == "mobilenet":
        model = MobileNet()
    return model

# ✅ 工厂模式
class ModelFactory:
    models = {
        "resnet": ResNet,
        "vgg": VGG,
        "mobilenet": MobileNet,
    }
    
    @classmethod
    def create(cls, model_name):
        model_class = cls.models.get(model_name)
        if not model_class:
            raise ValueError(f"未知模型: {model_name}")
        return model_class()

# 使用
model = ModelFactory.create("resnet")
```

**Dataset 工厂**

```python
class DatasetFactory:
    datasets = {
        "imagenet": ImageNetDataset,
        "cifar10": CIFAR10Dataset,
        "coco": COCODataset,
    }
    
    @classmethod
    def create(cls, dataset_name, **kwargs):
        dataset_class = cls.datasets[dataset_name]
        return dataset_class(**kwargs)

# 使用
dataset = DatasetFactory.create("cifar10", root="/data/cifar10")
```

### 6.2 策略模式

不同的算法/策略独立封装，易于切换。

```python
# 定义优化器策略
class OptimizerStrategy:
    def __call__(self, params, lr):
        raise NotImplementedError

class SGDStrategy(OptimizerStrategy):
    def __call__(self, params, lr):
        return torch.optim.SGD(params, lr=lr)

class AdamStrategy(OptimizerStrategy):
    def __call__(self, params, lr):
        return torch.optim.Adam(params, lr=lr)

class CosineAnnealingStrategy(OptimizerStrategy):
    """学习率策略"""
    def __call__(self, optimizer, T_max):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max)

# 使用：易于切换策略
class Trainer:
    def __init__(self, optimizer_strategy, scheduler_strategy):
        self.opt_strategy = optimizer_strategy
        self.sched_strategy = scheduler_strategy
    
    def setup(self, model, lr):
        optimizer = self.opt_strategy(model.parameters(), lr)
        scheduler = self.sched_strategy(optimizer, T_max=100)
        return optimizer, scheduler

# 配置不同策略
trainer = Trainer(AdamStrategy(), CosineAnnealingStrategy())
```

### 6.3 单例模式

确保全局只有一个实例（如日志、配置）。

```python
class Logger:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# 使用
logger1 = Logger()
logger2 = Logger()
print(logger1 is logger2)  # True

# 更简洁的单例：装饰器
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Config:
    def __init__(self):
        self.lr = 0.001

cfg1 = Config()
cfg2 = Config()
print(cfg1 is cfg2)  # True
```

**💡 对 AI 的重要性**：

大项目（CV / NLP / RL）必须保证代码可扩展性。设计模式让代码更灵活、易维护。

---

## 7. AI 工程中的 Python 实战能力

### 7.1 Dataset / DataLoader 自定义

**自定义迭代器**

```python
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        # 延迟加载：只在需要时加载图像
        image = load_image(self.file_list[idx])
        if self.transform:
            image = self.transform(image)
        return image

# 使用
dataset = CustomDataset(file_list)
loader = DataLoader(dataset, batch_size=32, num_workers=4)
for images in loader:
    # 多进程加载数据
    pass
```

**多进程加载**

```python
# num_workers > 0 会启用多进程
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,      # 4 个进程
    pin_memory=True,    # 锁定内存加速 GPU 转移
    prefetch_factor=2,  # 预加载因子
)

# 每个进程独立调用 __getitem__
```

**动态数据增强**

```python
from torchvision import transforms

# 定义增强策略
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

class AugmentedDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __getitem__(self, idx):
        image = self.images[idx]
        # 每次返回不同的增强版本
        image = self.transform(image)
        return image, self.labels[idx]
```

### 7.2 训练框架封装

**装饰器记录训练过程**

```python
from functools import wraps
import logging

def log_training(func):
    """记录训练过程"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        logging.info(f"【开始】{func.__name__}")
        result = func(self, *args, **kwargs)
        logging.info(f"【完成】{func.__name__}")
        return result
    return wrapper

class Trainer:
    @log_training
    def train_epoch(self):
        # 训练代码
        pass
    
    @log_training
    def validate(self):
        # 验证代码
        pass
```

**异步数据预处理**

```python
from concurrent.futures import ThreadPoolExecutor

class AsyncDataLoader:
    def __init__(self, dataset, batch_size=32, num_workers=4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
    
    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            # 提交预处理任务
            future = self.executor.submit(
                self._load_batch,
                i, i + self.batch_size
            )
            yield future.result()
    
    def _load_batch(self, start, end):
        batch = [self.dataset[i] for i in range(start, end)]
        return batch
```

**模型自动保存/恢复**

```python
import torch
import os

class CheckpointManager:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save(self, model, optimizer, epoch, metrics):
        path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics,
        }, path)
        print(f"保存检查点: {path}")
    
    def load(self, model, optimizer, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        print(f"加载检查点: {path}，恢复到 epoch {epoch}")
        return epoch

# 使用
ckpt_mgr = CheckpointManager('./checkpoints')
ckpt_mgr.save(model, optimizer, epoch=10, metrics={'acc': 0.95})
```

### 7.3 模型部署

**TorchScript**

```python
import torch

class SimpleModel(torch.nn.Module):
    def forward(self, x):
        return x * 2

model = SimpleModel()
# 转为 TorchScript（可脱离 Python 环境运行）
scripted = torch.jit.script(model)
scripted.save('model.pt')

# 加载并推理
loaded_model = torch.jit.load('model.pt')
output = loaded_model(torch.randn(1, 10))
```

**多进程推理服务**

```python
from multiprocessing import Process, Queue
import torch

class InferenceWorker:
    def __init__(self, model_path, input_queue, output_queue):
        self.model = torch.jit.load(model_path)
        self.input_queue = input_queue
        self.output_queue = output_queue
    
    def run(self):
        while True:
            request_id, input_data = self.input_queue.get()
            output = self.model(input_data)
            self.output_queue.put((request_id, output))

# 启动多个推理进程
input_q = Queue()
output_q = Queue()
workers = [
    InferenceWorker('model.pt', input_q, output_q)
    for _ in range(4)
]
for w in workers:
    p = Process(target=w.run)
    p.start()

# 提交推理请求
input_q.put(('req_1', torch.randn(1, 3, 224, 224)))
request_id, output = output_q.get()
```

**异步 API 推理服务**

```python
import asyncio
import torch
from aiohttp import web

class AsyncInferenceServer:
    def __init__(self, model_path):
        self.model = torch.jit.load(model_path)
    
    async def infer(self, request):
        data = await request.json()
        input_tensor = torch.tensor(data['input'])
        
        # 异步推理（不阻塞其他请求）
        output = await asyncio.to_thread(
            self._sync_infer,
            input_tensor
        )
        
        return web.json_response({'output': output.tolist()})
    
    def _sync_infer(self, x):
        return self.model(x)

# 启动服务
app = web.Application()
server = AsyncInferenceServer('model.pt')
app.router.add_post('/infer', server.infer)
web.run_app(app, port=8080)

# 客户端请求
# curl -X POST http://localhost:8080/infer -d '{"input": [1, 2, 3]}'
```

---

## 总结

| 主题 | 核心概念 | AI 应用场景 |
|------|--------|----------|
| **执行模型** | GIL、引用计数 | 理解为什么 DataLoader 用多进程 |
| **迭代器/生成器** | yield、延迟计算 | 高效加载大数据集 |
| **装饰器** | 闭包、元编程 | 训练日志、性能监控、自动重试 |
| **并行/并发** | 多进程、异步 | 多核数据加载、异步推理服务 |
| **性能优化** | 向量化、内存管理 | 加速训练、减少显存占用 |
| **设计模式** | 工厂、策略、单例 | 大项目代码结构、易扩展性 |
| **工程实战** | Dataset、训练框架、部署 | 完整的 AI 开发流程 |

掌握这些知识，你就能写出**高效、可维护、易扩展**的 AI 代码！































