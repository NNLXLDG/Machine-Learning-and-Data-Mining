# 定义一个简单的装饰器
def simple_decorator(func):     # 传入一个函数作为参数
    # 包装原始函数的内部函数
    def wrapper():
        print("即将执行函数...")
        func()                  # 执行原始函数
        print("函数执行完毕。")
    return wrapper

# 使用装饰器
@simple_decorator
def greet():
    print("你好！")

# 调用被装饰的函数
greet()
