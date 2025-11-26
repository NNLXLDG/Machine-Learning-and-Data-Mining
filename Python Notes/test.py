import sys

a = "hello, world"
b = "hello, world"
print(a is b)  # False

c = sys.intern("hello, world")
d = sys.intern("hello, world")
print(c is d)  # True
