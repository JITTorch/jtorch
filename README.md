# JTorch: 一个全兼容 PyTorch 接口的高性能动态编译深度学习框架

JTorch 是一个完全兼容 PyTorch 接口的深度学习框架，同时基于 Jittor 元算子与统一计算图特性的加持，实现高性能动态编译，同时，用户原来使用的PyTorch代码，不需要进行任何修改，即可加速运行。总结而言，JTorch具有以下几点优势：

1. 零成本：完全兼容原生 PyTorch 接口， 用户代码不需要作任何更改。
2. 速度快：通过统一计算图执行方法，JTorch可以实现对代码的动态编译以及加速，相比原版 PyTorch拥有更好的性能。
3. 支持硬件多：JTorch底层通过元算子抽象，可以快速兼容适配多种人工智能芯片。
4. 兼容生态： 对原有 PyTorch 生态形成兼容，如各种第三方开发的 PyTorch 模型库。
5. 兼容计图： JTorch完全兼容计图，计图中的接口可以混合使用，性能高。
6. 完全自主可控： JTorch 具有完全的自主知识产权，用户完全不需要安装 Torch，即可直接使用。


JTorch相关连接：

*  [Github](https://github.com/JITTorch/jtorch)
*  [Jittor 论坛](https://discuss.jittor.org/)
*  即时通信: QQ Group(761222083)

# 安装与测试

安装方法如下：

```
python3 -m pip install jtorch
```

注意，请使用python3.7及以上的版本

运行简单测试：

```
python3 -m jtorch.test.test_tutorial
```

# 快速入门

## 使用 JTorch 实现简单动态网络（PyTorch兼容）

```python
# -*- coding: utf-8 -*-
import random
import torch
import math


class DynamicNet(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate five parameters and assign them as members.
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))
        self.e = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 4, 5
        and reuse the e parameter to compute the contribution of these orders.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same parameter many
        times when defining a computational graph.
        """
        y = self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3
        for exp in range(4, random.randint(4, 6)):
            y = y + self.e * x ** exp
        return y

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f'y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3 + {self.e.item()} x^4 ? + {self.e.item()} x^5 ?'


# Create Tensors to hold input and outputs.
x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

# Construct our model by instantiating the class defined above
model = DynamicNet()

# Construct our loss function and an Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)
for t in range(60000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t % 2000 == 1999:
        print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print(torch.liveness_info())

print(f'Result: {model.string()}')
```

## 联系我们

电子邮件：jtorch@qq.com

提出issue：https://github.com/jittorch/jtorch/issues

QQ 群：761222083


## 版权声明

如LICENSE.txt文件中所示， JTorch 使用Apache 2.0版权协议。

