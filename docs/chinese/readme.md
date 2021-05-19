# CasADiTO 机器人轨迹优化框架

## 简介

CasADi是一个开源的软件工具，用于一般的数值优化，尤其是最优控制（即涉及微分方程的优化）中。 CasADiTO这个框架整合了一些定义最优控制问题的常用代码，使用面向对象的方式让用CasADi来开发轨迹优化问题更加地简便。

目前，这个框架有如下几个主要部分构成

1. model: 用于构建简单的机器人模型，推导运动学和动力学
2. optGen: 用于构建轨迹优化问题

## Model

本框架利用CasADi来辅助动力学和运动学的推导。具体思路为，为模型定义一些CasADi的符号变量，例如`q_1, q_2, ... dq_1, dq_2`, 然后使用这些符号变量表示机器人关键点的位置，动能，势能。再借助拉格朗日方程和自动求导功能即可求出机器人的运动学和动力学

### 使用已有的模型

下面用`model\leggedRobot2D.py`为例，这个定义了一个在xz平面的机器人，可以用来优化后空翻等左右对称的动作

一个模型被封装为了一个类，可以之间传入相应参数字典或者传入`yaml`文件来定义

```python
from model.leggedRobot2D import LeggedRobot2D
model = LeggedRobot2D.fromYaml("data/robotConfigs/JYminiLite.yaml")
```

这个模型有两个重要的成员函数或者变量，分别是 `buildEOMF`, `pFuncs`, `visulize`

#### `buildEOMF(self, consMap, name=""):`

`consMap: (bool, bool)`, 分别代表后腿和前腿是否有 holonomic constraint. 

返回一个Casadi Function `g(x, u, F, ddq)->[SX]`, 用于判断给定的`x`, `u`, `F`, `ddq` 是否满足EoM (Equation of Motion)

`x`, `u`, `F`, 分别是机器人状态，输入，约束外加的力。如果一个consMap为 `True`, 则会增加对应脚尖`ddq=0`的约束。如果一个consMap为 `False`，则会增加对应约束外力`F`为0 的约束。

#### `pFuncs`

一个字典，包含所有的位置函数 Casadi Function `(x)->p`

#### `visulize(self, x)`

使用 `matplotlib.pyplot` 将机器人现在的状态画出来， 返回画出来的线。

**使用示例**

```python
def animate(i):
    x = xlog[i] # xlog is some history of the trajectory
    linesol = model.visulize(x)
    return linesol,

ani = animation.FuncAnimation(
    fig, animate, ...)
```

### 构建一个新的模型

请见 `legedRobot2D`中的详细定义。 目前新模型都是用 `articulatedBody.py`中定义的类。

## optGen

`optGen`使用面向对象的方式来方面轨迹优化问题的定义。在定义轨迹优化问题的过程中，我们一般需要一个大循环，执行以下步骤: (1)变量的声明(例如当前步的系统输入), (2)系统动力学的模拟(通过增加约束或者模拟的方式，到下一个时间点)，(3)损失函数和约束的添加。在本框架中，我们会定义一个轨迹优化对象，使用类似`addCost`的方法实现(3)添加损失函数和约束，使用`step`自动声明变量以及移动到下一个时间点。 因此，定义一个轨迹优化的代码会有如下的结构

```python
opt = optGen(...) # 定义轨迹优化对象
opt.init(...)
for i in range(NSteps):
    opt.step(...)
    opt.addCost(...)
    opt.addConstraint(...)
```

### 使用optGen中的对象

**下面以`optGen\trajOptimizer.py`中`TowrCollocation`**为例

一个`TowrCollocation` 对象会有四个子对象，分别为 `Xgen`, `Ugen`, `Fgen`, `dTgen`。 顾名思义，这些子对象分别负责声明机器人状态，输入，约束外力，时间步长几种变量。之所以采用这种模块化的设计，是因为这样只用更改一个模块就可以构建不同的轨迹优化问题。 例如 `TowrCollocationDefault` 和 `TowrCollocationVTiming` 之间的区别在于 `dTgen`是否要将每个阶段的步长设为变量，这二者分别使用了 `dTGenVariable` 和 `dTGenDefault`

优化问题的构建过程中，`opt`会维护一个存放当前*状态*的字典，例如`TowrCollocation`每一步的状态就有`"x", "u", "F", "ddq1"`

#### `begin(self, x0, u0, F0, **kwargs)`

声明变量`x,u,F`，同时增加`x = x0`的约束。初始化`opt`当前状态(`opt._state`)

#### `step(self, dynF, x0, u0, F0, **kwargs):`

`dynF`是一个`Function (dx, x, UF)->g`, 用来设置动力学约束。 `opt`会将当前状态中的`x,u,F,ddq`喂进去(`ddq`是使用下一步的`x`)计算出来的。

`x0,u0,F0`都是初值。

#### `addCost(self, func)`; `addConstraint(self, func, lb, ub)`

添加当前步的cost function或者约束

`func(**kwargs)->[SX/MX]` 这个函数会被关键字参数的方式调用。出入的参数为`opt`当前步的*状态*(`_state`). 因此， func只能作用于state中包含的量。

`lb,ub`: `lower bound`, `upper bound` 

> 为了使用方便，`func`的参数并不需要涵盖所有的state, 用什么写什么就行

### 添加超参数

### 添加变量

### 自定义算法


