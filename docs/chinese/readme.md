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

一个`TowrCollocation` 对象会有四个子对象，分别为 `Xgen`, `Ugen`, `Fgen`, `dTgen`。 顾名思义，这些子对象分别负责声明机器人状态，输入，约束外力，时间步长几种变量。之所以采用这种模块化的设计，是因为这样只用更改一个模块就可以构建不同的轨迹优化问题。 例如 `TowrCollocationDefault` 和 `TowrCollocationVTiming` 之间的区别在于 `dTgen`是否要将每个阶段的步长设为变量，这二者分别使用了 `dTGenVariable` 和 `dTGenDefault`。
模块化可以极大地方便算法的开发和测试，还以`towrCollocation`为例，`towrCollocation`只负责把`ddq = f(x,u,F,dt)`的动力学约束加入到优化问题里面，但至于`dt`是一个优化变量，还是一个常数，还是一个超参数，这是由`dTgen`来决定的，改变`dt`的类型完全不会影响其他模块的功能和实现

优化问题的构建过程中，`opt`会维护一个存放当前*状态*的字典`opt._state`，例如`TowrCollocation`每一步的状态就有`"x", "u", "F", "ddq1"`

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

### 使用超参数

超参数是在问题的构建中，可变，但并非优化变量的参数。例如，各个cost的权重全都可以设置成超参数。跳台阶的已知地形也是超参数

> 超参数功能的引入是为了方便CPP优化问题的生成和使用的。如果只用python进行离线优化的话，其实用常数设置或者使用一个变量设置优化问题并没有区别。但对于CPP而言，如果没有超参数的话，每个参数变动后(例如不同的地形)则需要生成和编译两遍，这是在线优化所不能允许的。 例如跳跃距离是常数的话，那么距离0.3和0.8需要作为两个不同的优化问题来编译。但如果距离也是一个casadi的变量的话，那么生成的CPP优化问题就会包括这个变量，编译一次之后，在执行时传入不同的数值即可。

#### 超参数声明

在优化问题的构建中，有两种方式可以初始化超参数。第一种只需要把变量声明为`ca.SX.sym`即可， 例如下面声明了用来表示地形的点列。然后在使用之后调用`opt.newhyperParam`

```python
terrainPoints = ca.SX.sym("terrainPoints",7, 2)
terrian = pointsTerrian2D(terrainPoints, 0.001) ## 像正常`ca.SX`一样使用
opt.newhyperParam(terrainPoints) ## 将这个超参数添加到opt里面
```

第二种方法是直接调用`opt.newhyperParam`, 传入超参数的名字和大小。即可自动生成一个`ca.SX`并且添加在`opt`里面。 例如 cost weights就可以使用这种方法添加

```python
costU = opt.newhyperParam("costU") # 声明和添加超参数 
# in loop ...
    opt.addCost(lambda x,u: costU*ca.dot(u[:4],u[:4])) # 使用超参数
```

#### 超参数赋值

在求解优化问题时，超参数需要被赋值，让优化问题成为一个特定的优化问题。在python中
```python
opt.setHyperParamValue({"terrainPoints": np.array([
                            [-2, 0.3],
                            ...
                            [2, 0.]
                        ]), 
                        "costU":0.01,
                        "costDDQ":0.0001,
                        "costQReg":0.1})
opt.solve(...) ## 赋值超参数之后便可以求解
```

在cpp环境中, 通过一个字符串和数值组成的向量来设置超参数
```c++
std::vector<std::pair<std::string, const double* > > hp = {
    std::make_pair("costU", &hp_cost_u),
    std::make_pair("costDDQ", &hp_cost_ddq),
    std::make_pair("costQReg", &hp_cost_qreg),
    std::make_pair("distance", &hp_dist)
    // std::make_pair("terrainPoints", hp_terrainPoints)
};

SmartPtr<TNLP> mynlp = new TONLP(..., hp);
```


### 添加变量

除了一条轨迹外, 有时我们可以用优化器求解其他的变量， 例如，求解每一个阶段的时间步长。 如果更夸张一点，可以同时求解参考轨迹和反馈的PD参数。我们可以通过修改`uGen`, `xGen`来实现这样的功能。

例如，下面的代码分别定义了`dTGenDefault` 和 `dTGenVariable`。 二者都暴露的同样的接口:`parse["t_plot"]`, `step`中返回`dt`。 但不同的是后者的`dT`是一个变量，而前者返回的是常数。这样在求解是，后者的`dt`就会被当做优化变量一并求解。

```python
class dTGenDefault(optGen):
    def __init__(self, dT):
        super().__init__()
        self._dT = dT
        self._t_plot = []
        self._parse.update({
            "t_plot": lambda: ca.vertcat(*self._t_plot)
        })
    
    def chMod(self, modName, *args, **kwargs):
        pass
    
    def _begin(self, **kwargs):
        self._t_plot.append(0)
    
    def step(self, step):
        self._t_plot.append(self._t_plot[-1] + self._dT)
        return self._dT

class dTGenVariable(dTGenDefault):
    def __init__(self, dT, dtLim):
        super().__init__(dT)
        self._dtLim = dtLim
        self.curent_dt = None
    
    def chMod(self, modName, *args, **kwargs):
        T = ca.MX.sym('dT_%s'%modName, 1)
        self._w.append(T)
        self._lbw.append([self._dtLim[0]])
        self._ubw.append([self._dtLim[1]])
        self._w0.append([self._dT])
        self.curent_dt = T

    def step(self, step):
        self._t_plot.append(self._t_plot[-1] + self.curent_dt)
        return self.curent_dt
```

添加新的变量，只需要将相应的值填入`self._w`, `self._lbw`, `self._ubw`, `self._w0`

### 自定义算法

这里的算法指的是`opt.step()`如何进行。 具体而言，优化问题的每一步需要初始化那些变量？需要增加那些约束来保证前后时间点符合动力学。

请见`TowrCollocation`中的算法定义。`TowrCollocation`每一步会分别调用它子模块的`step`生成新的优化变量, `u`, `F`, `x`。 它会使用三次样条来拟合前后两个时间点的状态`q_0,dq_0, q_1,dq_1`，这样便可以从样条曲线中求得`ddq`。之后施加动力学约束约束`ddq_0 = F(x_0, u_0, F_0)`, 和`ddq`连续的约束: 前一步的`ddq_1`等于后一步的`ddq_0`

## 目前已有的优化问题

