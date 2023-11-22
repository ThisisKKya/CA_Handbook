# 实验原理


## 1. Apollo 决策规划

&emsp;&emsp;Apollo的决策规划，相当于自动驾驶的“大脑”，通过感知模块对周围环境信息进行整合后，在本环节需要完成自动驾驶任务的行为决策。
<center><img src="../assets/task3-theory1.png" width = 600></center>

### 2.1 Routing 模块
&emsp;&emsp;Apollo Routing模块主要作用：提供了一种高层次的路线规划，为规划模块提供基准路径，负责确定从当前位置到目的地的最优路径。

&emsp;&emsp;常见的路由规划算法如：A*算法、Dijkstra等图搜索算法，在道路网络中查找最短或最优路径。
### 2.2 Planning 模块

&emsp;&emsp;Planning模块根据Routing提供的路线（Path）负责生成车辆的实时行驶轨迹（trajectory），确保安全、平稳且高效地遵循预定路线。在这个过程中需要实时考虑道路上的情况例如红绿灯、障碍物等情况。

&emsp;&emsp;Apollo的Planning模块属于基于规则（Rule-based）的算法实现。具体对于Planning模块的算法实现解析及原理介绍可参考：

1. https://zhuanlan.zhihu.com/p/453001890
   
2. https://zhuanlan.zhihu.com/p/61982682
   
3. https://www.bilibili.com/video/BV1Jg411z7mn/?share_source=copy_web&vd_source=26e8a492df076d8a7ddc782907e20e38
