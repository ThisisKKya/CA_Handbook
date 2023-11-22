# 实验原理
## 1. 虚拟仿真测试
### 1.1 Apollo自动驾驶系统
Apollo是由百度研发的工业界主流开源自动驾驶系统,主要工作流如图所示。Perception感知模块获取车辆周围环境信息，识别障碍物红绿灯等。Prediction预测模块预测各障碍物未来的运动轨迹与速度。Planning规划模块根据预测的结果计算车辆自身的驾驶轨迹。Control模块根据轨迹，生成最后的油门和方向盘的控制指令。

因此，感知作为自动驾驶任务的最上游输入，对于整个自动驾驶的正常工作至关重要。实验1中同学们训练的Smoke模型正是感知中的重要视觉模型。在实验2中，我们将要把实验1中的模型部署到自动驾驶系统上，并对它的效果进行测试分析。
<center><img src="../assets/apollo.png" width = 600></center>
### 1.2 Carla虚拟仿真器
 要对自动驾驶系统进行检测，例如 **测试同学们训练的AI视觉感知模型** 在实际的自动驾驶任务中的运行表现，在现实中测试成本高昂且容易发生意外。Carla是一款基于Unreal引擎的3D虚拟世界仿真器，能在仿真环境中模拟自动驾驶汽车运行的环境，从而达到测试的目的。
 <center><img src="../assets/carla.gif" width = 600></center>
### 1.3 Bridge桥接器
    如何把大象放进冰箱：打开冰箱门，放入大象🐘，关上冰箱门。
    如何做自动驾驶虚拟仿真：仿真发送传感器数据、计算自动驾驶任务、执行车辆控制指令。
<center><img src="../assets/three_things.png" width = 500></center>

桥接器Bridge作为Apollo与Carla的中间层，将自动驾驶系统Apollo接入Carla进行仿真测试，一共做3件事：

* Bridge将Carla中的传感器仿真数据（摄像机画面、激光雷达数据、车辆自身数据）传给Apollo。<center><img src="../assets/sensor.png" width = 400></center>

* Apollo根据接收到的传感器数据进行感知识别（使用同学们训练的AI视觉模型），决策规划，产生最后的控制指令（~~右满舵，油门踩死~~）。
    <div style="display: flex;">
        <img src="../assets/pn.png" style="width: 70%;">
        <img src="../assets/cat.gif" style="width: 30%;">
    </div>
* Bridge将Apollo产生的控制指令传给Carla，控制Carla中的车进行驾驶。
<center><img src="../assets/highway.png" width = 300></center>


## 2. 实验任务
    “阿Sir，讲了这么多，我们到底要做什么呢？”
<center><img src="../assets/案例实战.png" width = 300></center>
在之前的实验中，同学们训练了自己的AI视觉感知模型。然而，一个模型的最终任务不是作为一个玩具，仅仅在数据集上进行识别。因此，在任务1中，同学们需要将自己的感知模型部署在工业界自动驾驶系统上，让它成为智能汽车的眼睛，实现它的自我价值与梦想。

而在任务2中，为了验证搭载了同学们自己训练的感知模型的自动驾驶系统能否上路，我们将在虚拟仿真平台中对其测试。同学们可以自由地设置不同的天气、障碍物、交通流来验证感知模型的性能表现，探索影响视觉感知的因素。
