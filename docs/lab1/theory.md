# 二、实验原理
## 1. PyTorch深度学习基本部署流程

**PyTorch入门：** 可参考 [PyTorch官方教程](https://pytorch.org/tutorials/)

**PyTorch工作流程图：**<center><img src="../assets/pytorch.png" width = 500></center>

PyTorch是一个开源的深度学习框架，它具有简单易用的API和灵活性，是许多研究人员和工程师喜欢使用的深度学习框架之一。PyTorch使用动态计算图，这意味着在每次前向传播时，计算图都是根据输入动态构建的。这样的设计理念使得调试和模型构建更加灵活。PyTorch的动态计算图使得调试变得更加容易。用户可以在任何时候查看中间结果，检查梯度，以及使用Python的标准调试工具来诊断问题。PyTorch的张量与NumPy数组非常类似，用户可以方便地在PyTorch和NumPy之间进行数据转换。

## 2. 3D目标检测算法SMOKE详解

**SMOKE论文原文：** 可查阅 [SMOKE论文](https://arxiv.org/abs/2002.10111)

**SMOKE框架图：：**<center><img src="../assets/smoke.png" width = 600></center>

SMOKE的基本思想主要是基于CenterNet做进一步开发。SMOKE的网络结构由特征提取模块，关键点检测模块和3D参数回归模块构成。首先，特征提取模块将目标图像作为输入，并提取高级特征供后两个模块使用。然后，关键点检测模块主要是检测目标3D边界框的中心点在图像坐标系上的投影。最后，3D参数回归模块预测描述每个目标3D边界框的8个参数，包括深度偏移，下采样中的x轴和y轴偏移，3个坐标尺寸偏移和2个偏航角。
