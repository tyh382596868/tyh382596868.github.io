+++
date = '2025-11-18T15:33:51+08:00'
draft = false
title = 'Real-Time Execution of Action Chunking Flow Policies'
+++


#### 1.1 同步推理：一台服务器，同时跑模型的推理与动作的执行。一次只能干一件事情，模型推理过程中机械臂不执行了，会导致动作的卡顿。
![](/attachment/RTC/e95f847b62e9fc0d436f58d5634c77d6.png)

#### 1.2 异步推理：两台服务器，一台跑模型的推理，一台跑机械臂的动作的执行。action chunk里有n个动作，执行k小于等于n个动作后执行端将观测发给推理端，模型开始推理，同时动作继续执行。经过inference delay延时后，模型将新的动作发给执行端。
![](/attachment/RTC/3e48da357d3b3547d255b0614e38acc3.png)

#### 2.1 推理延迟
什么是推理延迟？
	左图中是一个避障的任务，灰色的部分是障碍物。上一个动作是从上面避障，在上一个动作执行k个动作后，模型开始推理，推理的结果是向下避障。如果推理能立马完成，下一个动作立马执行没什么问题，但是由于推理不能立马完成有个inference delay，在推理完成时，上一个动作已经执行部分了，切换到下一个动作会跳变。
![](/attachment/RTC/00fd4e3cb24fb60d9332755ce47eb4b1.png)

#### 2.2 如何解决推理延迟？
参考图像 Inpainting的方法，inference delay期间上一个动作中肯定会执行的部分看成是原图中没mask掉好的部分，可调整的的部分看成是原图中mask掉退化的图像的部分，可以自由发挥，可调整部分指数衰减，长度不够填充部分权重为零，这也叫做soft mask。
![](/attachment/RTC/77bc6a310147e6392ae4ce3316b73460.png)

#### 2.3 代码实现
$x_0$ 是高斯纯噪声，算出中间过程中带噪声的动作$x_t$ , 由训练的flow matching模型得到速度场$v_t$ ,正常是得到下一时刻去噪的action $x_{t+1}$.
RTC中加了个约束，最终生成的干净的action $x_1$ , 在inference delay部分要与上一个action chunk保持一致。
如何加入这个约束的呢？
	先根据当前带噪声动作$x_t$,和速度场$v_t$得到一个理论上的最终动作$\bar{x_1}$,
	$\bar{x_1}$对$x_t$求导得到Jacobian矩阵，
	将$\bar{x_1}$与$x_{prev}$做差得到error，乘以一个soft mask权重，
	将error与Jocobian做VJP（Vector Jacobian Product）,并更新$v_t$到$v_t'$
	根据更新后的$v_t'$和当前带噪声动作$x_t$得到下一步动作$x_{t+1}$
![](/attachment/RTC/1a0d1457966211422b7a5a5444fc5950.png)

![](/attachment/RTC/b0f10f68b436003dba0eefd3f4f93f34.png)

#### 3.1 雅可比矩阵与向量雅可比矩阵积
![](/attachment/RTC/1aeb003baaf39bfac33e8437428421a0.png)

![](/attachment/RTC/88362327f78a5011578a91084fa5f0a4.png)

![](/attachment/RTC/33759cd7ecfa4ee25e822fe6b30ad009.png)

#### Reference

RTC blog

https://www.physicalintelligence.company/research/real_time_chunking

Training-free Linear Image Inverses via Flows 论文

https://arxiv.org/abs/2310.04432

介绍雅可比矩阵

https://wangkuiyi.github.io/jacobian.html