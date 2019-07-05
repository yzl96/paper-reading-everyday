# Fast LIDAR-based Road Detection Using Fully Convolutional Neural Networks



## 输入：

​	$x \in [6, 46] , y \in [-10,10]$  ,栅格 0.1*0.1
​	点的个数
​	平均反射率
​	高度均值
​	高度标准差
​	最大高度
​	最小高度
​	
​	200 × 400 × 6



## 网络结构：

​	输入6通道
​	encoder 提取特征然后maxpooling , 主要作用是下采样减小内存。
​	context module 聚集多尺度的上下文信息通过dilated convolutions.
​	decoder 上采样feature map ，通过 max-unpooling layer接两个卷积层
​	输出层返回一个道路置信图，每个像素代表对应lidar中栅格位置的道路的概率



## Context module：

​	保持参数和层数尽量少的情况下扩大感受野.
​	dilated convolution 提供感受野的指数级增长但是不损失分辨率（特征图大小不变).



![1562234638250](/home/ovo/.config/Typora/typora-user-images/1562234638250.png)



## 数据扩增：

​	绕 lidar z轴 [-30,30] , 3度步进。然后关于x轴镜像。数据量扩大了42倍。

## label生成：

​	road从图像到点云投影不准，所以把点云投影到对应的图像视角中，决定那个点属于road.
​	采用相同的输入处理方式，每个cell中不再是高度等数据，而是 属于road的概率.
​	为了增加点云的密度，得到一个稠密的标注，点云被线性插值



## Loss:

​		$L=-\frac{1}{N \times W \times H} \sum_{i=1}^{N} \sum_{m=1}^{W} \sum_{n=1}^{H} \log p_{m, n}^{i}$
​		p代表正确分类的概率
​		batch 4

在每个dilated convolution layer之后都接了spatial dropout layer (0.25)



## 效果：

![1562237462480](/home/ovo/.config/Typora/typora-user-images/1562237462480.png)





# Progressive LiDAR Adaptation for Road Detection

​	将lidar信息引入视觉道路检测有困难，lidar数据和提取到的特征 和 图像数据和特征不在一个空间上。
​	论文主要解决了如何将雷达信息引入基于视觉的道路检测。
​	
​	data space adaptation : transforms lidar data to the visual data space to align with the perspective view by applying altitude(高度) difference-based transformation.

​	feature space adaptation : adapts lidar features to visual features through a cascaded fusion structure.

​	目前利用lidar数据提升视觉道路检测的效果不明显。
​	作者调查了一下为什么不明显，提出了新的方法。
​	两种困难：
​		1.lidar 和 camera是两种不同的数据，难以定义一个空间去结合这两种数据。尽管可以将点云投影到图像中，但这回改变道路在lidar 数据中的样子, 使得道路在点云空间中更不易于区分，这会导致深度学习模型无法从lidar数据中学到东西。

  2. 难以结合从两种数据提取出来的特征。图像中道路是使用rgb的像素表示，点云中道路是使用离散的点表示，非常可能两者提取出来的特征也在不同空间。

     ​	

     融合视觉和雷达的方法准确率不如纯视觉。
     		

     作者提出了一种转换关系能够把雷达数据转换到视觉空间，雷达特征到视觉特征空间。

     数据转换的同时能够使点云中的道路容易区分。
     然后通过一个串联的融合结构，特征空间转换使得雷达特征更好的补充和提升视觉特征。

     loss:
     $\min _{W} \sum_{i} \sum_{x, y} \mathcal{L}\left.\left(f\left(I_{i}, L_{i} ; W\right), \hat{y}\right)\right|_{x, y}$

     i 代表哪个样本

     $f(I, L ; W)=f_{\text {parsing}}\left(f_{\text {fuse}}\left(f_{\text {vis}}\left(I ; W_{v i s}\right), g\left(L ; W_{\text {lidar}}\right)\right)\right)$
     g代表progressive lidar adaptation function
     $f_{vis}$ 代表 visual image-based road detection function ， ResNet101
     $f_{fuse}$ 融合操作
     $f_{parsing}$ 最后的二分类, pyramid scene parsing module 后 接 2-class softmax.

     g由两部分组成： data space adaptation step and feature space adaptation step.
     data space adaptation 中把雷达数据转到2d同时使道路易于区分。
     feature space adaptation 中 引入一个可学习的module 将lidar feature转换到一个更好的补充视觉特征的空间。

     $g\left(L ; W_{\text {lidar}}\right)=g_{\text {feat}}\left(f_{\text {lidar}}\left(g_{\text {lata}}(L) ; W_{\text {lidar}}\right)\right)$

     

     ![1562292868306](/home/ovo/.config/Typora/typora-user-images/1562292868306.png)



data space adaptation step:  altitude difference-based transformation method to transform the lidar data space.

altitude  difference-based transformation基于观察，3d空间中的道路表面是平坦的并且在高度防线相对平滑，相对于车辆和建筑来说.
在将点云映射到图像平面后，这种平滑性可以被保存通过记录原始3d点云的高度信息。作为结果，道路区域可以更好的被区分在投影后的点云数据中根据在图像平面中的高度变化。
altitude difference-based transfromation 根据以下公式计算点的坐标:

$g_{d a t a}\left.(L)\right|_{x, y}=V_{x, y}=\frac{1}{M} \sum_{N_{x}, N_{y}} \frac{\left|Z_{x, y}-Z_{N_{x}, N_{y}}\right|}{\sqrt{\left(N_{x}-x\right)^{2}+\left(N_{y}-y\right)^{2}}}$

在图像平面跟周围点做了一个高度差平均。

![1562294732487](/home/ovo/.config/Typora/typora-user-images/1562294732487.png)

feature space adaptation:
	目的：将lidar feature space 转换到另一个space 去使得lidar feature更好的补充图像特征并提高基于图像的道路检测性能。因为不知道什么样的变换好，所以引入了learning-base module去学习这个操作。
	

​	假设线性变换可以定义这个操作：
​		$g_{\text {feat}}\left(\mathbf{f}_{\text {lidar}}\right)=\alpha \mathbf{f}_{\text {lidar}}+\beta$   ， 缩放加平移

​		$\mathbf{f}_{\text {lidar}}=f_{\text {lidar}}\left(g_{\text {data}}(L) ; W_{\text {lidar}}\right)$
​		
​		通过神经网络来估计这个变换，		
​		

​		$\alpha=f_{\alpha}\left(\mathbf{f}_{\text {lidar}}, \mathbf{f}_{v i s} ; W_{\alpha}\right)$  ， $f_{\alpha}$ 代表计算 $\alpha$ 的神经网络函数， $W_{\alpha}$ 代表对应的权重

​		$\beta=f_{\beta}\left(\mathbf{f}_{\text {lidar}}, \mathbf{f}_{v i s} ; W_{\beta}\right)$
​	
​		
​		$f_{\alpha}$ 与 $f_{\beta}$  都使用全卷积操作。

​		$f_{lidar} 与 f_{vis}$  concatenate 作为 $f_{\alpha} 与 f_{\beta}$ 的输入

​		![1562295598118](/home/ovo/.config/Typora/typora-user-images/1562295598118.png)

​		feature space adaptation 只 包含 3 个 1×1的卷积操作，三个element-wise multiplication or addition operations.
​	

​		Cascaded Fusion for Adapted Lidar Information

​			$f_{fuse}$ 通过使用residual-based cascaded fusion structure.
​			





