通过测试熵最小化（test entropy minimization）来进行适应 $\longrightarrow$ 根据预测的熵来优化模型的置信度

为什么选择熵？其与误差和偏移之间的联系
- 熵和误差相关，因为更自信的预测大体上更正确
	![image.png](https://papernote-1394983352.cos.ap-nanjing.myqcloud.com/tta-note-img/20260122113624937.png)
		熵较低的预测在损坏的 CIFAR-100-C 上的错误率较低。确定性可以作为测试过程中的监督
		
- 熵和因损坏引起的偏移相关，因为更多的损坏结果会导致更多的熵。随着损坏程度的加剧，其与图像分类损失具有强相关性
  ![image.png](https://papernote-1394983352.cos.ap-nanjing.myqcloud.com/tta-note-img/20260122114545269.png)
		更多的损坏会导致CIFAR-100-C数据集上更多的损失和熵值增加。熵值可以在没有训练数据或标签的情况下估计数据偏移的程度
		

最小化熵的方式：逐 batch 估计统计量并优化仿射参数（affine parameters）
特点：低维，逐通道特征调制（channel-wise feature modulation）

通过微调的迁移学习（Transfer learning by fine-tuning）：目标标签可用
域适应：源和目标数据可用
测试时训练：源数据可用
以上 settings 无法涵盖当源、目标或监督不能同时可用的实际情况

根据数据，训练和测试过程中损失的不同区分适应 settings
![image.png](https://papernote-1394983352.cos.ap-nanjing.myqcloud.com/tta-note-img/20260122134940274.png)

所提方法不改变训练，在给定参数 $\theta$ 和目标数据 $x_t$ 的情况下，最小化测试期间的预测熵

#### 熵目标
所提方法的测试时目标是最小化模型预测的熵。但是如果单独优化一个预测容易产生平凡解
![image.png](https://papernote-1394983352.cos.ap-nanjing.myqcloud.com/tta-note-img/20260122145254564.png)
解决方案：

#### 参数调整
$\theta$ 是测试时对于源数据的唯一表征，改变 $\theta$ 可能会导致模型偏离其训练
模型 $f$ 可能是非线性的，而 $\theta$ 可以是高维的，这使得测试时使用优化变得过于敏感且效率低下
![image.png](https://papernote-1394983352.cos.ap-nanjing.myqcloud.com/tta-note-img/20260122154442787.png)

解决方案：只更新线性（scales and shifts）和低维（channel-wise）的特征调制
只重新调整源模型的 normalization 层 $\longrightarrow$ 在测试期间更新所有层和通道的 normalization 统计数据和仿射参数
![image.png](https://papernote-1394983352.cos.ap-nanjing.myqcloud.com/tta-note-img/20260122154259552.png)
![image.png](https://papernote-1394983352.cos.ap-nanjing.myqcloud.com/tta-note-img/20260122154357036.png)

![image.png](https://papernote-1394983352.cos.ap-nanjing.myqcloud.com/tta-note-img/20260122155151761.png)
