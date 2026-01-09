探索新的泛化方法：不会预测分布变化，而是在测试时从中学习

自监督辅助任务选择旋转预测任务：将图像旋转 0、90、180 和 270度，让模型预测旋转的角度（4分类问题）
模型：标准的 K 层神经网络
主任务（分类任务）和自监督辅助任务共享模型的前 $\kappa$ 层参数，而后的 $\kappa + 1$ ~ K 层，两个任务均拥有对应于自身任务的参数，即主任务分支和辅助任务分支

![](https://papernote-1394983352.cos.ap-nanjing.myqcloud.com/tta-note-img/20260109170829904.png)

![](https://papernote-1394983352.cos.ap-nanjing.myqcloud.com/tta-note-img/20260109170438271.png)

训练以多任务学习的方式进行，通过分类损失优化主任务分支，自监督损失优化辅助任务分支，共享部分则由两个损失共同优化

![](https://papernote-1394983352.cos.ap-nanjing.myqcloud.com/tta-note-img/20260109171042491.png)

在测试时，对于单个测试样本 $x$ ，TTT 通过最小化 $x$ 上的辅助任务损失来微调共享特征提取器

![](https://papernote-1394983352.cos.ap-nanjing.myqcloud.com/tta-note-img/20260109171513204.png)

通过最小化 2 式得到参数 ($\theta_e$, $\theta_s$)，通过最小化和 $x$ 有关的 3 式得到参数 $\theta_e^*$ ，通过 ($\theta_e^*$, $\theta_s$) 预测 $x$
后，$\theta_e^*$ 被丢弃，共享部分的参数变回 $\theta_e$

在线 TTT：
![17-48-02.png](https://papernote-1394983352.cos.ap-nanjing.myqcloud.com/tta-note-img/20260109171933544.png)
