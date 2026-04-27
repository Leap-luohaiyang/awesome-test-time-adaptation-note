测试时适应（TTA）：进行预测前，在测试过程中使预训练模型适应未标记数据
TTA 方法可分为三种不同情况：
1、test-time domain adaptation (TTDA)
2、test-time batch adaptation (TTBA)
3、online test-time adaptation (OTTA)

![image.png](https://papernote-1394983352.cos.ap-nanjing.myqcloud.com/tta-note-img/20260119103658241.png)
<font size="2">TTA 的目的是在进行预测之前使预训练模型适应各种类型的未标记测试数据，包括单个 batch（TTBA），流数据（OTTA）或者一整个数据集（TTDA）
</font>

为了更好地说明，考虑场景：在测试时有 $m$ 个未标记的 mini-batches，表示为 $b_1, \cdots ,b_m$

### TTDA
也称为 source-free domain adaptation，在生成最终预测前利用所有的 $m$ 个测试 batch 进行多个 epoch 的适应。在适应过程中需要见到所有测试数据（目标数据）
#### 基于伪标签的方法
基于伪标签的方法旨在为每个未标记样本 $x$ 分配类别标签 $\hat{y}$，并通过优化以下监督学习目标来指导学习过程
$$\min_\theta\mathbb{E}_{\{x,\hat{y}\}\in\mathcal{D}_t}w_{pl}(x)\cdot d_{pl}(\hat{y},p(y|x;\theta))$$
其中 $w_{pl}(x)$ 表示和每个伪标记样本 $\left \{ x, \hat{y} \right \}$ 相关的实值权重，$d_{pl}(\cdot)$ 表示预测标签概率分布与伪标签概率之间的散度，如果使用交叉熵作为散度度量，则为 $-\sum_c\hat{y}_c\log[p(y|x;\theta)]_c$
由于域偏移，伪标签不可避免地不准确，存在三种解决方式：
（1）通过去噪提高伪标签的质量
（2）通过 $w_{pl}(x)$ 过滤不准确的伪标签
（3）开发用于伪标记的鲁棒散度度量
##### 基于质心的伪标签
关键思想：基于网络预测和目标特征获得目标-特定的类质心，通过最邻近的质心导出无偏伪标签
$$\begin{cases}m_c=\sum_x[p_\theta(y_c|x)\cdot g(x)]/\sum_xp_\theta(y_c|x),c\in[1,C],\\\hat{y}=\arg\min_cd(g(x),m_c),\forall x\in\mathcal{D}_t,&\end{cases}$$
其中 $p_\theta(y_c|x) = [p(y|x;\theta)]_c$ 表示与第 $c$ 个类别相关的概率，$g(x)$ 表示输入 $x$ 的特征，$m_c$ 表示第 $c$ 个类别质心，$d(\cdot, \cdot)$ 表示余弦距离函数。 这种方式的优势在于：类别质心总是包含鲁棒的判别信息并能缓解类别不平衡问题，普遍用于一些 TTDA 研究中

[[BMD A General Class-balanced Multicentric  Dynamic Prototype Strategy for Source-free  Domain Adaptation|BMD]]：粗略的质心可能无法有效地表示模糊数据，采用 K-means 聚类来挖掘每个类别的多个原型

##### 基于邻居的伪标签
结合相邻标签的预测来生成伪标签，依赖于局部平滑性假设
[[Self-Supervised Noisy Label Learning for Source-Free Unsupervised Domain Adaptation|SSNLL]]：在目标域中执行 $K$-means 聚类，并聚合样本所在簇内的邻居的预测

##### 互补伪标签
[[Domain adaptive semantic segmentation without source data|LD]]：开发了一种启发式策略来随机选择具有中等预测分数的信息化互补标签

##### 基于优化的伪标签
通过利用目标标签分布的先验知识（例如类别平衡），一些 TTDA 方法改变每个类别的阈值从而为每个类别选择一定比例的样本点。这种策略有助于避免“赢家通吃”困境，即伪标签只来自几个主要类别


##### 基于集成的伪标签

#### 一致性训练

#### 源分布估计
通过从预训练模型推断数据来弥补源数据的缺失，将具有挑战性的 TTDA 问题转化为易于学习的 DA 问题。现有的源分布估计方法可分为三组：从随机噪声生成数据、数据转换以及数据选择

![image.png|428](https://papernote-1394983352.cos.ap-nanjing.myqcloud.com/tta-note-img/20260417131104176.png)
##### 数据生成
[[Model adaptation Unsupervised domain adaptation without source data|3C-GAN]](CVPR 2020)：引入了一个以随机采样标签为条件的生成器，结合预训练好的分类器生成目标风格的带标签的样本

##### 数据转换


### TTBA
单独将预训练模型适应一个或几个实例。换句话说，每个 batch 的预测独立于其他 batch 的预测

### OTTA
以在线方式将预训练模型适应目标数据 $\left \{ b_1, \cdots ,b_m \right \}$，其中每个 batch 只能观察一次。重要的是，从之前的 batch 中学到的知识可以促进对当前 batch 的适应

在测试时使预训练的源模型适应一个域，一个 batch 或者甚至一个实例，这些离线的测试时适应通常需要一定数量的样本来形成 batch 或域，这对于数据连续且按顺序到达的流（streaming）数据场景可能不可行

给定在源域上训练好的模型 $f_\mathcal{S}$ 和一系列未标记的 batch $\left \{ \mathcal{B}_1, \mathcal{B}_2, \cdots \right \}$。OTTA 旨在积累在以前见过的 batch 中学到的知识以适应当前的 batch

上述设定对应于 [[TENT FULLY TEST-TIME ADAPTATION  BY ENTROPY MINIMIZATION|Tent]] 中解决的问题。然而，测试时的样本可能来自各种不同的分布，从而导致错误累积和灾难性遗忘等新的挑战。为了解决这个问题，[[Continual Test-Time Domain Adaptation|CoTTA]] 和 EATA [298] 研究了持续测试时间适应问题，使预训练的源模型适应不断变化的测试数据

#### 伪标签
采用在测试时生成的伪标签来进行模型更新
[[DLTTA Dynamic Learning Rate for Test-time  Adaptation on Cross-domain Medical Images|DLTTA]]：指出对域偏移程度不同的测试样本应用相同学习率会产生次优结果，应当针对不同测试样本采用动态学习率；存储由之前的测试样本的特征和预测标签构成的对，判断当前测试样本的偏移程度
[[TEST-TIME ADAPTATION VIA SELF-TRAINING  WITH NEAREST NEIGHBOR INFORMATION|TAST]]：


#### 一致性正则化
[[Robust Mean Teacher for Continual and Gradual Test-Time Adaptation|RMT]]：指出在均值学生-教师网络中，交叉熵损失的不良梯度特性，并证明对称交叉熵具有更好的梯度属性；利用增强的测试数据和对比学习来学习输入空间中微小变化的不变性，同时将测试特征空间拉向源预训练模型适定的源域
[[Decorate the newcomers Visual domain prompt for continual test time adaptation|VDP]]：冻结源模型，学习特定于域的提示和域不变提示，和测试图像和提示加和后，利用重构的图像进行训练和测试；提出对域偏移敏感的参数识别方法，惩罚对域偏移敏感的参数，并稳定更新域不敏感参数以巩固域不变知识

#### 防遗忘正则化



