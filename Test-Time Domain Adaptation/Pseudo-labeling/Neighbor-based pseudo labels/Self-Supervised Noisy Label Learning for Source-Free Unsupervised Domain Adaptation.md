本文以噪声标签学习的视角解决 Source-Free Unsupervised Domain Adaptation

源预训练模型在目标域上进行推理时，得到的标签不可避免地包含噪声
source-free UDA 可视为另一种形式的噪声标签学习
本文提出了自监督噪声标签学习（Self-Supervised Noisy Label Learning，SSNLL）
和其他自监督学习方法类似，其关键在于避免<font color="#e36c09">模型崩溃</font>（将所有图像都分类到同一类别中）

思想：基于源预训练模型生成的噪声标签包含任务的有效线索

将目标数据 $\mathcal{X}_t$ 划分为两部分：相对于预生成标签损失更小的更干净部分 $\mathcal{X}_{cl}$，相对于预生成标签损失更大的更带噪部分 $\mathcal{X}_{no}$。为了避免上述模型崩溃问题，进一步将其发展为基于标签的数据集分割（label-wise dataset splitting）方法，确保 $\mathcal{X}_{cl}$ 中不存在空类别。之后，分别从 $\mathcal{X}_{cl}$ 和 $\mathcal{X}_{no}$ 中均匀采样图像，以使用预生成标签和自生成标签训练网络

利用源数据预训练的模型记为 $f$，并利用该模型为目标数据 $\mathcal{X}_t$ 预生成带噪标签 $\mathcal{Y}_t$：
$$
\mathcal{Y}_t = {\operatorname*{\operatorname*{\arg\max}}}f(\mathcal{X}_t)
$$
source-free UDA 的第二阶段就转化为如何利用预先生成的噪声标签来微调 $f$

#### 标签去噪预处理
目的：在微调 $f$ 之前修正预先生成的噪声标签 $\mathcal{Y}_t$，因为更低的噪声率有利于后续的噪声标签学习过程，采用两种标签去噪技巧
##### 自适应 Batch Normalization
Batch Normalization 层的总体统计量（均值 $\mu$ 和方差 $\sigma$ ）编码了特定于域（domain-specific）的信息。当数据分布从源域变为目标域时，需要重新计算 BN 层的总体统计量，以便自适应地迁移特征表示，这有利于实现目标域上噪声率更低的伪标签
Adaptive Batch Normalization（AdaBN）倾向于将一个个 batch 的目标数据输入 $f$ 以计算 batch 的统计量，并通过基于动量（momentum-based）的统计量移动平均值来更新总体统计量：
$$
\left \{ \mu, \sigma \right \} = \lambda\left \{ \mu, \sigma \right \} + (1-\lambda)\left \{ \mu, \sigma \right \}_{batch} 
$$
其中 $\left \{ \mu, \sigma \right \}_{batch}$ 表示目标域上当前 batch 的统计量，$\left \{ \mu, \sigma \right \}$ 表示由源域统计量初始化的总体统计量

##### 深度迁移聚类
使用深度迁移聚类（Learning to discover novel visual categories via deep transfer clustering）对预生成的伪标签 $\mathcal{Y}_t$ 去噪，强制同一簇中的样本共享相同的标签

具体来说，对目标特征进行 $k$-means 聚类
$$\min_{\mathcal{M}\in\mathbb{R}^{d\times k}}\frac{1}{N}\sum_{n=1}^N\min_{c_n\in\{0,1\}^k}\|e(x_n)-\mathcal{M}c_n||_2^2\mathrm{~s.t.~}c_n^T\mathbf{1}_k=1$$
$x_n$：第 $n$ 个目标样本 $c_n$：簇分配
$\mathcal{M} \in \mathbb{R}^{d\times k}$ 为簇质心矩阵，每一列是一个聚类中心向量

聚类之后，对于第 $n$ 个样本，找到与它属于同一聚类（$c_j = c_n$）的所有样本索引集合，将这些样本的分类概率分布进行平均，然后取聚合后概率分布的最大值对应的类别作为该样本的**精炼伪标签**
$$
\begin{aligned}
\widehat{p_n}&=\frac{1}{|\Omega|}\sum_{i\in\Omega}p_i\mathrm{~where~}\Omega=\{j\mid c_j=c_n\}\\ \widehat{y_n}&=\arg\max\widehat{p_n}
\end{aligned}
$$

##### 自监督噪声标签学习
将整个数据集划分为更干净的部分 $\left \{ \left ( \mathcal{X}_{cl}, \widehat{\mathcal{Y}_{cl}} \right ) \right \}$ 和更带噪的部分 $\left \{ \mathcal{X}_{no} \right \}$，分别使用预生成和自生成标签来训练这两个部分。因为损失较小的样本倾向于是标签正确的（small-loss trick）
为了确保 $\left \{ \left ( \mathcal{X}_{cl}, \widehat{\mathcal{Y}_{cl}} \right ) \right \}$ 能涵盖所有类别以避免模型崩溃，基于伪标签将数据集分为几组然后对每组内的损失进行排序

![image.png](https://papernote-1394983352.cos.ap-nanjing.myqcloud.com/tta-note-img/20260121194647274.png)

$\mathcal{X}_{cl}$ 直接使用预训练模型预生成的 $\widehat{\mathcal{Y}_{cl}}$ 进行训练
$\mathcal{X}_{no}$ 使用动态自生成标签进行训练：$\min_f\mathcal{L}_{ce}(f(x_n^1),\arg\max f(x_n^2)))$
其中 $x_n^1$ 和 $x_n^2$ 是 $x_n$ 的两种不同的随机变换

随着训练的进行，相对于预生成标签，错误标记样本将获得更大的损失，而正确标记样本将获得更小的损失。因此，$\mathcal{X}_{cl}$ 中的错误标记样本和 $\mathcal{X}_{no}$ 中的正确标记样本将在每次交替中根据 small-loss trick 进行交换，从而逐步驱动更好的优化

算法流程：
![image.png](https://papernote-1394983352.cos.ap-nanjing.myqcloud.com/tta-note-img/20260121194732352.png)
