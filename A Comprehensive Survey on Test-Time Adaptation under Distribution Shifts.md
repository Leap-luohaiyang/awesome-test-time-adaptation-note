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

##### 基于优化的伪标签

##### 基于集成的伪标签

#### 一致性训练


### TTBA
单独将预训练模型适应一个或几个实例。换句话说，每个 batch 的预测独立于其他 batch 的预测

### OTTA
以在线方式将预训练模型适应目标数据 $\left \{ b_1, \cdots ,b_m \right \}$，其中每个 batch 只能观察一次。重要的是，从之前的 batch 中学到的知识可以促进对当前 batch 的适应

在测试时使预训练的源模型适应一个域，一个 batch 或者甚至一个实例，这些离线的测试时适应通常需要一定数量的样本来形成 batch 或域，这对于数据连续且按顺序到达的流（streaming）数据场景可能不可行

给定在源域上训练好的模型 $f_\mathcal{S}$ 和一系列未标记的 batch $\left \{ \mathcal{B}_1, \mathcal{B}_2, \cdots \right \}$。OTTA 旨在积累在以前见过的 batch 中学到的知识以适应当前的 batch

