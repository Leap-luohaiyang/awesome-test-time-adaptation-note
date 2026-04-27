增强自标记（augmented self-labeling）：利用 Sinkhorn-Knopp 算法解决最优传输问题以提升目标样本伪标签
模型重训练（model re-training）：通过自标记目标数据重新训练模型

#### Augmented Self-Labeling
采用随机数据增强的样本相对应的多个输出预测的加权平均值来优化目标标签
原始样本 $x_i$ 和 $M$ 个不同的增强版本 $\left \{ x_i^m \right \}_{m=1}^M$
具体增强方式：随机调整裁剪大小、随机自动对比度和随机颜色失真
为减少噪声影响，集成样本原始版本和增强版本的预测概率，$x_i$ 属于类别 $y$ 的概率：
$$p_{iy} = \frac{1}{2}p(y\mid x_i;\theta) + \frac{1}{2M}\sum_{m=1}^Mp(y\mid x_i^m;\theta)$$
通过以下增强自标记提升伪标签：
$$
\begin{aligned}&\min_{\{q_{iy}\}}-\sum_{i=1}^N\sum_{y=1}^Kq_{iy}\log p_{iy}+\lambda\sum_{i=1}^N\sum_{y=1}^Kq_{iy}\log q_{iy}\\&s.t.\quad\forall i,y:\quad q_{iy}\in[0,1],\quad\sum_{y=1}^Kq_{iy}=1,\quad\sum_{i=1}^Nq_{iy}=\frac{N}{K}.\end{aligned}
$$
通过带约束的优化问题生成更可靠的“软标签”（soft pseudo-labels）
$q_{iy}$：待求解的软标签（优化变量）
第一项：$-\sum_{i=1}^N \sum_{y=1}^K q_{iy}\log p_{iy}$ 让软标签 $q$ 与模型预测给出的伪标签 $p$ 保持一致
第二项：$\sum_{i=1}^N \sum_{y=1}^K q_{iy}\log q_{iy}$ 增大 $q_{ij}$ 的熵，让 $q_{ij}$ 更平滑、不极端
平衡项：$\lambda$ 越大，软标签越均匀（越不确定）

约束条件：
- $\sum_{y=1}^K q_{iy} = 1$ $\longrightarrow$ 软标签属于各类别的概率之和为 1
- $\sum_{i=1}^N q_{iy} = \frac{N}{K}$ $\longrightarrow$ 对于每种类别 $y$，分配给该类别的概率质量为 $\frac{N}{K}$，等分约束。若数据集类别平衡，这是自然约束

为何要施加等分约束？
考虑对于单个样本 $i$ 的原始问题：
$$
\begin{aligned}
\min_{q_{i}}L_i = -\sum_{y=1}^Kq_{iy}\log p_{iy}+\lambda\sum_{y=1}^Kq_{iy}\log q_{iy}
\end{aligned}
$$
约束：
$$\sum_{y=1}^K q_{iy} = 1, q_{iy} \ge 0$$
引入拉格朗日乘子 $\mu_i$：
$$\mathcal{L} = -\sum_y q_{iy}\log p_{iy} + \lambda \sum_y q_{iy}\log q_{iy} - \mu_i(\sum_y q_{iy} - 1)$$
对 $q_{iy}$ 求导：
$$\frac{\partial \mathcal{L}}{\partial q_{iy}} = -\log p_{iy} + \lambda(\log q_{iy} + 1) - \mu_i = 0$$
解得：
$$q_{iy} = \exp\left ( \frac{\mu_i - \lambda}{\lambda} \right ) \cdot p_{iy}^{1/\lambda}$$
所以，如果模型对所有样本的预测 $p_i$ 都偏向同一个类别 $y^*$，所有的软标签也会倾向于该类别
**等分约束用来避免将所有样本分配相同任意标签的退化解**
**一句话总结**：平滑是**每个样本内部**的均匀化，而等分约束是**跨样本**的均匀化。前者无法阻止所有样本都偏向同一类，后者通过全局计数约束强制类别平衡

以上用于伪标签细化的增强自标记问题实际上是最优传输问题的一个实例。转换为矩阵形式：
$[Q]_{iy} = q_{iy}$ 是维度为 $N \times K$ 的标签矩阵，$[P]_{iy} = p_{iy}$ 是维度为 $N \times K$ 的预测概率矩阵。目标可重写为：
$$\min_{Q\in U(r, c)} \left \langle Q, -\log P \right \rangle - \lambda H(Q)$$
$\left \langle \cdot \right \rangle$：逐元素相乘后求和，即原来的交叉熵项
$H(Q)$：熵

约束条件构成“传输多面体”：
$$U(r, c) := \left \{ Q \in \mathbb{R}_+^{N \times K} \mid Q1_K = r, Q^{\top}1_N = c \right \}$$
其中 $r = 1_N$ 表示元素均为 1 的列向量，$Q1_K = r$ 表示每一个样本的软标签的概率之和为 1
$c = \frac{N}{K}1_K$ 及 $Q^{\top}1_N = c$ 表示等分约束

这是一个离散最优传输问题，直观理解：  
你要把 $N$ 个样本的“概率质量”分配到 $K$ 个类别，  
每行输出 1 单位质量，每列恰好接收 $\frac{N}{K}$ 单位质量  

这个问题可以通过 Sinkhorn-Knopp 算法的快速版本来解决

采用条件熵最小化将决策边界推离数据密集区域以支持聚类假设：
$$\mathcal{L}_{ent} = -\frac{1}{N}\sum_{i=1}^N\sum_{y=1}^Kp(y\mid x_i;\theta)\log p(y\mid x_i;\theta)$$
理论上，我们希望最小化总体条件熵，但实际只能用有限的无标签样本估计。这种近似仅在模型**局部Lipschitz**时成立。如果模型变化非常剧烈（例如深度网络过拟合），即使输入轻微变化，输出 $q$ 也会剧烈变化 $\longrightarrow$ 经验熵不稳定，不能代表真实分布
因此，进一步添加虚拟对抗损失作为正则化项，强制模型在输入数据的微小邻域内预测保持稳定，从而**显式地迫使模型满足局部Lipschitz约束**
