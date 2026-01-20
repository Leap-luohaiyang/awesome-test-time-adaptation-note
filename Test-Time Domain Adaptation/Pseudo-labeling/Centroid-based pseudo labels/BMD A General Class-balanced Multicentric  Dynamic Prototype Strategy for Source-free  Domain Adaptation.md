source-free domain adaptation 中，为了弥补源数据的缺乏，现有方法可分为两大类：
- 基于生成的方法
- 基于自训练的方法

现有策略通过实例级预测结果来实现，这些预测结果存在类别偏差，并且往往会引入噪声标签
<font color="#de7802">因为不同类别在源域和目标域之间的域差距是不同的</font>
作者认为：由于域差距，每个类别的粗略单中心特征原型无法有效地表示目标数据，并且会引入负迁移，尤其是对于那些难迁移的数据

本文所提方法包含：
- 一种类间平衡采样策略 $\longrightarrow$ 避免易于迁移的类在原型生成上逐渐占据主导地位 $\longrightarrow$ 聚合潜在和代表性的样本
- 一种类内多中心聚类策略 $\longrightarrow$ 减少难迁移样本的噪声标签 $\longrightarrow$ 为每个类别生成多个特征原型，以分配更稳健和更精确的伪标签
- 动态伪标记策略 $\longrightarrow$ 现有的在固定训练周期更新伪标签库的策略可能无法有效利用网络优化的动态信息 $\longrightarrow$ 在模型适应过程中纳入网络更新信息

##### 类间平衡采样
目标域中易于迁移的类别会获得相对更高的预测置信分数
预测置信度越高的样本越有可能被预测正确，对于类别 $k$，选出目标域中在该类别上的预测置信度最高的前 M 个样本，然后根据这些样本的特征构建类平衡特征原型并分配伪标签
$$
\begin{aligned}\mathcal{M}_{k}&\begin{aligned}&=\arg\max_{\begin{array}{c}x_t\in\mathcal{X}_t\\|\mathcal{M}_k|=M\end{array}}\delta_k(\hat{f}_t(x_t)),\end{aligned}\\\mathrm{c}_k&=\frac{1}{M}\sum_{i\in\mathcal{M}_k}\hat{g}_t(x_t^i),\\\hat{y}_{t}&=\underset{k}{\operatorname*{\operatorname*{\arg\min}}}D_f(\hat{g}_t(x_t),c_k).
\end{aligned}
$$
其中 $M=\max\{1,\lfloor\frac{n_t}{r\times K}\rfloor\}$，$r$ 控制选择比例，K 为类别数量

该策略跨类别平衡的原理图示：
![image.png](https://papernote-1394983352.cos.ap-nanjing.myqcloud.com/tta-note-img/20260120173416475.png)

然后，可以根据样本和类原型间的相似性迭代，以获得更稳定的原型和伪标签
$$\begin{aligned}\mathcal{M}_{k}&=\arg\max_{\begin{array}{c}x_t\in\mathcal{X}_t\\|\mathcal{M}_k|=M\end{array}}\frac{\exp\left(\hat{g}_t(x_t)\cdot c_k\right)}{\sum_{j=1}^K\exp\left(\hat{g}_t(x_t)\cdot c_j\right)},\\\mathrm{c}_k&=\frac{1}{M}\sum_{i\in\mathcal{M}_k}\hat{g}_t(x_t^i),\\\hat{y}_{t}&=\underset{k}{\operatorname*{\operatorname*{\arg\min}}}D_f(\hat{g}_t(x_t),c_k).\end{aligned}$$

##### 类内多中心原型
粗略的单中心特征原型可能无法有效地表示那些模糊的数据，甚至引入负迁移
假设第 $k$ 个类别的采样数据实例为 $\mathcal{X}_{t}^k$，预定义特征原型的数量为 S
通过 k-means 实现类内的聚类
直接将 S 个聚类质心 $\left \{ c_k^i \right \}_{i=1}^S$ 表示为第 $k$ 个类别的类内多特征原型，并分配伪标签：
$$\hat{y}_t=\arg\max_k\frac{\max_{1\leq i\leq S}(\exp(\hat{g}_t(x_t)\cdot c_k^i))}{\sum_{j=1}^K\max_{1\leq i\leq S}(\exp(\hat{g}_t(x_t)\cdot c_j^i))},$$
可以像前文一样迭代这个过程以获得更稳定的原型和伪标签
$$\begin{aligned}&\mathcal{M}_k=\arg\max_{\begin{array}{c}x_t\in \mathcal{X}_t\\|\mathcal{M}_k|=M\end{array}}\frac{\max_{1\leq i\leq S}(\exp(\hat{g}_t(x_t)\cdot c_k^i))}{\sum_{j=1}^K\max_{1\leq i\leq S}(\exp(\hat{g}_t(x_t)\cdot c_j^i))},\\&\{c_k^i\}_{i=1}^S=Kmeans(\hat{g}_t(x_t^n)),\\&\hat{y}_t=\arg\max_k\frac{\max_{1\leq i\leq S}(\exp(\hat{g}_t(x_t)\cdot c_k^i))}{\sum_{j=1}^K\max_{1\leq i\leq S}(\exp(\hat{g}_t(x_t)\cdot c_j^i))}.\end{aligned}$$

##### 动态伪标签
现有策略在固定的训练周期更新伪标签库，这在优化过程中可能无法有效地利用更新的网络
在 epoch 开始的时候，从全局角度更新每个类的多特征原型以及每个实例的相应伪标签
通过指数移动平均更新特征原型并获得动态伪标签
$$\begin{aligned}&\hat{y}_t^d=\frac{\max_{1\leq i\leq S}(\exp(\hat{g}_t(x_t)\cdot c_k^i))}{\sum_{j=1}^K\max_{1\leq i\leq S}(\exp(\hat{g}_t(x_t)\cdot c_j^i))},\\&p_{k}^{i}(x_{t}^{n})=\frac{\exp(\hat{g}_{t}(x_{t}^{n})\cdot c_{k}^{i})}{\sum_{j=1}^{K}\sum_{s=1}^{S}\exp(\hat{g}_{t}(x_{t}^{n})\cdot c_{j}^{s})},\\&\hat{c}_k^i=\frac{\sum_{n=1}^N\hat{g}_t(x_t^n)\cdot p_k^i(x_t^n)}{\sum_{n=1}^Np_k^i(x_t^n)},\\&c_k^i\leftarrow\lambda c_k^i+(1-\lambda)\hat{c}_k^i.\end{aligned}$$
$p_k^i(x_t)$ 表示 $x_t$ 和特征原型的相似性，$\hat{c}_k^i$ 表示当前 batch 计算的类别 $k$ 的第 $i$ 个特征原型

采用交叉熵损失的鲁棒变体——对称交叉熵损失来增强对噪声伪标签的容忍度
$$\begin{aligned}\mathcal{L}_{dym}&=-\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K}\hat{y}_{t,k}^{d}\log\delta_{k}(f_{t}(x_{t}^{i})))-\\&\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K}\delta_{k}(f_{t}(x_{t}^{i})))\log\hat{y}_{t,k}^{d},\end{aligned}$$



