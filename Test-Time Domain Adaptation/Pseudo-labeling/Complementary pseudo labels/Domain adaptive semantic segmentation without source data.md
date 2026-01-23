两个组件：
- positive learning：根据类内阈值，选择类别平衡伪标签像素，防止“赢家通吃”现象
- negative learning：根据所提启发式互补标签选择（heuristic complementary label selection），对于每个像素，探索其**不属于**哪个类别

在密集型语义分割任务中，基于伪标签的模型容易出现“赢家通吃”现象，即模型倾向于过拟合多数类而忽略少数类

#### Positive Learning
选择高置信度的伪标签能够有效去除错误伪标签，但也会陷入“赢家通吃”现象

![image.png](https://papernote-1394983352.cos.ap-nanjing.myqcloud.com/tta-note-img/20260123163425710.png)

选择类内置信度较高的像素（过滤噪声标签），为了避免选择不平衡，类内阈值定义为：
$$
\delta^{(c)} = \top_{\alpha}(\mathcal{P}_T^{(c)} ),
$$
$\top_{\alpha}$ 表示对于类别 $c \in C$ 的 softmax 预测值的集合 $\mathcal{P}_T^{(C)}$ 中排名前 $\alpha \%$ 的值，然后对于每种类别 $c$，选择 softmax 置信度大于 $\delta^{(c)}$ 的那些样本

在每个 epoch 开始的时候更新伪标签，并使用这些伪标签和对应样本通过交叉熵损失更新网络

对于未被选择的伪标签，最小化它们的预测熵

#### Negative Learning
为避免错误伪标签，提出 negative learning
思想：与其确定像素属于哪个类别，不如推断它不属于哪个类别，这更容易实现
具体做法：对于输入 $x \in \mathcal{X}_T$ 和 $\hat{y}$，生成一个互补标签 $\overline{y} \in \overline{\mathcal{Y}}_T (\overline{y} \ne \hat{y})$ 并优化以下损失：
$$\mathcal{L}_{neg}=-\sum_{(x,\overline{y})\in(\mathcal{X}_T,\overline{\mathcal{Y}}_T)}\bar{y}\log(1-S(x,\theta))$$
优化目标是让模型预测为 $\overline{y}$ 的概率**尽可能低**

什么样的软标签值得优化？看看下面这个图
![image.png](https://papernote-1394983352.cos.ap-nanjing.myqcloud.com/tta-note-img/20260123195415646.png)

这张图的 (x, y) 表示有 y 个样本的真实标签对应于其第 x 高的 softmax 预测
head 数据部分：预测概率最高的类别的确最有可能就是样本的真实标签
但是也有相当一部分样本的真实标签并未对应于预测概率最高的类别，如果对这些样本进行负学习会造成错误累计

body 数据部分：聚集了大量被错误预测样本的真实类别