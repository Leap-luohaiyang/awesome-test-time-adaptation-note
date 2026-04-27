提出测试时模板调整器（T3A），在测试时调整线性分类器（深度神经网络的最后一层）
遵循 optimization-free 调整线性分类器权重：
- 使用在线未标记数据和源预训练分类器为每个类别创建伪原型（pseudo-prototype）
- 根据每个样本到伪原型的距离对其进行分类

T3A 不改变训练阶段，而且可以与任何分类模型一起使用，它只调整表征之上的线性分类器

域泛化算法的基准测试：给定包含 $n_d$ 个域的数据集，通常使用 Leaveone-domain-out 过程，该过程使用单个域作为测试域，其他域作为训练域。该过程重复 n 次，每次都改变测试域

原型定义为最后一层线性分类器的权重：
![e35f423c-85cc-4b3c-85dc-76e98342fb40.png](https://papernote-1394983352.cos.ap-nanjing.myqcloud.com/linear-algebra/20260426135805726.png)

对于时间步 t 到达的一个 batch 的测试数据，通过模型预测得到其伪标签 $\hat{y}$，更新 support set $\mathbb{S}_t^k$：
$$\mathbb{S}_t^k = \begin{cases}
 \mathbb{S}_{t-1}^k \cup \left \{ \frac{f_{\theta }(x)}{\left \| f_{\theta }(x) \right \| }  \right \}  & \text{ if } \hat{y}=y^k \\
 \mathbb{S}_{t-1}^k & else
\end{cases} $$
$\mathbb{S}_0^k = \left \| \frac{\omega_k}{\left \| \omega_k \right \|} \right \|$。对一个 batch 的每个样本重复上述过程更新 support set，然后完成预测：
$$\arg \max_{y_k} \gamma_c(Y = y_k \mid f_\theta(x)) = \frac{\exp(z \cdot c^k)}{\sum_j \exp(z \cdot c^j)}$$
$$c^k = \frac{1}{|\mathbb{S}^k|}\sum_{z \in \mathbb{S}^k_t}z$$
**作者通过实验表明，冻结特征提取器，仅仅调整分类层可以显著提升性能，而且本方法无需 SGD**
