在总结现有工作中的各种定义后，首先根据两个关键因素对 TTT 进行分类：
- 在现实的 TTT 设置下，测试样本按顺序流式传输，并且必须在新测试样本到达之前立即做出预测。更具体地，在时间戳 T 到达的测试样本 $X_T$ 的预测不应受到任何后续样本 $\left \{ X_t \right \}_{t=T+1 ... \infty}$ 的影响。在本文中，将顺序流称为**单遍适应协议**，违反此假设的任何其他协议称为**多遍适应（模型可以在推理之前根据多个 epoch 的所有测试数据进行更新）**，
- 一些工作必须修改源域的训练损失，例如引入额外的自监督分支，这会带来额外的开销。在本文中，目标是解决最现实和最具挑战性的 TTT 协议：**不修改训练目标的单遍测试时训练**。类似于 [[TENT FULLY TEST-TIME ADAPTATION  BY ENTROPY MINIMIZATION|Tent]]，不过此处可以访问少量源域信息
  
本文定义的协议称为：sequential test time training（sTTT）

motivation：鼓励测试样本在特征空间中形成集群 $\longrightarrow$  问题：在没有从源域正则化的情况下单独学习目标域中的聚类无法保证有效适应 $\longrightarrow$ 解决方法：
- 通过高斯混合来识别源域和目标域中的簇，每个高斯分量对应一个类别
- 将源域的按类别统计数据作为锚点，最小化 KL 散度作为 sTTT 的训练目标，将目标域集群与锚点进行匹配

#### Anchored Clustering for Test-Time Training
通过高斯混合建模源域和目标域的簇，具体来说，每个类别的簇被建模为一个高斯分布。最小化每个类别的源簇和目标簇之间的 KL 散度

#### Clustering through Pseudo Labeling
使用轻量级的时间一致性伪标签过滤方法——计算随着时间的推移，最大预测概率间的差异
进一步引入一个直接基于后验概率的附加伪标签过滤器

#### Global Feature Alignment
为了利用所有可用的测试样本，将全局目标数据分布与源数据分布对齐

#### Efficient Iterative Updating
迭代更新高斯分布的运行统计数据