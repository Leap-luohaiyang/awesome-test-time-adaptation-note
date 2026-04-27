本文提出一种协作类条件生成对抗网络（Collaborative Class Conditional Generative Adversarial Networks，3C-GAN）
组件：
- 在源域上预训练的预测模型 $C$
- 用于匹配目标分布的判别器 $D$
- 以随机采样标签为条件的生成器 $G$，用于生成有效的目标风格的训练样本

![image.png](https://papernote-1394983352.cos.ap-nanjing.myqcloud.com/tta-note-img/20260417141810617.png)
标准的 GAN 模型仅以噪声向量 $z$ 为条件，文中的 $G$ 进一步以预定义标签 $y$ 为条件 $x_g = G(y, z)$
$D$ 被训练用于区分 $x_g$ 和 $x_t$：
$$\max_{\theta_D} \mathbb{E}_{x_t \sim \mathcal{D_t}}[\log D(x_t)] + \mathbb{E}_{y, z}[\log (1-D(G(y, z)))] $$
同时，$G$ 被更新以通过生成与 $x_t$ 具有相似分布的 $x_g$ 来欺骗 $D$。$G$ 的对抗损失 $\mathcal{l}_{adv}$：
$$l_{adv}(G) = \mathbb{E}_{y,z}[\log D(1 - G(y, z))]$$
虽然该损失模拟了目标分布，但它不能保证与输入标签 $y$ 的语义相似性

语义相似损失 $l_{sem}$ 基于预测模型 $C$ 强制 $x_g$ 和输入标签 $y$ 之间的语义相似度
$$l_{sem}(G) = \mathbb{E}_{y,z}[-y\log p_{\theta_C}(G(y, z))]$$
通过生成的目标风格的实例 $\left \{ x_g, y \right \}$ 可以提升 $C$ 在目标域上的表现。$C$ 和 $G$ 互相协作：增强的 $C$ 可以为 $G$ 提供更准确的指导，而更可靠的生成反过来可以提高 $C$ 的性能

额外添加两项正则化项以提升 $C$ 的表现：

**权重正则化**
防止预测模型 $C$ 的参数偏离源数据集中学习的预训练模型的参数
$$l_{\omega Reg} = \left \| \theta_C - \theta_{C_s} \right \|^2$$
其中 $\theta_{C_S}$ 是在源域上预训练的 $C$ 的参数，该参数是固定的

**基于聚类的正则化**
未标记的目标数据可以用于探索目标域上的判别信息
熵最小化 + 虚拟对抗
$$l_{cluReg} = \mathbb{E}_{x_t \sim \mathcal{D}_t} [-p_{\theta_C}(x_t)\log p_{\theta_C}(x_t)]
+ [KL(p_{\theta_C}(x_t) || p_{\theta_C}(x_t + \tilde{r}))]$$