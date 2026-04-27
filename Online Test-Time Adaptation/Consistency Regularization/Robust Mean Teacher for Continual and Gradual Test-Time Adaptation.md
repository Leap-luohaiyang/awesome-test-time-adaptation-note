### Robust Mean Teacher
通过使用网络的预测作为伪标签来更新自身来进行无监督域适应或测试时适应已被证明是非常有效的
但只有当伪标签是可靠的时候，这种策略才有效
平均教师（Mean Teachers）已经被证实能够提供比学生模型更准确的预测

在 t = 0 时刻，使用源预训练权重 $\theta_0$ 初始化学生和教师模型
在测试阶段，学生 $f_{\theta_t}$ 通过最小化交叉熵更新：
$$\mathcal{L}_{CE}(q, p) = -\sum_{c=1}^Cq_c \log p_c$$
p：学生的 softmax 预测
q：教师的 softmax 预测

教师的权重 $\theta_t^{'}$ 是不训练的，使用指数移动平均更新：$\theta_{t+1}^{'} = \alpha \theta_t^{'} + (1-\alpha)\theta_{t+1}$
