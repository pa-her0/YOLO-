import sys
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Bottleneck, Conv, C2f, SPPF, Detect


# 加载约束训练后的模型
yolo = YOLO("./best.pt")
model = yolo.model

# 遍历模型的所有层，如果某一层是 torch.nn.BatchNorm2d 类型，则提取其权重和偏置的绝对值，分别存入 ws 和 bs 列表，并打印出最大值和最小值 
# 剪枝后，确保BN层的大部分bias足够小(接近于0)，否则重新进行稀疏训练
ws = []
bs = []

RED = "\033[91m"
RESET = "\033[0m"
for name, m in model.named_modules():
    if isinstance(m, torch.nn.BatchNorm2d):
        w = m.weight.abs().detach()
        b = m.bias.abs().detach()
        ws.append(w)
        bs.append(b)
        print('bn name: {}, weight max: {:.10f}, weight min: {}{:.10f}{}, bias max: {:.10f}, bias min: {}{:.10f}{}'.format(
               name, w.max().item(), RED, w.min().item(), RESET, b.max().item(), RED, b.min().item(), RESET))

# 将所有收集到的权重拼接成一个张量，并通过排序选择位于 factor（这里是0.8）位置的权重作为剪枝的阈值。这意味着保留权重值最大的80%
# 具体，对 ws 张量进行降序排序，并返回排序后的值和排序的索引，[0] 选择了排序后的值，从排序后的值中选择一个值作为阈值
factor = 0.6 # 保持率 0.8
ws = torch.cat(ws)
threshold = torch.sort(ws, descending=True)[0][int(len(ws) * factor)]
print('threshold: {:.10f}'.format(threshold))

# conv1的输出作为conv2的输入
# 对conv1的输出通道进行剪枝（在滤波器的维度进行剪枝），
# 同时conv2的输入通道要相应的剪枝（在卷积核的通道维度进行剪枝）
def prune_conv(conv1: Conv, conv2: Conv):
    gamma = conv1.bn.weight.data.detach()
    beta  = conv1.bn.bias.data.detach()
    
    keep_idxs = []
    local_threshold = threshold
    # 逐步降低阈值来确保在剪枝过程中至少保留 8 个通道（如果小于8 Nvidia GPU 会导致利用率很低，影响性能）
    # 防止过度剪枝：如果直接使用初始阈值，可能会导致保留的通道数太少，这可能严重影响模型的能力，导致性能下降
    # 因此，逐步降低阈值可以保证在剪枝过程中至少保留一定数量的通道，从而在减少计算量的同时，尽量维持模型的表达能力
    # 设定通道数下限：设定一个保底的通道数（比如8个）确保模型不会因为剪枝导致过度简化，从而仍然能够保留一些重要的特征
    while len(keep_idxs) < 8:
        # 取按照阈值过滤得到要保留的通道索引
        keep_idxs = torch.where(gamma.abs() >= local_threshold)[0]
        local_threshold = local_threshold * 0.5
        
    n = len(keep_idxs)
    
    print(n / len(gamma) * 100) # 输出保留通道相对于原始通道数的百分比
    
    # 根据保留的通道索引 keep_idxs 更新，只保留剪枝后的通道
    conv1.bn.weight.data = gamma[keep_idxs] # 更新 BN 层的权重
    conv1.bn.bias.data   = beta[keep_idxs] # 更新 BN 层的偏置
    conv1.bn.running_var.data = conv1.bn.running_var.data[keep_idxs] # 更新 BN 层的方差估计值
    conv1.bn.running_mean.data = conv1.bn.running_mean.data[keep_idxs] # 更新 BN 层的均值估计值
    conv1.bn.num_features = n # 更新 BN 层的通道数
    conv1.conv.weight.data = conv1.conv.weight.data[keep_idxs] # 更新卷积层的权重
    conv1.conv.out_channels = n # 更新卷积层的输出通道数
    if conv1.conv.bias is not None: # 如果存在，更新卷积层的偏置
        conv1.conv.bias.data = conv1.conv.bias.data[keep_idxs]

    # 更新与conv1层连接的后续卷积层conv2的 in_channels 和相应的权重，
    # 使得模型的各层结构保持一致，避免因为剪枝导致通道数不匹配的问题
    if not isinstance(conv2, list):
        conv2 = [conv2]
        
    for item in conv2:
        if item is not None:
            if isinstance(item, Conv):
                conv = item.conv
            else:
                conv = item
            # 将卷积层 conv 的输入通道数更新为 n，即之前在 conv1 剪枝后保留的通道数
            # 这是因为 conv2 的输入来自于 conv1 的输出，而 conv1 的输出通道数已经被剪枝，
            # 因此需要同步更新 conv2 的输入通道数
            conv.in_channels = n
            conv.weight.data = conv.weight.data[:, keep_idxs]  # 同步更新对应的保留通道，注意是在输入通道维度（卷积核的通道维度）进行剪枝
    
def prune(m1, m2):
    if isinstance(m1, C2f):      # C2f as a top conv
        m1 = m1.cv2
    
    if not isinstance(m2, list): # m2 is just one module
        m2 = [m2]
        
    for i, item in enumerate(m2):
        if isinstance(item, C2f) or isinstance(item, SPPF):
            m2[i] = item.cv1
    
    prune_conv(m1, m2)

### 1. 剪枝c2f 中的Bottleneck
for name, m in model.named_modules():
    if isinstance(m, Bottleneck):
        prune_conv(m.cv1, m.cv2)

### 2. 指定剪枝不同模块之间的卷积核       
seq = model.model
for i in range(3, 9): 
    if i in [6, 4, 9]: continue
    prune(seq[i], seq[i+1])

### 3. 对检测头进行剪枝
detect:Detect = seq[-1]
last_inputs   = [seq[15], seq[18], seq[21]]
colasts       = [seq[16], seq[19], None]
for last_input, colast, cv2, cv3 in zip(last_inputs, colasts, detect.cv2, detect.cv3):
    prune(last_input, [colast, cv2[0], cv3[0]])
    prune(cv2[0], cv2[1])
    prune(cv2[1], cv2[2])
    prune(cv3[0], cv3[1])
    prune(cv3[1], cv3[2])

# ***step4，一定要设置所有参数为需要训练。因为加载后的model他会给弄成false。导致报错
# pipeline：
# 1. 为模型的BN增加L1约束，lambda用1e-2左右
# 2. 剪枝模型，比如用全局阈值
# 3. finetune，一定要注意，此时需要去掉L1约束。最终final的版本一定是去掉的

for name, p in yolo.model.named_parameters():
    p.requires_grad = True

torch.save(yolo.ckpt, "prune1.pt")
yolo.export(format="onnx", simplify=True)
print("done")
