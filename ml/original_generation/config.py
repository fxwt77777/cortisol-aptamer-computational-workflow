# src/config.py

# 序列最大长度（我们已经过滤在 50~65，所以直接用 65）
L_MAX = 65

# DNA 碱基字典
NUC_VOCAB = ["A", "C", "G", "T"]

# 分类类别数：weak / medium / strong
NUM_CLASSES = 3

# 训练相关
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-5
N_EPOCHS = 200
EARLY_STOP_PATIENCE = 20

# 模型结构
CONV_FILTERS = 128
KERNEL_SIZES = [6, 8, 10]  # 不同长度的 motif
HIDDEN_DIM = 128
DROPOUT = 0.3

# 多任务：分类 loss 的权重
LAMBDA_CLS = 0.0

# 随机种子
SEED = 42
