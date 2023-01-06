# 保存模型的目录和名称

modelname = "model/MUUFL_fusion_weights.h5"
modelname_f = "model/MUUFL_fusion_weights_f.h5"
model_at = "save/at.h5"
model_ec = "save/ec.h5"

# 数据
SAVA_PATH = './Image/file/'  # 训练集，验证集，测试集numpy数组保存路径
PATH = './Image/MUUFL'  # 源数据保存路径
HSIName = 'Data.mat'  # 高光谱数据
LiDARName = 'lidar.mat'  # 激光雷达数据
gth_train = 'train.mat'  # 训练集数据的gth
gth_test = 'test.mat'  # 测试集数据的gth
gth = 'gth.mat'
# 参数
'''

 Num  ：  数据集类别总数
 r  ：  补0扩充半径
 lchn  ：   lidar数据通道数
 hchn  ：   hsi数据通道数
 batch_size : batch_size
 epochs : epoch
'''
NUM_CLASS = 11
r = 6
lchn = 2
hchn = 64
BATCH_SIZE = 128
epochs = 30
num_scale = 1