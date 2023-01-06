Step1. 运行demo.py 72-91行获得三分支网络和图卷积网络的预测结果

Step2. 将预测结果转为matlab文件格式，并使用./PostProcess/sup_relabel.m生成伪标签，./PostProcess/rand_samples.m抽取合适数量的伪标签获得out.m

Step3. 运行demo.py 93-102行实现半监督训练
