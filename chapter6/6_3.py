from BP import BPNet
import pandas as pd
import torch
from SVM import KernelSVM,predict_k

data = pd.read_csv('iris_dataset.csv')
all_features = pd.get_dummies(data)#dummy_na会把缺失值也单开一列，作为独立的特征（哑变量处理）
all_features = all_features.astype(float)
#print(all_features.iloc[:, -3:])
train_features = torch.tensor(all_features.iloc[:, :-3].values, dtype=torch.float32)
train_labels = torch.tensor(all_features.iloc[:, -3:].values, dtype=torch.float32)
train_labels1 = torch.tensor(all_features.iloc[:, -3].values, dtype=torch.float32)
train_labels1 = torch.where(train_labels1 == 0, torch.tensor(-1.), train_labels1)
train_labels2 = torch.tensor(all_features.iloc[:, -2].values, dtype=torch.float32)
train_labels2 = torch.where(train_labels2 == 0, torch.tensor(-1.), train_labels2)
train_labels3 = torch.tensor(all_features.iloc[:, -1].values, dtype=torch.float32)
train_labels3 = torch.where(train_labels3 == 0, torch.tensor(-1.), train_labels3)

net = BPNet(n_feature=train_features.shape[1])
svm = KernelSVM(input_dim=train_features.shape[1],C=1)#线性核
svm_rbf = KernelSVM(input_dim=train_features.shape[1],C=1,sigma=1)#rbf
svm2 = KernelSVM(input_dim=train_features.shape[1],C=1)#线性核
svm2_rbf = KernelSVM(input_dim=train_features.shape[1],C=1,sigma=1)#rbf
svm3 = KernelSVM(input_dim=train_features.shape[1],C=1)#线性核
svm3_rbf = KernelSVM(input_dim=train_features.shape[1],C=1,sigma=1)#rbf

BP_train_loss = net.optimiser_gd_lr(train_features,train_labels)
BP_predict = net.predict(train_features,train_labels)

svm_train_loss = svm.trainmodel(train_features,train_labels1)
svm_rbf_train_loss = svm_rbf.trainmodel(train_features,train_labels1)
svm2_train_loss = svm2.trainmodel(train_features,train_labels2)
svm2_rbf_train_loss = svm2_rbf.trainmodel(train_features,train_labels2)
svm3_train_loss = svm3.trainmodel(train_features,train_labels3)
svm3_rbf_train_loss = svm3_rbf.trainmodel(train_features,train_labels3)

model = [svm,svm2,svm3]
model_rbf = [svm_rbf,svm2_rbf,svm3_rbf]
predict_k(train_features,train_labels,model_rbf)