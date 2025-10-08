from matplotlib import pyplot as plt
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as opt
import pandas as pd
import torch
import matplotlib.pyplot as plt
# 试编程实现标准BP算法和累积BP算法，在西瓜数据集3.0上分别
# 用这两个算法训练一个单隐层网络，并进行比较.

class BPNet(nn.Module):
    def __init__(self,n_feature):
        super().__init__()
        self.Floor_Linear = nn.Linear(n_feature,10)
        self.Activate=nn.ReLU()
        self.Output = nn.Linear(10,1)
        #self.softmax = nn.Softmax(dim=1)
        nn.init.xavier_normal_(self.Floor_Linear.weight)# Xavier正态分布   
        nn.init.constant_(self.Floor_Linear.bias,val=0)
        nn.init.xavier_normal_(self.Output.weight)# Xavier正态分布   
        nn.init.constant_(self.Output.bias,val=0)
    
    def forward(self,x):#前向传播的时候会自动调用forward方法
        y = self.Floor_Linear(x)
        y = self.Activate(y)
        return self.Output(y)
    
    #累积BP：每次针对所有样例更新
    def optimiser_gd(self,x,y,lr=0.1,num_epochs=1000):
        costs = []
        loss = nn.BCEWithLogitsLoss()#不加参数默认是平均损失,多分类才用交叉熵损失
        for i in range(num_epochs):
            y_hat = self.forward(x)
            l = loss(y_hat,y)
            l.backward()
            #参数更新
            for param in self.parameters():
                param.data -= lr * param.grad
            #梯度清零
            costs.append(l.item())
            for param in self.parameters():
                param.grad.zero_()
        return costs
    
    #标准BP：每次针对一个样例更新
    def optimiser_sgd(self,x,y,lr=0.1,num_epochs=1000,batch_size=1):
        costs = []
        loss = nn.BCEWithLogitsLoss()
        dataset = Data.TensorDataset(x,y)
        iter = Data.DataLoader(dataset,batch_size,shuffle=True)
        for i in range(num_epochs):
            for x1,y1 in iter:
                y_hat = self.forward(x1)
                l = loss(y_hat,y1)
                l.backward()
                #参数更新
                for param in self.parameters():
                    param.data -= lr * param.grad
                #梯度清零
                for param in self.parameters():
                    param.grad.zero_()
            y_hat = self.forward(x)
            l = loss(y_hat,y)
            costs.append(l.item())
        return costs
    
data = pd.read_csv('xigua3_0.csv')
#data.drop(columns=['编号'],inplace=True)#inplace:直接在原数据上修改
data['好瓜'].replace(['是', '否'], [1, 0], inplace=True)
all_features = pd.get_dummies(data, dummy_na=True)#dummy_na会把缺失值也单开一列，作为独立的特征（哑变量处理）
all_features.drop(columns=['好瓜'],inplace=True)
all_features = all_features.astype(float)
train_features = torch.tensor(all_features.values, dtype=torch.float32)
train_labels = torch.tensor(data['好瓜'].values, dtype=torch.float32).view(-1, 1)
net1 = BPNet(n_feature=train_features.shape[1])
net2 = BPNet(n_feature=train_features.shape[1])
costs = net1.optimiser_gd(train_features,train_labels)
costs2 = net2.optimiser_sgd(train_features,train_labels)

#画图，标准BP收敛的更快
plt.plot(costs)
plt.plot(costs2)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training Loss')
plt.show()
