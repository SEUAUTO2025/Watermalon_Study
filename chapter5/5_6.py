from matplotlib import pyplot as plt
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as opt
import pandas as pd
import torch
import matplotlib.pyplot as plt
import math

class BPNet(nn.Module):
    def __init__(self,n_feature):
        super().__init__()
        self.Floor_Linear = nn.Linear(n_feature,50)
        self.Activate=nn.ReLU()
        self.Output = nn.Linear(50,3)
        #self.softmax = nn.Softmax(dim=1)
        nn.init.xavier_normal_(self.Floor_Linear.weight)# Xavier正态分布   
        nn.init.constant_(self.Floor_Linear.bias,val=0)
        nn.init.xavier_normal_(self.Output.weight)# Xavier正态分布   
        nn.init.constant_(self.Output.bias,val=0)
    
    def forward(self,x):#前向传播的时候会自动调用forward方法
        y = self.Floor_Linear(x)
        y = self.Activate(y)
        return self.Output(y)
     
    #标准BP：每次针对一个样例更新
    def optimiser_sgd_lr(self,x,y,lr=0.01,num_epochs=150,batch_size=1):
        costs = []
        loss = nn.CrossEntropyLoss()
        dataset = Data.TensorDataset(x,y)
        iter = Data.DataLoader(dataset,batch_size,shuffle=True)
        for i in range(num_epochs):
            #动态调整学习率
            if(i%10==0):
                lr=lr*0.9
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
    
    def optimiser_sgd(self,x,y,lr=0.01,num_epochs=150,batch_size=1):
        costs = []
        loss = nn.CrossEntropyLoss()
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
    
    def optimiser_gd(self,x,y,lr=0.1,num_epochs=300):
        costs = []
        loss = nn.CrossEntropyLoss()
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
    def optimiser_gd_lr(self,x,y,lr_max=0.1,num_epochs=300):
        costs = []
        loss = nn.CrossEntropyLoss()
        for i in range(num_epochs):
            lr = lr_max*(1+math.cos(i*math.pi/num_epochs))/2 #余弦退火学习率
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
    
data = pd.read_csv('iris_dataset.csv')
all_features = pd.get_dummies(data)#dummy_na会把缺失值也单开一列，作为独立的特征（哑变量处理）
all_features = all_features.astype(float)
#print(all_features.iloc[:, -3:])
train_features = torch.tensor(all_features.iloc[:, :-3].values, dtype=torch.float32)
train_labels = torch.tensor(all_features.iloc[:, -3:].values, dtype=torch.float32)
net1 = BPNet(n_feature=train_features.shape[1])
net2 = BPNet(n_feature=train_features.shape[1])
costs = net1.optimiser_gd_lr(train_features,train_labels)
costs2 = net2.optimiser_gd(train_features,train_labels)

#画图，对比标准BP和增量BP
plt.plot(costs, label='GD_LR (lr=0.01)')
plt.plot(costs2, label='GD (lr=0.01)')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.show()


