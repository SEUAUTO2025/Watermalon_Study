import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA 

class SVMNetwork_Original(nn.Module):#原始的软约束SVM，使用hinge损失函数
    """原始线性SVM（保留用于对比）"""
    def __init__(self,in_shape):
        super(SVMNetwork_Original,self).__init__()
        self.linear = nn.Linear(in_shape,1)
    
    def forward(self,x):
        return self.linear(x)
    
    def calloss(self,x,y,c):#x:(batch_size,in_shape) y:(batch_size,1)
        L2_item = 0.5 * self.linear.weight.pow(2).sum()
        hinge_loss = c*(torch.sum(torch.clamp(1 - y*self.forward(x), min=0)))
        return (L2_item + hinge_loss)
    
    def calloss_kernel(self,x,y,c):
        L2_item = 0.5 * self.linear.weight.pow(2).sum()
        
    
    def optimiser(self,lr,decay,momentum=None):
        if momentum != None:
            return optim.SGD(self.parameters(), lr=lr,weight_decay=decay,momentum=momentum)
        else:
            return optim.AdamW(self.parameters(), lr=lr,weight_decay=decay)
        
    def trainmodel(self,epochs,x_train,y_train,x_val,y_val,c=0.005,lr=0.01,decay=0.0005,momentum=0.9): #c不能设的太高
        #深复制一份，然后断开计算图（如果x,y进来的时候带着梯度，不这样做就会出问题，会更新它们的梯度）
        x_train = x_train.clone().detach()
        y_train = y_train.clone().detach().view(-1, 1)
        x_val = x_val.clone().detach()
        y_val = y_val.clone().detach().view(-1, 1)
        opt = self.optimiser(lr,decay,momentum)
        train_loss = []
        val_loss = []
        for i in range(epochs):
            self.train()
            opt.zero_grad()
            loss = self.calloss(x_train,y_train,c)
            loss.backward()
            opt.step()
            train_loss.append(loss.item())
            
            self.eval()
            with torch.no_grad():
                loss = self.calloss(x_val,y_val,c)
                val_loss.append(loss.item())
            print(f"epoch {i},train_loss:{train_loss[-1]},val_loss:{val_loss[-1]}")
        return train_loss,val_loss
    
    def visualize_svm(self, X, y, losses):
        """可视化SVM决策边界和训练过程"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 左图: 决策边界
        X_np = X.numpy()
        y_np = y.numpy().flatten()

        # 绘制数据点
        ax1.scatter(X_np[y_np == 1, 0], X_np[y_np == 1, 1],
                    c='red', label='Class +1', alpha=0.6, edgecolors='k')
        ax1.scatter(X_np[y_np == -1, 0], X_np[y_np == -1, 1],
                    c='blue', label='Class -1', alpha=0.6, edgecolors='k')

        # 绘制决策边界
        w = self.linear.weight.data.numpy().flatten()
        b = self.linear.bias.data.numpy()[0]

        # 决策边界: w1*x1 + w2*x2 + b = 0
        x1_min, x1_max = X_np[:, 0].min() - 1, X_np[:, 0].max() + 1
        x1 = np.linspace(x1_min, x1_max, 100)
        x2 = -(w[0] * x1 + b) / w[1]

        ax1.plot(x1, x2, 'k-', linewidth=2, label='Decision Boundary')

        # 绘制支持向量边界 (margin)
        x2_margin_plus = -(w[0] * x1 + b - 1) / w[1]
        x2_margin_minus = -(w[0] * x1 + b + 1) / w[1]
        ax1.plot(x1, x2_margin_plus, 'k--', linewidth=1, alpha=0.5)
        ax1.plot(x1, x2_margin_minus, 'k--', linewidth=1, alpha=0.5)

        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        ax1.set_title('SVM Decision Boundary')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 右图: 损失曲线
        ax2.plot(losses, linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()   
        
class KernelSVM(nn.Module):#基于对偶方法的SVM，通过选择合适的核函数可以做高维空间的二分类问题
    def __init__(self,input_dim=1,sigma=None,C=None,d=None):
        super(KernelSVM,self).__init__()
        self.sigma = sigma #高斯核
        self.C = C #软间隔 batch batch
        self.d = d #多项式核
        self.alpha = None #alpha
        self.b = None #基于支持向量计算
    
    def kernel(self,x1,x2):#逐个对每两个样本计算核函数（结果是(batch,batch)）    
        if self.d!=None:#(batch,dim)
            return torch.mm(x1,x2.t()) ** self.d #要的就是逐元素幂，不是矩阵乘法
        elif self.sigma!=None:#x1**2+x2**2-2x1x2()
            dist2 = torch.cdist(x1, x2, p=2) ** 2  # 欧氏距离平方 (batch1, batch2)
            # mask = ~torch.eye(dist2.size(0), dtype=torch.bool)
            # off_diag_elements = dist2[mask]
            # # 求平均
            # mean_value = off_diag_elements.mean()
            self.sigma = 1 #sigma不能变化，不然就丢失核本身的特性了
            return torch.exp(-dist2/(2*self.sigma**2))
            #return torch.exp(-(torch.sum(x1**2,dim=1)+torch.sum(x2**2,dim=1)-2*torch.mm(x1,x2.t()))/(2*self.sigma**2))#(batch,1)+(batch,1)+(batch,batch)
        else:
            return torch.mm(x1,x2.t())
        
    def calloss(self,x,y):
        #标准流程
        x = x.clone().detach()
        y = y.clone().detach().view(-1,1)#避免传进来的是个向量造成运算错误
        # return 0.5 * self.alpha.t() @ y.t() @ self.kernel(x,x) @ y @ self.alpha
        yky = torch.mm(y,y.t())*self.kernel(x,x) #先yTy做矩阵乘法，再逐元素与核矩阵相乘（技巧是只要维度匹配，不丢项，得到的就是唯一正确的解）
        return (0.5 * torch.mm(torch.mm(self.alpha.t(),yky),self.alpha) - torch.sum(self.alpha,dim=0)) #这里是alphaT@yky@alpha
    
    def trainmodel(self,x_train,y_train,epochs=1000):
        x_train = x_train.clone().detach()
        y_train = y_train.clone().detach().view(-1, 1)
        self.alpha = nn.Parameter(torch.zeros(x_train.size()[0],1,requires_grad=True)) #alpha是拉格朗日乘子！数量是样本数！！！
        self.train()
        optimiser = optim.SGD(params=self.parameters(),lr=0.01,momentum=0.9,weight_decay=0) #优化\alpha
        train_loss = []
        for i in range(epochs):
            #更新参数
            self.train()
            optimiser.zero_grad()
            loss = self.calloss(x_train,y_train)
            loss.backward()
            optimiser.step()
            train_loss.append(loss.item())
           
            self.eval()
            with torch.no_grad():
                #首先引入软约束条件，强制满足约束（alpha \of[0,C]）
                self.alpha.clamp_(0,self.C)
                sum_y = torch.sum(self.alpha*y_train) #\sigma alpha_i yi =0
                if(torch.abs(sum_y)>=0.001):
                    alpha_index = (self.alpha>1e-5) & (self.alpha<self.C-1e-5)
                    """ (alpha-corrlation * y)*y = alpha *y - corrlation * mask.sum() = 0,corrlation=sum/mask.sum(),
                    mask.sum()会返回所有需要调整的alpha的个数（True+1,false+0） """
                    corrlation = sum_y / alpha_index.sum()
                    self.alpha[alpha_index] -= corrlation * y_train[alpha_index] 
                    self.alpha.clamp_(0,self.C)
                #寻找支持向量，阈值为1e-4，离边界至少这个距离，在边界范围之内的为支持向量
                support_index = ((self.alpha.view(-1) > 1e-5) & (self.alpha.view(-1) < self.C - 1e-5)).nonzero(as_tuple=True)[0] #记得要转成向量，一维的，方便推导;nonzero会返回结果是True的元组，每个里面是一个向量，取[0]把里面的向量(tensor，可以直接用于提取下标),and，or链式比较操作符不能直接用于多元素张量，要用&
                support_vec = x_train[support_index]
                support_label = y_train[support_index]
                support_alpha = self.alpha[support_index]
                self.support_vec = support_vec
                self.support_label = support_label
                self.support_alpha = support_alpha
                #计算b,b=\frac{1}{|S|} \sum_{s \in S}\left(y_s-\sum_{i \in S} \alpha_i y_i \boldsymbol{x}_i^{\mathrm{T}} \boldsymbol{x}_s\right) .
                self.b = torch.mean(support_label - torch.sum(support_alpha*support_label*self.kernel(support_vec,support_vec),dim=1))#(len(lndex),len(index))->sum(len(index))
            #print(f"epoch {i},train_loss:{loss.item()},bia:{self.b.item():.4f}, alpha_min:{self.alpha.min().item():.4f}, alpha_max:{self.alpha.max().item():.4f}, sum_y:{sum_y.item():.4f}")
        return train_loss
        
    def decision_function(self,x):#\sigma \alpha_i y_ik(x,x_i) + b,y_i,x_i都是从样本中挑出来的的支持向量
        kernel = self.kernel(self.support_vec,x)
        return self.b + torch.sum(self.support_alpha*self.support_label*kernel,dim = 0).view(-1,1)#(sample,b)-> b 
        
    def predict(self, X):
        """预测标签"""
        result = torch.sign(self.decision_function(X)) #最终结果按照正负进行二分类，+1，-1（根据标签来确定）
        print(result)
        return result

    def score(self, X, y):
        """计算准确率"""
        y_pred = self.predict(X).view(-1)       # 展平成 (n,)
        y_true = y.view(-1)                     # 展平成 (n,)
        print(f"pred:{y_pred},true:{y_true}")
        correct = (y_pred == y_true).float()    # 要保证都是向量，要不然可能维度匹配错了
        acc = correct.mean().item()
        return acc

def predict_k(x, y, modelk):
    
    """使用 k 个二分类 SVM 做多分类预测（one-vs-rest 风格）
    方法：
    - 对于每个模型，优先使用 model.decision_function(x) 作为得分（连续值，越大越倾向该类），
      如果没有 decision_function，则使用 model.predict(x) 的 ±1 作为得分回退。
    - 将所有模型的得分堆叠为形状 (n_samples, k) 的张量 scores。
    - 对每一行取 argmax 得到预测的类别索引（0..k-1）。

    返回：
    - preds: (n_samples,) 的整型张量，表示每个样本被预测为哪个类别（模型索引）
    - scores: (n_samples, k) 的浮点张量，表示每个样本在每个模型上的得分
    """
    y_preds = []
    for model in modelk:
        # 优先使用 decision_function 获取连续分数
        with torch.no_grad():
            s = model.decision_function(x).view(-1).float()  # (n,)
        y_preds.append(s)
    #堆叠为 (k, n) -> 转置为 (n, k)
    scores = torch.stack(y_preds, dim=1)  # (n_samples, k)
    # argmax 返回每行得分最大的模型索引
    preds = torch.argmax(scores, dim=1)
    true_labels = torch.argmax(y, dim=1)
    # 把张量移动到 CPU 并转换为一维迭代对象，逐元素比较并计数相等项
    pred_cpu = preds.cpu()
    true_cpu = true_labels.cpu()
    equal_count = 0
    for p, t in zip(pred_cpu, true_cpu):
        if int(p.item()) == int(t.item()):
            equal_count += 1
    print(f"SVM acc:{equal_count/pred_cpu.size()[0]}")
    return preds, scores

# # 生成简单的二分类数据
# def generate_data(n_samples=100,seed1 = 42):
#     """生成线性可分的二分类数据"""
#     np.random.seed(seed1)

#     # 类别1: 均值[2, 2]
#     class1 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
#     # 类别2: 均值[-2, -2]
#     class2 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])

#     # 合并数据
#     X = np.vstack([class1, class2])
#     # 标签: 1 和 -1 (SVM标准标签)
#     y = np.hstack([np.ones(n_samples // 2), -np.ones(n_samples // 2)])

#     return torch.FloatTensor(X), torch.FloatTensor(y).reshape(-1, 1)

# x_train,y_train = generate_data(seed1=42)
# x_val,y_val = generate_data(seed1 = 100)
# net = KernelSVM(in_shape=2)
# train_loss,val_loss = net.trainmodel(x_train=x_train,x_val=x_val,y_train=y_train,y_val=y_val,epochs=1000)
# net.visualize_svm(x_val,y_val,train_loss)

def visualize_kernel_svm(model, X, y, losses, title="Kernel SVM", feature_names=None, use_pca=True):
    """可视化核SVM的决策边界和训练过程

    Args:
        model: 训练好的KernelSVM模型
        X: 特征数据 (n_samples, n_features)
        y: 标签 (n_samples, 1) 或 (n_samples,)
        losses: 训练损失列表
        title: 图表标题
        feature_names: 特征名称列表，用于2D可视化时的轴标签
        use_pca: 当特征维度>2时，是否使用PCA降维可视化
    """
    fig = plt.figure(figsize=(16, 5))

    X_np = X.numpy()
    y_np = y.numpy().flatten()

    # 将标签转换为1和-1（如果是0和1）
    if np.any(y_np == 0):
        y_plot = np.where(y_np == 1, 1, -1)
    else:
        y_plot = y_np

    # 处理高维数据：使用PCA降维到2D进行可视化
    if X_np.shape[1] > 2 and use_pca:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X_np)
        explained_var = pca.explained_variance_ratio_
        xlabel = f'PC1 ({explained_var[0]*100:.1f}% var)'
        ylabel = f'PC2 ({explained_var[1]*100:.1f}% var)'
        print(f"使用PCA降维: {X_np.shape[1]}维 -> 2维，保留方差: {sum(explained_var)*100:.2f}%")
    elif X_np.shape[1] == 2:
        X_2d = X_np
        if feature_names and len(feature_names) >= 2:
            xlabel, ylabel = feature_names[0], feature_names[1]
        else:
            xlabel, ylabel = 'Feature 1', 'Feature 2'
    else:
        # 如果维度大于2且不使用PCA，则只绘制前两个特征
        X_2d = X_np[:, :2]
        xlabel, ylabel = 'Feature 1', 'Feature 2'
        print(f"警告: 仅显示前2个特征，总共{X_np.shape[1]}个特征")

    # 左图: 决策边界
    ax1 = plt.subplot(1, 3, 1)

    # 绘制数据点
    ax1.scatter(X_2d[y_plot == 1, 0], X_2d[y_plot == 1, 1],
                c='red', label='good melon(+1)', alpha=0.6, edgecolors='k', s=80)
    ax1.scatter(X_2d[y_plot == -1, 0], X_2d[y_plot == -1, 1],
                c='blue', label='bad melon(-1)', alpha=0.6, edgecolors='k', s=80)

    # 绘制支持向量（需要投影到2D空间）
    if hasattr(model, 'support_vec') and model.support_vec is not None:
        support_indices = []
        for sv in model.support_vec:
            # 找到原始数据中对应的索引
            for i, x in enumerate(X):
                if torch.allclose(x, sv):
                    support_indices.append(i)
                    break
        if support_indices:
            ax1.scatter(X_2d[support_indices, 0], X_2d[support_indices, 1],
                       s=200, facecolors='none', edgecolors='green', linewidths=2.5,
                       label='Support Vectors')

    # 绘制决策边界（使用原始高维数据进行预测，但在2D空间展示）
    if X_np.shape[1] > 2 and use_pca:
        # 对于PCA降维的情况，在2D空间创建网格，然后反向投影到原始空间
        x1_min, x1_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
        x2_min, x2_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
        xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                               np.linspace(x2_min, x2_max, 100))

        # 反向投影到原始特征空间
        grid_2d = np.c_[xx1.ravel(), xx2.ravel()]
        grid_original = pca.inverse_transform(grid_2d)
        grid_tensor = torch.FloatTensor(grid_original)

        with torch.no_grad():
            Z = model.decision_function(grid_tensor).numpy().reshape(xx1.shape)
    else:
        # 对于2D数据，直接在特征空间绘制
        x1_min, x1_max = X_2d[:, 0].min() - 0.1, X_2d[:, 0].max() + 0.1
        x2_min, x2_max = X_2d[:, 1].min() - 0.1, X_2d[:, 1].max() + 0.1
        xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                               np.linspace(x2_min, x2_max, 100))

        # 如果原始特征多于2个，需要填充其他维度（用均值）
        if X_np.shape[1] > 2:
            grid_2d = np.c_[xx1.ravel(), xx2.ravel()]
            # 添加其他特征的均值
            other_features = np.tile(X_np[:, 2:].mean(axis=0), (grid_2d.shape[0], 1))
            grid_full = np.hstack([grid_2d, other_features])
            grid_tensor = torch.FloatTensor(grid_full)
        else:
            grid_tensor = torch.FloatTensor(np.c_[xx1.ravel(), xx2.ravel()])

        with torch.no_grad():
            Z = model.decision_function(grid_tensor).numpy().reshape(xx1.shape)

    # 决策边界 (f=0) 和间隔 (f=±1)
    ax1.contour(xx1, xx2, Z, levels=[0], colors='k', linewidths=2.5, linestyles='-')
    ax1.contour(xx1, xx2, Z, levels=[-1, 1], colors='gray', linewidths=1.5, linestyles='--', alpha=0.7)
    ax1.contourf(xx1, xx2, Z, levels=[-100, 0, 100], colors=['blue', 'red'], alpha=0.1)

    ax1.set_xlabel(xlabel, fontsize=11)
    ax1.set_ylabel(ylabel, fontsize=11)
    ax1.set_title(title, fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 中图: 损失曲线
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(losses, linewidth=2, color='purple')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Dual Objective', fontsize=11)
    ax2.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # # 右图: α分布
    # ax3 = plt.subplot(1, 3, 3)
    # alpha_np = model.alpha.detach().numpy().flatten()
    # ax3.hist(alpha_np, bins=30, color='orange', alpha=0.7, edgecolor='black')
    # ax3.axvline(x=0, color='r', linestyle='--', linewidth=1, label='α=0')
    # ax3.axvline(x=model.C, color='r', linestyle='--', linewidth=1, label=f'α=C({model.C})')
    # ax3.set_xlabel('α value', fontsize=11)
    # ax3.set_ylabel('Count', fontsize=11)
    # ax3.set_title('Distribution of Dual Variables α', fontsize=12, fontweight='bold')
    # ax3.legend(fontsize=9)
    # ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# def generate_linear_data(n_samples=100, seed=42):
#     """生成线性可分的数据"""
#     np.random.seed(seed)
#     class1 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
#     class2 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
#     X = np.vstack([class1, class2])
#     y = np.hstack([np.ones(n_samples // 2), -np.ones(n_samples // 2)])
#     return torch.FloatTensor(X), torch.FloatTensor(y)


# def generate_circle_data(n_samples=200, seed=42):
#     """生成同心圆数据（非线性可分）"""
#     np.random.seed(seed)

#     # 内圆（正类）
#     r_inner = np.random.rand(n_samples // 2) * 1.5
#     theta_inner = np.random.rand(n_samples // 2) * 2 * np.pi
#     class1 = np.column_stack([
#         r_inner * np.cos(theta_inner),
#         r_inner * np.sin(theta_inner)
#     ])

#     # 外圆（负类）
#     r_outer = np.random.rand(n_samples // 2) * 1.5 + 2.5
#     theta_outer = np.random.rand(n_samples // 2) * 2 * np.pi
#     class2 = np.column_stack([
#         r_outer * np.cos(theta_outer),
#         r_outer * np.sin(theta_outer)
#     ])

#     X = np.vstack([class1, class2])
#     y = np.hstack([np.ones(n_samples // 2), -np.ones(n_samples // 2)])
#     return torch.FloatTensor(X), torch.FloatTensor(y)


# def generate_xor_data(n_samples=200, seed=42):
#     """生成XOR数据（非线性可分）"""
#     np.random.seed(seed)
#     X = np.random.randn(n_samples, 2) * 2
#     y = np.ones(n_samples)
#     y[(X[:, 0] * X[:, 1]) < 0] = -1  # XOR pattern
#     return torch.FloatTensor(X), torch.FloatTensor(y)


if __name__ == '__main__':#西瓜数据集上的测试
    data = pd.read_csv('xigua3_0.csv')
    #data.drop(columns=['编号'],inplace=True)#inplace:直接在原数据上修改
    data['好瓜'].replace(['是', '否'], [1, -1], inplace=True)  # SVM需要1和-1标签
    all_features = pd.get_dummies(data, dummy_na=True)#dummy_na会把缺失值也单开一列，作为独立的特征（哑变量处理）
    all_features.drop(columns=['好瓜'],inplace=True)
    all_features = all_features.astype(float)
    train_features = torch.tensor(all_features.values, dtype=torch.float32)
    train_labels = torch.tensor(data['好瓜'].values, dtype=torch.float32).view(-1, 1)#一维转二维，本来就是二维就不用转换了

    model_poly = KernelSVM(input_dim=train_features.shape[0], C=1,d=2)#, sigma=1)  # 设置sigma:使用高斯核,设置d:使用多项式核，不设置:线性核
    losses_poly = model_poly.trainmodel(train_features, train_labels, epochs=1000)
    accuracy_poly = model_poly.score(train_features, train_labels)
    print(f"Training Accuracy: {accuracy_poly * 100:.2f}%\n")

    visualize_kernel_svm(model_poly, train_features, train_labels, losses_poly,
                        title=f"SVM-watermelon(acc:{accuracy_poly*100:.1f}%)") #PCA因为投影到了2维平面空间，并不一定准确表征分类的结果，acc现实的才是真正的准确率，实验来看，二维多项式核的结果最好，RBF次之，最不好的是线性核