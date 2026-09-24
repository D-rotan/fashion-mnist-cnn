import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

trainloss = []
valloss = []
trainacc = []
valacc = []
#设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#数据引入
train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])#训练集启用数据增强
)
val_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
    ])#验证集不启用数据增强
)

#数据切分
traindataset = Subset(train_dataset, range(50000))
valdataset = Subset(val_dataset, range(50000, 60000))

#模型准备
model = nn.Sequential(
    nn.Conv2d(1,32,kernel_size=3,padding=1,stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Conv2d(32,64,kernel_size=3,padding=1,stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2,stride=2),
    nn.Flatten(),
    nn.Linear(3136,10)
)
model.to(device)

#调整器准备
optimizer = torch.optim.Adam(model.parameters(),
                             lr = 0.001,
                             weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50)

#损失函数准备
loss_fn = nn.CrossEntropyLoss()

#加载器准备
batchsize = 100
train_loader = DataLoader(traindataset,batch_size = batchsize,shuffle=True)
val_loader = DataLoader(valdataset,batch_size = len(val_dataset),shuffle=False)

#开始训练
epochs = 30
best_val_loss = 1000
for epoch in range(epochs):
    train_loss_sum = 0
    val_loss_sum = 0
    train_correct = 0
    train_total = 0
    model.train()
    for X_batch,y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        train_logits = model(X_batch)
        train_loss = loss_fn(train_logits,y_batch)
        train_pred = torch.argmax(train_logits,dim = 1)
        train_loss.backward()
        optimizer.step()
        train_loss_sum += train_loss.item()
        train_correct += (train_pred == y_batch).sum().item()
        train_total += y_batch.size(0)
    scheduler.step()
    train_accuracy = train_correct / train_total
    trainacc.append(train_accuracy)
    #一个训练结束，开始验证
    model.eval()
    with torch.no_grad():
        for X_val,y_val in val_loader:
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            val_logits = model(X_val)
            val_loss = loss_fn(val_logits,y_val)
            val_pred = torch.argmax(val_logits,dim = 1)
            val_accuracy = (val_pred == y_val).float().mean()
            valacc.append(val_accuracy.item())
            val_loss_sum += val_loss.item()
    train_loss_mean = train_loss_sum / len(train_loader)
    trainloss.append(train_loss_mean)
    val_loss_mean = val_loss_sum / len(val_loader)
    valloss.append(val_loss_mean)
    print(f"Epoch:{epoch + 1}/{epochs} | Train Loss:{train_loss_mean:.4f} | Val Loss:{val_loss_mean:.4f} | Train Accuracy:{train_accuracy:.4f} | Val Accuracy:{val_accuracy:.4f}")
    #保存模型参数
    if best_val_loss > val_loss_mean:
        torch.save(model.state_dict(), "models/model.pth")
        best_val_loss = val_loss_mean
print(f"Best model saved. Val Loss: {best_val_loss:.4f}")
#测试部分在另一个程序

#数据可视化
plt.plot(range(1,epochs+1),trainloss,label="train_loss")
plt.plot(range(1,epochs+1),valloss,label="val_loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Train/Val Loss")
plt.tight_layout()
plt.savefig("results/training_loss.png")
plt.show()
plt.plot(range(1,epochs+1),trainacc,label="train_accuracy")
plt.plot(range(1,epochs+1),valacc,label="val_accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Train/Val Accuracy")
plt.tight_layout()
plt.savefig("results/training_accuracy.png")
plt.show()