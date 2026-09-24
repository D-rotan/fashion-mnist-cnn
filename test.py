import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
all_images = []
all_labels = []
all_pred = []
probs = []
confusion_matrix = torch.zeros(10, 10, dtype=torch.int64)

#设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#数据准备
test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

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
model.load_state_dict(torch.load("models/model.pth"))
model.to(device)

#损失函数准备
loss_fn = nn.CrossEntropyLoss()

#测试开始
model.eval()
with torch.no_grad():
    for X_test,y_test in DataLoader(test_dataset,batch_size=len(test_dataset),shuffle=False):
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        test_logits = model(X_test)
        test_loss = loss_fn(test_logits,y_test)
        test_pred = torch.argmax(test_logits,dim = 1)
        test_accuracy = (test_pred == y_test).float().mean()
        prob = torch.softmax(test_logits,dim = 1)
        probs.append(prob.cpu())
        all_images.append(X_test.cpu())
        all_labels.append(y_test.cpu())
        all_pred.append(test_pred.cpu())
        print(f"Test Loss:{test_loss.item():.4f}")
        print(f"Test Accuracy:{test_accuracy:.4f}")
        for class_id in range(10):
            mask = (y_test == class_id)
            test_accuracy = (test_pred[mask] == y_test[mask]).float().mean()
            print(f"Class{class_id}_Accuracy:{test_accuracy.item():.4f}")
        for true,pred in zip(y_test,test_pred):
            confusion_matrix[true][pred] += 1

#数据可视化
classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
plt.imshow(confusion_matrix,cmap="Blues")
plt.xticks(range(10),classes,rotation=50)
plt.xlabel("Pred class")
plt.yticks(range(10),classes)
plt.ylabel("True class")
plt.title("Confusion matrix")
plt.colorbar()
for i in range(10):
    for j in range(10):
        plt.text(j,i,confusion_matrix[i][j].item(),ha="center",va="center")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.show()