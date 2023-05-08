import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=32 * 8 * 8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


def train(device, train_path, test_path, save_path, lr, epochs, batch_size):
    # 数据预处理
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = CIFAR10(root=train_path, train=True, transform=transform, download=True)
    test_dataset = CIFAR10(root=test_path, train=False, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 定义模型、损失函数和优化器
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练模型
    for epoch in range(epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            total_correct = 0
            total_loss = 0
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                total_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct += (predicted == labels).sum().item()

            accuracy = total_correct / len(test_dataset)
            print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch+1, epochs, total_loss/len(test_loader),
                                                                         accuracy))

    # 保存模型
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for training (cpu or cuda)')
    parser.add_argument('--train_path', type=str, default='./data/train', help='Path to training dataset')
    parser.add_argument('--test_path', type=str, default='./data/test', help='Path to test dataset')
    parser.add_argument('--save_path', type=str, default='./saved_models', help='Path to save trained model')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    args = parser.parse_args()

    device = torch.device(args.device)
    train(device=device, train_path=args.train_path, test_path=args.test_path, save_path=args.save_path, lr=args.lr,
          epochs=args.epochs, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
