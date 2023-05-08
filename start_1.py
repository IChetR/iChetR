import os
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertForSequenceClassification, BertTokenizer

# 超参数设置
max_epochs = 10
batch_size = 32
learning_rate = 2e-5

# 模型架构设置
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 加载数据集
data_dir = './data'
train_file = os.path.join(data_dir, 'train.csv')
val_file = os.path.join(data_dir, 'val.csv')
test_file = os.path.join(data_dir, 'test.csv')

train_data = pd.read_csv(train_file)
val_data = pd.read_csv(val_file)
test_data = pd.read_csv(test_file)

# 预处理数据
train_encodings = tokenizer(list(train_data['text']), truncation=True, padding=True)
val_encodings = tokenizer(list(val_data['text']), truncation=True, padding=True)
test_encodings = tokenizer(list(test_data['text']), truncation=True, padding=True)

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, list(train_data['label']))
val_dataset = SentimentDataset(val_encodings, list(val_data['label']))
test_dataset = SentimentDataset(test_encodings, list(test_data['label']))

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义优化器和损失函数
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
loss_fn = nn.CrossEntropyLoss()

# 定义学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# 定义 TensorBoard
log_dir = './logs'
writer = SummaryWriter(log_dir=log_dir)

# 将模型移动到 GPU 上(如果可用)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# 训练模型
for epoch in range(max_epochs):
    # 训练模式
    model.train()
    train_loss = 0.0
    train_accuracy = 0.0
    train_f1 = 0.0
    total_steps = len(train_dataloader)
    for step, batch in enumerate(train_dataloader):
        # 将数据移动到 GPU 上(如果可用)
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        loss = loss_fn(outputs.logits, b_labels)

        # 反向传播
        loss.backward()

        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # 更新参数
        optimizer.step()

        # 更新学习率
        scheduler.step()

        # 统计指标
        predictions = np.argmax(outputs.logits.detach().cpu().numpy(), axis=1)
        true_labels = b_labels.detach().cpu().numpy()
        train_loss += loss.item()
        train_accuracy += accuracy_score(true_labels, predictions)
        train_f1 += f1_score(true_labels, predictions)

        # 输出训练信息
        if (step + 1) % 100 == 0 or (step + 1) == total_steps:
            train_loss /= (step + 1)
            train_accuracy /= (step + 1)
            train_f1 /= (step + 1)
            print(f"Epoch {epoch + 1}/{max_epochs} | Step {step + 1}/{total_steps} | Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy*100:.2f}% | Train F1 Score: {train_f1*100:.2f}%")

    # 在每个 epoch 结束时评估模型
    train_loss, train_accuracy, train_f1 = evaluate(model, train_dataloader)
    val_loss, val_accuracy, val_f1 = evaluate(model, val_dataloader)
    test_loss, test_accuracy, test_f1 = evaluate(model, test_dataloader)

    # 输出评估信息
    print(f"Epoch {epoch + 1}/{max_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy*100:.2f}% | Train F1 Score: {train_f1*100:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy*100:.2f}% | Val F1 Score: {val_f1*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy*100:.2f}% | Test F1 Score: {test_f1*100:.2f}%")

    # 写入 TensorBoard
    writer.add_scalar("train/loss", train_loss, epoch)
    writer.add_scalar("train/accuracy", train_accuracy, epoch)
    writer.add_scalar("train/f1_score", train_f1, epoch)
    writer.add_scalar("val/loss", val_loss, epoch)
    writer.add_scalar("val/accuracy", val_accuracy, epoch)
    writer.add_scalar("val/f1_score", val_f1, epoch)
    writer.add_scalar("test/loss", test_loss, epoch)
    writer.add_scalar("test/accuracy", test_accuracy, epoch)
    writer.add_scalar("test/f1_score", test_f1, epoch)

# 定义评价指标
def evaluate(model, dataloader):
    model.eval()
    total_loss, total_accuracy, total_f1 = 0.0, 0.0, 0.0
    predictions , true_labels = [], []
    for batch in dataloader:
        # 将数据移动到 GPU 上(如果可用)
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)

        # 前向传播
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            loss = loss_fn(outputs.logits, b_labels)

        # 统计指标
        predictions.extend(np.argmax(outputs.logits.detach().cpu().numpy(), axis=1))
        true_labels.extend(b_labels.detach().cpu().numpy())
        total_loss += loss.item()

    # 计算评价指标
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    loss = total_loss / len(dataloader)

    return loss, accuracy, f1
