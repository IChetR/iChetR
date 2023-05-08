import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer, AdamW, get_linear_schedule_with_warmup

# 超参数设置
max_epochs = 10
batch_size = 32
learning_rate = 2e-5

# 模型架构设置
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 加载数据集
train_data = pd.read_csv('train.csv')
val_data = pd.read_csv('val.csv')
test_data = pd.read_csv('test.csv')

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
optimizer = AdamW(model.parameters(),
                  lr = learning_rate, 
                  eps = 1e-8)
loss_fn = nn.CrossEntropyLoss()

# 定义学习率调度器
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)*max_epochs)

# 训练模型
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
for epoch in range(max_epochs):
    for step, batch in enumerate(train_dataloader):
        model.train()
        b_input_ids = batch['input_ids'].to(device)
        b_input_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)
        loss = loss_fn(outputs.logits, b_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()  # 调整学习率

        if step % 100 == 0:
            print(f"Epoch {epoch + 1}/{max_epochs} | Step {step}/{len(train_dataloader)} | Train Loss: {loss.item():.4f}")

    # 在每个 epoch 结束时评估模型
    train_loss, train_accuracy, train_f1 = evaluate(model, train_dataloader)
    val_loss, val_accuracy, val_f1 = evaluate(model, val_dataloader)
    test_loss, test_accuracy, test_f1 = evaluate(model, test_dataloader)

    print(f"Epoch {epoch + 1}/{max_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy*100:.2f}% | Train F1 Score: {train_f1*100:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy*100:.2f}% | Val F1 Score: {val_f1*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_accuracy*100:.2f}% | Test F1 Score: {test_f1*100:.2f}%")

# 定义评价指标
def evaluate(model, dataloader):
    model.eval()
    total_loss, total_accuracy, total_f1 = 0, 0, 0
    predictions , true_labels = [], []
    for batch in dataloader:
        b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch.values())
        with torch.no_grad():
            outputs = model(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
        logits = outputs.logits
        loss = loss_fn(logits, b_labels)
        total_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend(np.argmax(logits, axis=1))
        true_labels.extend(label_ids)
    total_accuracy = accuracy_score(true_labels, predictions)
    total_f1 = f1_score(true_labels, predictions)
    total_loss = total_loss / len(dataloader)
    return total_loss, total_accuracy, total_f1
