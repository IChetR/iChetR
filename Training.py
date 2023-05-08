import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.utils.tensorboard import SummaryWriter


# 超参数设置
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 5

# 加载数据集
texts = []
labels = []
with open('data/sentiment.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip().split('\t')
        texts.append(line[0])
        labels.append(int(line[1]))

# 数据预处理和特征提取
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def preprocess(text):
    text = text.lower()  # 转换为小写
    tokens = tokenizer.tokenize(text)  # 分词
    tokens = [token for token in tokens if token.isalnum()]  # 移除标点符号
    input_ids = tokenizer.convert_tokens_to_ids(tokens)  # 将文本转换为数字向量
    return input_ids

# 定义数据集
class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        input_ids = preprocess(text)
        return input_ids, label

# 将数据集分为训练集和验证集
train_texts = texts[:int(0.8*len(texts))]
train_labels = labels[:int(0.8*len(labels))]
val_texts = texts[int(0.8*len(texts)):]
val_labels = labels[int(0.8*len(labels)):]

train_dataset = SentimentDataset(train_texts, train_labels)
val_dataset = SentimentDataset(val_texts, val_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 定义模型
class SentimentModel(nn.Module):
    def __init__(self):
        super(SentimentModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(768, 3)  # 输出层为3,分别对应正面、负面和中性
    def forward(self, input_ids):
        outputs = self.bert(input_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        return logits

# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentimentModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 创建 TensorBoard 写入器
writer = SummaryWriter()

for epoch in range(EPOCHS):
    train_loss = 0.0
    train_acc = 0.0
    val_loss = 0.0
    val_acc = 0.0

    # 训练模型
    model.train()
    for batch in train_loader:
        input_ids, labels = batch
        input_ids = torch.stack(input_ids).to(device)
        labels = torch.tensor(labels).to(device)
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * input_ids.size(0)
        _, predictions = torch.max(outputs, 1)
        train_acc += torch.sum(predictions == labels)

    # 验证模型
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            input_ids, labels = batch
            input_ids = torch.stack(input_ids).to(device)
            labels = torch.tensor(labels).to(device)
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * input_ids.size(0)
            _, predictions = torch.max(outputs, 1)
            val_acc += torch.sum(predictions == labels)

    train_loss /= len(train_dataset)
    train_acc = train_acc.float() / len(train_dataset)
    val_loss /= len(val_dataset)
    val_acc = val_acc.float() / len(val_dataset)

    # 将训练和验证指标写入 TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)

    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'.format(epoch+1, EPOCHS, train_loss, train_acc, val_loss, val_acc))

# 关闭 TensorBoard 写入器
writer.close()
