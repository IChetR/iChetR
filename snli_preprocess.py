import pandas as pd

# 加载数据集
train_df = pd.read_csv('snli_1.0_train.txt', sep='\t')
dev_df = pd.read_csv('snli_1.0_dev.txt', sep='\t')
test_df = pd.read_csv('snli_1.0_test.txt', sep='\t')

# 解析字段
train_data = []
for i, row in train_df.iterrows():
    premise = row['sentence1']
    hypothesis = row['sentence2']
    label = row['gold_label']
    train_data.append((premise, hypothesis, label))

dev_data = []
for i, row in dev_df.iterrows():
    premise = row['sentence1']
    hypothesis = row['sentence2']
    label = row['gold_label']
    dev_data.append((premise, hypothesis, label))

test_data = []
for i, row in test_df.iterrows():
    premise = row['sentence1']
    hypothesis = row['sentence2']
    label = row['gold_label']
    test_data.append((premise, hypothesis, label))
    import torch
from transformers import GPT2Tokenizer, GPT2Model

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

def encode_input(premise, hypothesis):
    input_text = premise + ' ' + hypothesis
    input_ids = torch.tensor([tokenizer.encode(input_text)])
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]
    return last_hidden_states[0][0].numpy()
    
train_encoded = [(encode_input(premise, hypothesis), label) for premise, hypothesis, label in train_data]
dev_encoded = [(encode_input(premise, hypothesis), label) for premise, hypothesis, label in dev_data]
test_encoded = [(encode_input(premise, hypothesis), label) for premise, hypothesis, label in test_data]
def encode_label(label):
    if label == 'entailment':
        return [1, 0, 0]
    elif label == 'neutral':
        return [0, 1, 0]
    elif label == 'contradiction':
        return [0, 0, 1]
    else:
        return None

train_data_encoded = [(input_, encode_label(label)) for input_, label in train_encoded]
dev_data_encoded = [(input_, encode_label(label)) for input_, label in dev_encoded]
test_data_encoded = [(input_, encode_label(label)) for input_, label in test_encoded]
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AdamW, get_linear_schedule_with_warmup

train_inputs = torch.tensor([input_ for input_, _ in train_data_encoded])
train_labels = torch.tensor([label for _, label in train_data_encoded])
train_dataset = TensorDataset(train_inputs, train_labels)

dev_inputs = torch.tensor([input_ for input_, _ in dev_data_encoded])
dev_labels = torch.tensor([label for _, label in dev_data_encoded])
dev_dataset = TensorDataset(dev_inputs, dev_labels)

test_inputs = torch.tensor([input_ for input_, _ in test_data_encoded])
test_labels = torch.tensor([label for _, label in test_data_encoded])
test_dataset = TensorDataset(test_inputs, test_labels)

batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 训练模型
# ...