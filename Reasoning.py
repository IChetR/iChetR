import torch
from transformers import BertTokenizer, BertModel

# 将模型移动到 GPU 上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertModel.from_pretrained('bert-base-uncased').to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 定义需要进行情感分类的句子
text = "I really enjoyed the movie!"

# 预处理文本并转换为模型输入
input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)]).to(device)
with torch.no_grad():
    # 获取模型的输出
    outputs = model(input_ids)
    # 获取模型的预测结果
    _, predicted = torch.max(outputs[0], dim=1)

# 打印预测结果
labels = ['Negative', 'Neutral', 'Positive']
print(labels[predicted])
