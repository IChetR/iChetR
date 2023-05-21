import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.transforms import transforms

# ����������ӣ��Ա������ظ�
torch.manual_seed(42)

# ����Ƿ��п��õ�GPU�豸
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ����Ԥ����
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ����ͼ���СΪ224x224����
    transforms.ToTensor(),  # ת��Ϊ����
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ��׼��ͼ��
])

# ����ѵ��������֤��
train_data = torchvision.datasets.ImageFolder('path_to_train_data', transform=transform)
val_data = torchvision.datasets.ImageFolder('path_to_validation_data', transform=transform)

# �������ݼ�����
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32, shuffle=False)

# ����CNNģ��
class FoodClassifier(nn.Module):
    def __init__(self):
        super(FoodClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)  # ������10��ʳ�����
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ����ģ��ʵ��
model = FoodClassifier().to(device)

# ������ʧ�������Ż���
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ѵ��ģ��
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
        train_loss += loss.item() * images.size(0)

    train_accuracy = train_correct / len(train_data)
    train_loss = train_loss / len(train_data)

    # ����֤��������ģ��
    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            val_correct += (predicted == labels).sum().item()
            val_loss += loss.item() * images.size(0)

    val_accuracy = val_correct / len(val_data)
    val_loss = val_loss / len(val_data)

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# ����ģ��
torch.save(model.state_dict(), 'food_classifier.pth')