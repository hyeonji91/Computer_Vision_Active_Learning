### Library ###
import os
import time

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm

### GPU Setting ###
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else 'cpu')
print(DEVICE)

### Custom Dataset ###
class CUB2011(Dataset):
    def __init__(self, transform, mode='train'):
        self.transform = transform
        self.mode = mode

        if self.mode == 'train':
            self.image_folder = os.listdir('./datasets/train')
        elif self.mode == 'valid':
            self.image_folder = os.listdir('./datasets/valid')
        elif self.mode == 'test':
            self.image_folder = os.listdir('./datasets/test')
    
    def __len__(self):
        return len(self.image_folder)
    
    def __getitem__(self, idx):
        """
        DataLoader 함수에서 image data matrix와 image 이름에 있던 label(class) 번호를 튜플 형태로 반환
        예시: image 파일 이름이 "1_48.jpg"라면 return img에는 pixel 데이터가 matrix 형태로 들어가고
        48(class 번호)가 label로 반환됨
        """
        img_path = self.image_folder[idx]
        img = Image.open(os.path.join('./datasets', self.mode, img_path)).convert('RGB')
        img = self.transform(img)

        label = img_path.split('_')[-1].split('.')[0]
        label = int(label)
        return (img, label)


### Data Preprocessing ###
transforms_train = transforms.Compose([transforms.Resize((448, 488)), transforms.ToTensor()])
transforms_valtset = transforms.Compose([transforms.Resize((448, 488)), transforms.ToTensor()])

train_set = CUB2011(mode='train', transform=transforms_train)
val_set = CUB2011(mode='valid', transform=transforms_valtset)
test_set = CUB2011(mode='test', transform=transforms_valtset)
print(f'Num of each dataset: {len(train_set)}, {len(val_set)}, {len(test_set)}')

BATCH_SIZE = 32
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)


### Model / Optimzier ###
EPOCH = 30
lr = 0.1

model = models.resnet18(pretrained=True)

### Transfer Learning ###
num_features = model.fc.in_features
model.fs = nn.Linear(num_features, 50)
model.to(DEVICE)

optimizer = optim.SGD(model.parameters(), lr=lr)
print('Created a learning model and optimizer')


### Train/Evaluation ###
def train(model, train_loader, optimizer, epoch):
    model.train()

    for i, (image, target) in enumerate(train_loader):
        image, target = image.to(DEVICE), target.to(DEVICE)
        output = model(image)
        optimizer.zero_grad()
        train_loss = F.cross_entropy(output, target).to(DEVICE)

        train_loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f'Train Epoch: {epoch} [{i}/{len(train_loader)}]\tLoss: {train_loss.item():.6f}')

    return train_loss

def evaluate(model, val_loader):
    model.eval()
    eval_loss = 0
    correct = 0

    with torch.no_grad():
        for i, (image, target) in enumerate(val_loader):
            image, target = image.to(DEVICE), target.to(DEVICE)
            output = model(image)

            eval_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    eval_loss /= len(val_loader.dataset)
    eval_accuracy = 100 * correct / len(val_loader.dataset)
    return eval_loss, eval_accuracy

### Main ###
start = time.time()
best = 0
for epoch in range(EPOCH):
    train_loss = train(model, train_loader, optimizer, epoch)
    val_loss, val_accuracy = evaluate(model, val_loader)

    if val_accuracy > best:
        best = val_accuracy
        torch.save(model.state_dict(), "./best_model.pth")
    print(f'[{epoch}] Validation Loss : {val_loss:.4f}, Accuracy : {val_accuracy:.4f}%')

# Test result
test_loss, test_accuracy = evaluate(model, test_loader)
print(f'[FINAL] Test Loss : {test_loss}, Accuracy : {test_accuracy:.4f}%')

end = time.time()
elasped_time = end - start
print(f'Elasped Time: {int(elasped_time/3600)}h, {int(elasped_time/60)}m, {int(elasped_time%60)}s')
