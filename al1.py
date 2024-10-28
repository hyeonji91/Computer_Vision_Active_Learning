import torchvision.transforms as transforms
import torchvision.models as models
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import glob
import cv2


from tqdm import tqdm
import pandas as pd

from customDataset import CustomDataset


### GPU setting ###
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.manual_seed(777)
print(device)

#파이토치의 랜덤시드 고정
import random
torch.manual_seed(0)
random.seed(0)

### Data Preprocessiong ###
BATCH_SIZE = 32
custom_transforms = transforms.Compose([transforms.Resize((448,448)),
                                        # transforms.CenterCrop((128,512)),
                                        transforms.ToTensor()])

train_set = CustomDataset(mode = "train",
                          transform=custom_transforms)
test_set = CustomDataset(mode = "test",
                          transform=custom_transforms)
valid_set = CustomDataset(mode = "valid",
                          transform=custom_transforms)
print("num of each dataset : ", len(train_set), len(test_set), len(valid_set))


#std와 mean 구하기
def get_mean_std(imageFolder):
    meanRGB = [np.mean(image.numpy(), axis = (1,2)) for image,_ in imageFolder]
    stdRGB = [np.std(image.numpy(), axis = (1,2)) for image,_ in imageFolder]

    # 각 채널별의 평균 뽑기
    meanR = np.mean([m[0] for m in meanRGB])
    meanG = np.mean([m[1] for m in meanRGB])
    meanB = np.mean([m[2] for m in meanRGB])

    stdR = np.mean([s[0] for s in stdRGB])
    stdG = np.mean([s[1] for s in stdRGB])
    stdB = np.mean([s[2] for s in stdRGB])

    return [[meanR, meanG, meanB], [stdR, stdG, stdB]]


train_mean, train_std = get_mean_std(train_set)
test_mean, test_std = get_mean_std(test_set)
valid_mean, valid_std = get_mean_std(valid_set)

print(f"test mean ; {test_mean}")
print(f"test std : {test_std}")

# normalization
normal_train_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Resize((448, 448)),
                                             transforms.Normalize(train_mean, train_std)
])
normal_test_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((448, 448)),
                                            transforms.Normalize(test_mean, test_std)
])
normal_valid_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((448, 448)),
                                            transforms.Normalize(valid_mean, valid_std)
])
train_set = CustomDataset(mode = "train",
                          transform=normal_train_transform)
test_set = CustomDataset(mode = "test",
                          transform=normal_test_transform)
valid_set = CustomDataset(mode = "valid",
                          transform=normal_valid_transform)

train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)
test_loader = DataLoader(test_set, batch_size = 1, shuffle=False)



### model ###
epoch = 30
lr = 0.1
model = models.resnet18(pretrained=True)

### transfer learning ###
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 50)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = lr)
lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma= 0.1) # 50 step마다 0.5곱하기
criterion = nn.CrossEntropyLoss().to(device)

from torchsummary import summary
summary(model, input_size=(3, 448, 448), device=device)



model.train()
for epoch in range(1, epoch+1):
    avg_cost = 0.
    train_progress = 0
    for batch_idx, (x_train, y_train) in enumerate(train_loader):
        x_train, y_train = x_train.to(device), y_train.to(device)

        optimizer.zero_grad()
        prediction = model(x_train)
        cost = criterion(prediction, y_train)
        cost.backward()
        optimizer.step()
        lr_sche.step()

        avg_cost += cost.item()
        train_progress += len(x_train)

        print("Train epoch : {} [{}/{}], learning cost {:.2f}, avg cost {:.2f}".format(
            epoch, train_progress, len(train_loader.dataset),
            cost.item(),
            avg_cost / (batch_idx + 1)
        ))          

    with torch.no_grad():
        model.eval()
        correct = 0.
        cost = 0.
        for batch_idx, (x_valid, y_valid) in enumerate(valid_loader):
            x_valid, y_valid = x_valid.to(device), y_valid.to(device)


            output = model(x_valid)
            prediction = torch.argmax(output, 1)
            cost += criterion(prediction, y_train)

            correct += (prediction == y_valid).sum().item()

        accuracy = 100. * correct / len(valid_loader.dataset)
        print("Test accuracy : {}% [{}/{}] \t Test cost : {:.2f}"
                .format(accuracy, correct, len(valid_loader.dataset), cost/len(valid_loader.dataset)))
            
print('Finished Training')



with torch.no_grad():
    model.eval()
    confusion_matrix = np.zeros(shape=(4,4))
    for batch_idx, data in enumerate(test_loader_all):
        x_test, y_test, path = data
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        output = model(x_test)
        prediction = torch.argmax(output, 1)

        incorrect_indices = np.where(y_test != prediction)[0]
        print("Incorrect indices:", incorrect_indices)

        cell = int(np.round(np.sqrt(len(incorrect_indices))))
        fig, axs = plt.subplots(cell, cell, constrained_layout=True, figsize=(cell * 4, cell * 4))
        i = 0
        for idx in incorrect_indices:
            confusion_matrix[y_test[i], prediction[i]] += 1
            print(idx)
            print(y_test[idx])

            if (y_test[idx] != prediction[idx]):
                print(idx)
                print(y_test[idx])
                img = x_test[idx].cpu().numpy().transpose((1, 2, 0))
                # 이미지 정규화 해제하기
                img = (test_std * img + test_mean)
                img = np.clip(img, 0, 1)

                # plt.imshow(img)
                # plt.title(
                #     f'[wrong] real {train_set.classes[y_test[idx]]} prediction {train_set.classes[prediction[idx]]}')
                # plt.show()
                row = int(i / cell)
                col = i % cell
                axs[row, col].imshow(img)
                axs[row, col].set_title(
                    f'[wrong] real : {train_set.classes[y_test[idx]]} \tprediction : {train_set.classes[prediction[idx]]} \n '
                    f'[path] : {path[idx]}',
                    size=6)
                i += 1
        plt.savefig('resnet #1.jpg', format='jpeg')
        plt.show()


        print(f'y test : {y_test}')
        print(f'prediction : {prediction}')
        print(confusion_matrix)


