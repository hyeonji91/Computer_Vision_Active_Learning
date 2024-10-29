#%%

import torchvision.transforms as transforms
import torchvision.models as models
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import math

import glob
import cv2


from tqdm import tqdm
import pandas as pd

from customDataset import CustomDataset


# to solve matplot error
plt.ion()  # 대화형 모드 활성화


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
BATCH_SIZE = 256
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
# normal_train_transform = transforms.Compose([transforms.ToTensor(),
#                                              transforms.Resize((448, 448)),
#                                              transforms.Normalize(train_mean, train_std)
# ])
# normal_test_transform = transforms.Compose([transforms.ToTensor(),
#                                             transforms.Resize((448, 448)),
#                                             transforms.Normalize(test_mean, test_std)
# ])
# normal_valid_transform = transforms.Compose([transforms.ToTensor(),
#                                             transforms.Resize((448, 448)),
#                                             transforms.Normalize(valid_mean, valid_std)
# ])
# train_set = CustomDataset(mode = "train",
#                           transform=normal_train_transform)
# test_set = CustomDataset(mode = "test",
#                           transform=normal_test_transform)
# valid_set = CustomDataset(mode = "valid",
#                           transform=normal_valid_transform)

train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)
test_loader = DataLoader(test_set, batch_size = 1, shuffle=False)

img = train_set[0][0].permute(1, 2, 0).numpy()
plt.imshow(img)
plt.show()


### model ###
epoch = 50
lr = 0.01
model = models.resnet18(pretrained=True)

### transfer learning ###
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 50)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr = lr)
lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma= 0.1) # 50 step마다 0.5곱하기
criterion = nn.CrossEntropyLoss().to(device)

from torchsummary import summary
summary(model, input_size=(3, 448, 448), device=device)


start = time.time()
val_accuracy = 0.
best_accuracy = 0.
accuracy_his = []
model.train()
for epoch in tqdm(range(1, epoch+1)):
    avg_cost = 0.
    train_progress = 0
    for batch_idx, (x_train, y_train) in enumerate(train_loader):
        x_train, y_train = x_train.to(device), y_train.to(device)

        optimizer.zero_grad()
        prediction = model(x_train)
        cost = criterion(prediction, y_train)
        cost.backward()
        optimizer.step()

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
        cost = 0
        for batch_idx, (x_valid, y_valid) in enumerate(valid_loader):
            x_valid, y_valid = x_valid.to(device), y_valid.to(device)

            output = model(x_valid)
            prediction = torch.argmax(output, 1)
            cost += criterion(output, y_valid)

            correct += (prediction == y_valid).sum().item()

        val_accuracy = 100. * correct / len(valid_loader.dataset)
        accuracy_his.append(val_accuracy)
        print("Test accuracy : {:.2f}% [{}/{}] \t Test cost : {:.2f}"
                .format(val_accuracy, correct, len(valid_loader.dataset), cost/len(valid_loader.dataset)))
    
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), 'model/al1_model.pth')
    lr_sche.step()


end = time.time()
elasped_time = end - start

print("Elasped Time : {}h, {}m, {}s"
      .format(int(elasped_time/3600),
              int(elasped_time/60),
              int(elasped_time%60)))
print('Finished Training')

# plot accuracy history
plt.plot(accuracy_his, label=f"Accuracy")
plt.title(f"Accuracy")
plt.show()



loaded_model = models.resnet18(pretrained=True)
num_features = loaded_model.fc.in_features
loaded_model.fc = nn.Linear(num_features, 50)
loaded_model.to(device)
loaded_model.load_state_dict(torch.load(f'model/al1_model.pth', weights_only=True))
cur_row = 0
cur_col = 0

# plot gradcam
col = math.ceil(math.sqrt(len(test_loader.dataset)))
row = col
# fig, axs = plt.subplots(row, col, constrained_layout=True, figsize=(18, 18))
fig, axs = plt.subplots(2, 2, constrained_layout=True, figsize=(18, 18))
print('start eval')

#with torch.no_grad():
loaded_model.eval()
correct = 0.
num = 0.
# confusion_matrix = np.zeros(shape=(50,50))
for batch_idx, data in enumerate(test_loader):
    x_test, y_test = data
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    output = loaded_model(x_test)
    prediction = torch.argmax(output, 1)
    correct += (prediction == y_test).sum().item()

    if(batch_idx == 0):
    # gradcam
        with GradCAM(loaded_model) as cam_extractor:
            output = loaded_model(x_test)
            print(x_test.size())
            img = x_test.squeeze(0) # batch c h w -> c h w
            # heat map 출력
            activation_map = cam_extractor(output.squeeze(0).argmax().item(), output)
            # Resize the CAM and overlay it
            result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(), mode='F'), alpha=0.7)
            axs[cur_row, cur_col].imshow(result)
            axs[cur_row, cur_col].axis('off')
            num +=1
        cur_row = int(num / row)
        cur_col = int(num % row)

fig.suptitle("resnet")
plt.savefig(f'al1_grad_cam .jpg', format='jpeg')
plt.show()
print("[Test] accuracy : {:.2f}% [{}/{}]".format(100.*correct/len(test_loader.dataset), correct, len(test_loader.dataset)))





# %%
