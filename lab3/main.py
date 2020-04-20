import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models
from torchvision import datasets, transforms
from torch.autograd.function import once_differentiable
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import random
from tqdm import tqdm
from PIL import Image
import PIL
from dataloader import RetinopathyDataset
from torch.utils.data import DataLoader

sns.set_style("whitegrid")
device = torch.device("cuda")

torch.cuda.set_device(0)


def cal_acc(model, loader):
    correct = 0
    preds = []
    targets = []
    with torch.no_grad():
        for (data, target) in tqdm(loader):
            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            preds.extend(pred)
            targets.extend(target.view_as(pred))
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    return (correct / len(loader.dataset)) * 100, targets, preds


def plot_confusion_matrix(y_true, y_pred, title):
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    classes = [0, 1, 2, 3, 4]
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(title + "_cfmatrx.png")
    plt.clf()
    plt.cla()
    plt.close()
    
    return ax


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    batch_size = 8
    

    
    augmentation = [
        transforms.RandomCrop(480),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
    ]
    train_dataset = RetinopathyDataset('./data', 'train', augmentation=augmentation)
    test_dataset = RetinopathyDataset('./data', 'test')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)
    
    to_train = False

    if to_train:
        model_names = ["Resnet18", "Resnet50", "Resnet18_pretrain", "Resnet50_pretrain"]
        load_models = [False, False, False, False]
#         model_names = ["Resnet50_pretrain_2", "Resnet50_2"]
        model_names = ["Resnet50_pretrain_23"]#, "Resnet50_2"]
#         model_names = ["Resnet18_pretrain_2", "Resnet18_2"]
        model_names = ['Resnet50', "Resnet18"]
        load_models = [False, False]
        
        for idx, model_name in enumerate(model_names):
            print(model_name)
            if model_name == "Resnet18":
                model = ResNet18().to(device)
                if load_models[idx]:
                    model.load_state_dict(torch.load("./" + model_name + ".pth"))
                iteration = 15
            elif model_name == "Resnet50":
                model = ResNet50().to(device)
                if load_models[idx]:
                    model.load_state_dict(torch.load("./" + model_name + ".pth"))
                iteration = 8
            elif model_name == "Resnet18_pretrain_2":
                if load_models[idx]:
                    model = ResNetPretrain(18, pretrained=False).to(device)
                    model.load_state_dict(torch.load("./" + model_name + ".pth"))
                else:
                    model = ResNetPretrain(18, pretrained=True).to(device)
                iteration = 15

            elif model_name == "Resnet50_pretrain_23":
                if load_models[idx]:
                    model = ResNetPretrain(50, pretrained=False).to(device)
                    model.load_state_dict(torch.load("./" + model_name + ".pth"))
                else:
                    model = ResNetPretrain(50, pretrained=True).to(device)
                iteration = 80
            else:
                print("Error! Cannot recognize model name.")
            
            train_accs = []
            test_accs = []
            max_acc = 0
            model.train(mode=True)
            optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
            for epoch in range(iteration):
                print("epoch:", epoch)
                correct = 0
                for (data, target) in tqdm(train_loader):
                    data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.long)
                    optimizer.zero_grad()
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()
                train_acc = (correct / len(train_loader.dataset)) * 100
                print('train_acc: ', train_acc)
                train_accs.append(train_acc)
                model.train(mode=False)
                test_acc, targets, preds = cal_acc(model, test_loader)
                    
                model.train(mode=True)
                if test_acc > max_acc:
                    max_acc = test_acc
                    torch.save(model.state_dict(), "./" + model_name + ".pth")
                print("test_acc:", test_acc)
                test_accs.append(test_acc)
                if test_acc>=82:
                    break

            print(train_accs)
            print(test_accs)
            plt.plot(train_accs, label="train")
            plt.plot(test_accs, label="test")
            plt.title(model_name)
            plt.legend(loc='lower right')
            plt.savefig(model_name + "_result.png")
            plt.clf()
            plt.cla()
            plt.close()
    
    else:
        model_names = ["./Resnet18_2.pth", "./Resnet50_2.pth", "./Resnet18_pretrain_3.pth", "Resnet50_pretrain_23_82.pth"]#"./Resnet50_pretrain_2.pth"]
        models = [ResNet18().to(device), ResNet50().to(device), ResNetPretrain(18, pretrained=False).to(device), ResNetPretrain(50, pretrained=False).to(device)]
#         model_names = ["./Resnet18.pth", "./Resnet18_pretrain.pth"]
#         models = [ResNet18().to(device), ResNetPretrain(18, pretrained=False).to(device)]
        print("Testing")
        for idx, name in enumerate(model_names):
            print(name[2:-6])
            model = models[idx]
            model.load_state_dict(torch.load(name))
            model.eval()
            acc, targets, preds = cal_acc(model, test_loader)
            targets = torch.stack(targets)
            preds = torch.stack(preds)
            plot_confusion_matrix(targets.cpu().numpy(), preds.cpu().numpy(), name[2:-6])
            
            print("model:", name, ", acc:", acc)


