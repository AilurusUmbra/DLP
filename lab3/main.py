import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models
from torchvision import datasets, transforms
import seaborn as sns
from tqdm import tqdm
from dataloader import RetinopathyDataset
from torch.utils.data import DataLoader

from models import ResNetPretrain
from utils import *

sns.set_style("whitegrid")
device = torch.device("cuda")

torch.cuda.set_device(1)

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
        model_names = ['Resnet18_2']#Resnet50_2']#, "Resnet18_2"]
        load_models = [False, False]
        
        for idx, model_name in enumerate(model_names):
            print(model_name)
            if model_name == "Resnet18_2":
                model = ResNetPretrain(18, pretrained=False).to(device)
                if load_models[idx]:
                    model.load_state_dict(torch.load("./" + model_name + ".pth"))
                iteration = 1
            elif model_name == "Resnet50_2":
                model = ResNetPretrain(50, pretrained=False).to(device)
                if load_models[idx]:
                    model.load_state_dict(torch.load("./" + model_name + ".pth"))
                iteration = 1
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
        model_names = ["./weight/Resnet18_2.pth", "./weight/Resnet50_2.pth", 
                "./weight/Resnet18_pretrain_2.pth", "./weight/Resnet50_pretrain_23_82.pth"]#"./Resnet50_pretrain_2.pth"]
        models = [ResNetPretrain(18, pretrained=False).to(device), 
                ResNetPretrain(50, pretrained=False).to(device), 
                ResNetPretrain(18, pretrained=False).to(device), 
                ResNetPretrain(50, pretrained=False).to(device)]

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


