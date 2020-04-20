from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

"""
device = torch.cuda.device("cuda")
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
"""

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
