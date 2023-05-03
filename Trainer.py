# Source: https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH
import os
import threading

import torch
import matplotlib.pyplot as plt
from torch import nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class Trainer(threading.Thread):
    def __init__(self, model, model_name, loaders, device, logger, log=True, validation=True, optimizer=None,
                 loss=nn.CrossEntropyLoss(), epochs=10, logSave=None, fileSave=None):
        threading.Thread.__init__(self)
        self.model = model.to(device)
        self.model_name = model_name
        self.loaders = loaders
        self.device = device
        self.logger = logger
        self.log = log
        self.validation = validation
        self.optimizer = optimizer
        self.criterion = loss
        self.epochs = epochs
        self.logSave = logSave
        self.fileSave = fileSave

    def train_step(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        _, predictions = outputs.max(1)

        return predictions, labels, loss

    def run(self):
        os.makedirs(os.path.dirname(self.logSave), exist_ok=True)
        with open(self.logSave, 'w') as logFile:
            for epoch in range(self.epochs):
                self.model.train()
                total = 0
                total_correct = 0
                total_loss = 0
                for i, (images, labels) in enumerate(self.loaders['train']):
                    predictions, labels, loss = self.train_step(images, labels)
                    total += images.size(0)
                    total_correct += (predictions == labels).sum()
                    total_loss += loss.item() * images.size(0)

                accuracy = total_correct / total
                loss = total_loss / total
                logFile.write(f'Train epoch {epoch}: Loss({loss:6.4f}) Accuracy ({accuracy:6.4f})\n')
                if self.validation:
                    eval_loss, eval_acc = self.evaluate(epoch, mode='test', logFile=logFile)
                    if self.log:
                        self.logger.add_scalars('Accuracy', {"train_{mn}".format(mn=self.model_name): accuracy,
                                                             "val_{mn}".format(mn=self.model_name): eval_acc},
                                                epoch)
                        self.logger.add_scalars('Loss', {"train_{mn}".format(mn=self.model_name): loss,
                                                         "val_{mn}".format(mn=self.model_name): eval_loss}, epoch)
                logFile.write(f'Epoch {epoch} done')

            torch.save(self.model.state_dict(), self.fileSave)

    def evaluate(self, epoch=0, mode='test', logFile=None):
        self.model.eval()
        total_loss = 0
        total = 0
        y_test = []
        y_pred = []
        for i, (images, labels) in enumerate(self.loaders[mode]):
            images = images.to(self.device)
            labels = labels.to(self.device)
            y_test.extend(labels.tolist())
            with torch.no_grad():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                total += images.size(0)
                _, predictions = outputs.max(1)
                y_pred.extend(predictions.tolist())
        loss = total_loss / total

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = map(float, cm.ravel())

        sensitivity = 0 if (tp + fn) == 0 else tp / (tp + fn)
        specificity = 0 if (tn + fp) == 0 else tn / (tn + fp)
        precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1score = 0 if (sensitivity + precision) == 0 else 2 * (sensitivity * precision) / (sensitivity + precision)

        #        ConfusionMatrixDisplay(cm).plot()
        #        plt.show()

        logFile.write(f'====={mode} epoch {epoch}: Loss({loss:6.4f}) Accuracy ({accuracy:6.4f})=====\n')
        logFile.write(f'=====Sensitivity ({sensitivity:6.4f}) Specificity ({specificity:6.4f})=====\n')
        logFile.write(f'=====Precision ({precision:6.4f}) F1 Score ({f1score:6.4f})=====\n')

        return loss, accuracy
