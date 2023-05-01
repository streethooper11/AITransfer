# Source: https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH

import torch
import matplotlib.pyplot as plt
from torch import nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class Trainer():
    def __init__(self, model, model_name, loaders, device, logger, log, validation=True, optimizer=None):
        self.model = model.to(device)
        self.model_name = model_name
        self.loaders = loaders
        self.device = device
        self.logger = logger
        self.log = log
        self.validation = validation
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer

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

    def train(self, epochs):
        for epoch in range(epochs):
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
            print(f'Train epoch {epoch}: Loss({loss:6.4f}) Accuracy ({accuracy:6.4f})')
            if self.validation:
                eval_loss, eval_acc = self.evaluate(epoch, mode='test')
                if self.log:
                    self.logger.add_scalars('Accuracy', {"train_{mn}".format(mn=self.model_name): accuracy,
                                                         "val_{mn}".format(mn=self.model_name): eval_acc}, epoch)
                    self.logger.add_scalars('Loss', {"train_{mn}".format(mn=self.model_name): loss,
                                                     "val_{mn}".format(mn=self.model_name): eval_loss}, epoch)
            print(f'Epoch {epoch} done')

    def evaluate(self, epoch=0, mode='test'):
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

        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1score = 2 * (sensitivity * precision) / (sensitivity + precision)

#        ConfusionMatrixDisplay(cm).plot()
#        plt.show()

        print(f'====={mode} epoch {epoch}: Loss({loss:6.4f}) Accuracy ({accuracy:6.4f})=====')
        print(f'=====Sensitivity ({sensitivity:6.4f}) Specificity ({specificity:6.4f})=====')
        print(f'=====Precision ({precision:6.4f}) F1 Score ({f1score:6.4f})=====')

        return loss, accuracy
