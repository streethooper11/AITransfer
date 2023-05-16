# Source: https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH

import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay

from DiseaseEnum import Disease


class TrainerMulti:
    def __init__(self, model, model_name, loaders, device, validation, optimizer,
                 loss, epochs, pref, numoutputs, listtops=5):
        self.model = model
        self.model_name = model_name
        self.loaders = loaders
        self.device = device
        self.validation = validation
        self.optimizer = optimizer
        self.criterion = loss
        self.epochs = epochs
        self.logSave = pref + '.log'
        self.fileSave = pref + '_epoch'
        self.matrixSave = pref + '_matrix'
        self.numoutputs = numoutputs
        self.listtops = listtops

    def train_step(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        self.optimizer.zero_grad()
        outputs = torch.sigmoid(self.model(images))
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        for i in range(len(outputs)):
            outputs[i] = torch.round(outputs[i])

        return outputs, labels, loss

    def train(self):
        os.makedirs(os.path.dirname(self.logSave), exist_ok=True)
        topmodels = []
        with open(self.logSave, 'w') as logFile:
            for epoch in range(self.epochs):
                logFile.write(f'Starting epoch {epoch}:\n')
                print(f'Starting epoch {epoch}:')
                self.model.train()
                total = 0
                total_correct = 0
                total_loss = 0.0
                counter = 0
                y_test = []
                y_pred = []
                for i, (images, labels) in enumerate(self.loaders['train']):
                    predictions, labels, loss = self.train_step(images, labels)
                    y_test.extend(labels.tolist())
                    y_pred.extend(predictions.tolist())
                    total += images.size(0) * self.numoutputs
                    total_correct += (predictions == labels).sum()
                    total_loss += loss.item()
                    counter += 1

                accuracy = total_correct / total
                loss = total_loss / counter

                cm = multilabel_confusion_matrix(y_test, y_pred)
                _, axes = plt.subplots(4, 4, figsize=(20, 20))

                for eachdisease in Disease:
                    if eachdisease.value != Disease.HasDisease.value:
                        cmd = ConfusionMatrixDisplay(cm[eachdisease.value], display_labels=np.unique(y_test))
                        cmd.plot(ax=axes[eachdisease.value // 4, eachdisease.value % 4])
                        cmd.ax_.set_title(eachdisease.name)

                axes[-1, -2].remove()
                axes[-1, -2] = None
                axes[-1, -1].remove()
                axes[-1, -1] = None
                cmdname = self.matrixSave + '_train_epoch' + str(epoch) + '_all.png'
                plt.savefig(cmdname)
                plt.close()

                logFile.write(f'Train epoch {epoch}: Loss({loss:6.4f}) Accuracy ({accuracy:6.4f})\n')
                print(f'Train epoch {epoch}: Loss({loss:6.4f}) Accuracy ({accuracy:6.4f})')

                f1score = 0
                if self.validation:
                    f1score = self.evaluate(epoch, mode='test', logFile=logFile)

                logFile.write(f'Epoch {epoch} done\n------------------------------\n')
                print(f'Epoch {epoch} done\n------------------------------')

                modelsavename = self.fileSave + str(epoch) + '_f1_' + str(f1score) + '.pth'
                topmodels.append((modelsavename, f1score))
                if len(topmodels) <= self.listtops:
                    torch.save(self.model.state_dict(), modelsavename)
                else:
                    topmodels.sort(key=lambda a: a[1], reverse=True)
                    lastf1 = topmodels.pop()
                    if f1score != lastf1[1]:
                        os.remove(lastf1[0])
                        torch.save(self.model.state_dict(), modelsavename)

    def evaluate(self, epoch=0, mode='test', logFile=None):
        self.model.eval()
        total_loss = 0.0
        counter = 0
        tp = fn = fp = tn = 0
        y_test = []
        y_pred = []

        for i, (images, labels) in enumerate(self.loaders[mode]):
            images = images.to(self.device)
            labels = labels.to(self.device)
            y_test.extend(labels.tolist())
            with torch.no_grad():
                outputs = torch.sigmoid(self.model(images))
                loss = self.criterion(outputs, labels)
                for j in range(len(outputs)):
                    outputs[j] = torch.round(outputs[j])

            y_pred.extend(outputs.tolist())
            total_iter = images.size(0) * self.numoutputs

            tp_iter = torch.logical_and((outputs == labels), (labels == 1.)).sum()
            fn_iter = (outputs < labels).sum()
            fp_iter = (outputs > labels).sum()
            tn_iter = total_iter - tp_iter - fn_iter - fp_iter

            fn += fn_iter
            fp += fp_iter
            tp += tp_iter
            tn += tn_iter

            total_loss += loss.item()
            counter += 1

        cm = multilabel_confusion_matrix(y_test, y_pred)

        sensitivity = 0 if (tp + fn) == 0 else tp / (tp + fn)
        specificity = 0 if (tn + fp) == 0 else tn / (tn + fp)
        precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1score = 0 if (sensitivity + precision) == 0 else 2 * (sensitivity * precision) / (sensitivity + precision)

        _, axes = plt.subplots(4, 4, figsize=(20, 20))

        for eachdisease in Disease:
            if eachdisease.value != Disease.HasDisease.value:
                cmd = ConfusionMatrixDisplay(cm[eachdisease.value], display_labels=np.unique(y_test))
                cmd.plot(ax=axes[eachdisease.value // 4, eachdisease.value % 4])
                cmd.ax_.set_title(eachdisease.name)

        axes[-1, -2].remove()
        axes[-1, -2] = None
        axes[-1, -1].remove()
        axes[-1, -1] = None
        cmdname = self.matrixSave + '_validation_epoch' + str(epoch) + '_all.png'
        plt.savefig(cmdname)
        plt.close()

        loss = total_loss / counter

        logFile.write(f'====={mode} epoch {epoch}: Loss({loss:6.4f}) Accuracy ({accuracy:6.4f})=====\n')
        logFile.write(f'=====Sensitivity ({sensitivity:6.4f}) Specificity ({specificity:6.4f})=====\n')
        logFile.write(f'=====Precision ({precision:6.4f}) F1 Score ({f1score:6.4f})=====\n')

        print(f'====={mode} epoch {epoch}: Loss({loss:6.4f}) Accuracy ({accuracy:6.4f})=====')
        print(f'=====Sensitivity ({sensitivity:6.4f}) Specificity ({specificity:6.4f})=====')
        print(f'=====Precision ({precision:6.4f}) F1 Score ({f1score:6.4f})=====')

        return float(f'{f1score.item():6.4f}')


class Trainer():
    def __init__(self, model, model_name, loaders, device, validation, optimizer,
                 loss, epochs, pref, name):
        self.model = model
        self.model_name = model_name
        self.loaders = loaders
        self.device = device
        self.validation = validation
        self.optimizer = optimizer
        self.criterion = loss
        self.epochs = epochs
        self.logSave = pref + '.log'
        self.fileSave = pref + '.pth'
        self.matrixSave = pref + '_matrix_epoch'
        self.name = name

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

    def train(self):
        os.makedirs(os.path.dirname(self.logSave), exist_ok=True)
        with open(self.logSave, 'w') as logFile:
            for epoch in range(self.epochs):
                logFile.write(f'Starting epoch {epoch}:\n')
                print(f'Starting epoch {epoch}:')
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
                print(f'Train epoch {epoch}: Loss({loss:6.4f}) Accuracy ({accuracy:6.4f})')
                if self.validation:
                    self.evaluate(epoch, mode='test', logFile=logFile)

                logFile.write(f'Epoch {epoch} done\n------------------------------\n')
                print(f'Epoch {epoch} done\n------------------------------')

            # torch.save(self.model.state_dict(), self.fileSave)

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

        ConfusionMatrixDisplay(cm).plot()
        cmdname = self.matrixSave + str(epoch) + '_' + self.name + '.png'
        plt.title('Confusion Matrix for ' + self.name)
        plt.savefig(cmdname)
        plt.close()

        logFile.write(f'====={mode} epoch {epoch}: Loss({loss:6.4f}) Accuracy ({accuracy:6.4f})=====\n')
        logFile.write(f'=====Sensitivity ({sensitivity:6.4f}) Specificity ({specificity:6.4f})=====\n')
        logFile.write(f'=====Precision ({precision:6.4f}) F1 Score ({f1score:6.4f})=====\n')

        print(f'====={mode} epoch {epoch}: Loss({loss:6.4f}) Accuracy ({accuracy:6.4f})=====')
        print(f'=====Sensitivity ({sensitivity:6.4f}) Specificity ({specificity:6.4f})=====')
        print(f'=====Precision ({precision:6.4f}) F1 Score ({f1score:6.4f})=====')
