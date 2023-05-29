# Source: https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay

from DiseaseEnum import Disease


# Source: https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay

from DiseaseEnum import Disease


class TrainerSingle:
    def __init__(self, best_path, modelinf, loaders, device, validation, loss, pref, name='', ratio=0.8):
        self.best_path = best_path
        self.model = modelinf.model
        self.loaders = loaders
        self.device = device
        self.validation = validation
        self.optimizer = modelinf.opt[0]
        self.criterion = loss
        self.epochs = modelinf.opt[4]
        self.pref = pref
        self.suffix = modelinf.name + modelinf.opt[1] + modelinf.opt[2] + '_decay_' + \
                      str(modelinf.opt[3]) + '_trainratio_' + str(ratio)
        self.name = name

    def train_step(self, images, labels):
        labels = labels.type(torch.float)
        self.optimizer.zero_grad()

        images = images.to(self.device)
        labels = labels.to(self.device)

        outputs = torch.sigmoid(self.model(images))
        loss = self.criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        self.optimizer.step()

        outputs = torch.round(outputs)

        return outputs, labels, loss

    def train(self):
        logsave = os.path.join(self.pref, 'logs', self.suffix + '.log')
        os.makedirs(os.path.dirname(logsave), exist_ok=True)
        with open(logsave, 'w') as logFile:
            for epoch in range(self.epochs):
                logFile.write(f'Starting epoch {epoch}:\n')
                print(f'Starting epoch {epoch}:')
                self.model.train()
                total = 0
                total_loss = 0.0
                y_test = []
                y_pred = []
                for i, (images, labels) in enumerate(self.loaders['train']):
                    predictions, labels, loss = self.train_step(images, labels)
                    y_test.extend(labels.tolist())
                    y_pred.extend(predictions.tolist())
                    total += images.size(0)
                    total_loss += loss.item() * images.size(0)

                loss = total_loss / total

                cm = confusion_matrix(y_test, y_pred)
                tn, fp, fn, tp = map(float, cm.ravel())
                accuracy = (tp + tn) / (tp + tn + fp + fn)

                ConfusionMatrixDisplay(cm).plot()
                cmdname = os.path.join(self.pref, 'matrices', self.suffix + '_train_epoch' + str(epoch) + '.png')
                os.makedirs(os.path.dirname(cmdname), exist_ok=True)
                plt.title('Training Confusion Matrix for ' + self.name)
                plt.savefig(cmdname)
                plt.close()

                logFile.write(f'Train epoch {epoch}: Loss({loss:6.4f}) Accuracy ({accuracy:6.4f})\n')
                print(f'Train epoch {epoch}: Loss({loss:6.4f}) Accuracy ({accuracy:6.4f})')

                f1score = 0
                if self.validation:
                    f1score = self.evaluate(epoch, mode='test', logFile=logFile)

                logFile.write(f'Epoch {epoch} done\n------------------------------\n')
                print(f'Epoch {epoch} done\n------------------------------')

                modelsavename = os.path.join(self.pref, 'models',
                                             'f1_' + str(f1score) + self.suffix + '_epoch' + str(epoch) + '.tar')
                os.makedirs(os.path.dirname(modelsavename), exist_ok=True)
                if self.best_path is None:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss,
                        'f1score': f1score,
                    }, modelsavename)
                    self.best_path = (modelsavename, f1score)
                else:
                    if f1score > self.best_path[1]:
                        os.remove(self.best_path[0])
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': loss,
                            'f1score': f1score,
                        }, modelsavename)
                        self.best_path = (modelsavename, f1score)

        return self.best_path

    def evaluate(self, epoch=0, mode='test', logFile=None):
        self.model.eval()
        total_loss = 0.0
        total = 0
        y_test = []
        y_pred = []

        for i, (images, labels) in enumerate(self.loaders[mode]):
            labels = labels.type(torch.float)
            with torch.no_grad():
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = torch.sigmoid(self.model(images))
                loss = self.criterion(outputs, labels.unsqueeze(1))

                outputs = torch.round(outputs)

            total_loss += loss.item() * images.size(0)
            total += images.size(0)

            y_pred.extend(outputs.tolist())
            y_test.extend(labels.tolist())

        loss = total_loss / total

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = map(float, cm.ravel())

        sensitivity = 0 if (tp + fn) == 0 else tp / (tp + fn)
        specificity = 0 if (tn + fp) == 0 else tn / (tn + fp)
        precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1score = 0 if (sensitivity + precision) == 0 else 2 * (sensitivity * precision) / (sensitivity + precision)

        ConfusionMatrixDisplay(cm).plot()
        cmdname = os.path.join(self.pref, 'matrices', self.suffix + '_validation_epoch' + str(epoch) + '.png')
        os.makedirs(os.path.dirname(cmdname), exist_ok=True)
        plt.title('Validation Confusion Matrix for ' + self.name)
        plt.savefig(cmdname)
        plt.close()

        logFile.write(f'====={mode} epoch {epoch}: Loss({loss:6.4f}) Accuracy ({accuracy:6.4f})=====\n')
        logFile.write(f'=====Sensitivity ({sensitivity:6.4f}) Specificity ({specificity:6.4f})=====\n')
        logFile.write(f'=====Precision ({precision:6.4f}) F1 Score ({f1score:6.4f})=====\n')

        print(f'====={mode} epoch {epoch}: Loss({loss:6.4f}) Accuracy ({accuracy:6.4f})=====')
        print(f'=====Sensitivity ({sensitivity:6.4f}) Specificity ({specificity:6.4f})=====')
        print(f'=====Precision ({precision:6.4f}) F1 Score ({f1score:6.4f})=====')

        return float(f'{f1score:6.4f}')


class TrainerMulti:
    def __init__(self, modelinf, loaders, device, validation, loss, pref, numoutputs, listtops=5, name='', ratio=0.8):
        self.model = modelinf.model
        self.loaders = loaders
        self.device = device
        self.validation = validation
        self.optimizer = modelinf.opt[0]
        self.criterion = loss
        self.epochs = modelinf.opt[4]
        self.pref = pref
        self.suffix = modelinf.name + modelinf.opt[1] + modelinf.opt[2] + '_decay_' + \
                      str(modelinf.opt[3]) + '_trainratio_' + str(ratio)
        self.numoutputs = numoutputs
        self.listtops = listtops
        self.name = name

    def train_step(self, images, labels):
        self.optimizer.zero_grad()

        images = images.to(self.device)
        labels = labels.to(self.device)

        outputs = torch.sigmoid(self.model(images))
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        for i in range(len(outputs)):
            outputs[i] = torch.round(outputs[i])

        return outputs, labels, loss

    def train(self):
        logsave = os.path.join(self.pref, 'logs', self.suffix + '.log')
        os.makedirs(os.path.dirname(logsave), exist_ok=True)
        topmodels = []
        with open(logsave, 'w') as logFile:
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

                if self.numoutputs > 1:
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
                else:
                    cmd = ConfusionMatrixDisplay(cm[0], display_labels=np.unique(y_test))
                    cmd.plot()
                    cmd.ax_.set_title(self.name)

                cmdname = os.path.join(self.pref, 'matrices', self.suffix + '_train_epoch' + str(epoch) + '.png')
                os.makedirs(os.path.dirname(cmdname), exist_ok=True)

                plt.savefig(cmdname)
                plt.close()

                logFile.write(f'Train epoch {epoch}: Loss({loss:6.4f}) Accuracy ({accuracy:6.4f})\n')
                print(f'Train epoch {epoch}: Loss({loss:6.4f}) Accuracy ({accuracy:6.4f})')

                f1score = 0
                if self.validation:
                    f1score = self.evaluate(epoch, mode='test', logFile=logFile)

                logFile.write(f'Epoch {epoch} done\n------------------------------\n')
                print(f'Epoch {epoch} done\n------------------------------')

                modelsavename = os.path.join(self.pref, 'models',
                                             'f1_' + str(f1score) + self.suffix + '_epoch' + str(epoch) + '.pth')
                os.makedirs(os.path.dirname(modelsavename), exist_ok=True)

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
            with torch.no_grad():
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = torch.sigmoid(self.model(images))
                loss = self.criterion(outputs, labels)

            for j in range(len(outputs)):
                outputs[j] = torch.round(outputs[j])

            y_pred.extend(outputs.tolist())
            y_test.extend(labels.tolist())
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

        if self.numoutputs > 1:
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
        else:
            cmd = ConfusionMatrixDisplay(cm[0], display_labels=np.unique(y_test))
            cmd.plot()
            cmd.ax_.set_title(self.name)

        cmdname = os.path.join(self.pref, 'matrices', self.suffix + '_validation_epoch' + str(epoch) + '.png')
        plt.savefig(cmdname)
        plt.close()

        loss = total_loss / counter

        logFile.write(f'====={mode} epoch {epoch}: Loss({loss:6.4f}) Accuracy ({accuracy:6.4f})=====\n')
        logFile.write(f'=====Sensitivity ({sensitivity:6.4f}) Specificity ({specificity:6.4f})=====\n')
        logFile.write(f'=====Precision ({precision:6.4f}) F1 Score ({f1score:6.4f})=====\n')

        print(f'====={mode} epoch {epoch}: Loss({loss:6.4f}) Accuracy ({accuracy:6.4f})=====')
        print(f'=====Sensitivity ({sensitivity:6.4f}) Specificity ({specificity:6.4f})=====')
        print(f'=====Precision ({precision:6.4f}) F1 Score ({f1score:6.4f})=====')

        return float(f'{f1score:6.4f}')
