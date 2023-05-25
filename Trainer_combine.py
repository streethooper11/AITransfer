# Source: https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay

from DiseaseEnum import Disease


class TrainerCombineSingle:
    def __init__(self, modelinf, loaders, device, validation, loss, pref, name='', ratio=0.8):
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
        bestmodel = None
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
                if bestmodel is None:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss,
                        'f1score': f1score,
                    }, modelsavename)
                    bestmodel = (modelsavename, f1score)
                else:
                    if f1score > bestmodel[1]:
                        os.remove(bestmodel[0])
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': loss,
                            'f1score': f1score,
                        }, modelsavename)
                        bestmodel = (modelsavename, f1score)

        return bestmodel[0]


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
