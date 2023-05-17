# Source: https://colab.research.google.com/drive/1c5lu1ePav66V_DirkH6YfJyKETul0yrH

import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay

from DiseaseEnum import Disease


class Tester:
    def __init__(self, model, loaders, device, loss, pref, suffix, numoutputs):
        self.model = model
        self.loaders = loaders
        self.device = device
        self.criterion = loss
        self.pref = pref
        self.suffix = suffix
        self.numoutputs = numoutputs

    def test(self, epoch=0, mode='test'):
        self.model.eval()
        total_loss = 0.0
        counter = 0
        tp = fn = fp = tn = 0
        y_test = []
        y_pred = []

        logsave = self.pref + 'logs\\' + self.suffix + '_validation' + '.log'

        with open(logsave, 'w') as logFile:
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
            cmdname = self.pref + 'matrices\\' + self.suffix + '_validation' + '.png'
            plt.savefig(cmdname)
            plt.close()

            loss = total_loss / counter

            logFile.write(f'====={mode} epoch {epoch}: Loss({loss:6.4f}) Accuracy ({accuracy:6.4f})=====\n')
            logFile.write(f'=====Sensitivity ({sensitivity:6.4f}) Specificity ({specificity:6.4f})=====\n')
            logFile.write(f'=====Precision ({precision:6.4f}) F1 Score ({f1score:6.4f})=====\n')

            print(f'====={mode} epoch {epoch}: Loss({loss:6.4f}) Accuracy ({accuracy:6.4f})=====')
            print(f'=====Sensitivity ({sensitivity:6.4f}) Specificity ({specificity:6.4f})=====')
            print(f'=====Precision ({precision:6.4f}) F1 Score ({f1score:6.4f})=====')
