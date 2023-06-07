import os

import pandas as pd
from sklearn.model_selection import train_test_split
import csv

from DiseaseEnum import Disease


def savestratifiedsplits(dataPath, savefolder, train_ratio):
    df = pd.read_csv(dataPath)

    y = df.iloc[:, -1]
    train, test = train_test_split(df, train_size=train_ratio, stratify=y, random_state=11)

    train.to_csv(os.path.join(savefolder, 'Entry_cleaned_2020_Train.csv'), index=False)
    test.to_csv(os.path.join(savefolder, 'Entry_cleaned_2020_Test.csv'), index=False)


def clean_then_save_csv(origFilePath, cleanFilePath, imgPath, colenum, position):
    with open(origFilePath, newline='') as csvreadfile:
        with open(cleanFilePath, 'w', newline='') as csvwritefile:
            csv_read = csv.reader(csvreadfile, delimiter=',')
            csv_write = csv.writer(csvwritefile, delimiter=',')
            firstrow = True

            if colenum == -1:
                for row in csv_read:
                    if firstrow is False:
                        img_path = os.path.join(imgPath, row[0])
                        if os.path.exists(img_path):
                            diseases = [0 for n in range(len(Disease) - 1)]
                            if row[1] == "No Finding":
                                values = [row[0]]
                                values.extend(diseases)
                                csv_write.writerow(values)
                            else:
                                diseasefindings = row[1].split('|')
                                for eachdisease in diseasefindings:
                                    diseases[Disease[eachdisease].value] = 1
                                values = [row[0]]
                                values.extend(diseases)
                                csv_write.writerow(values)
                    else:
                        firstrow = False
                        csv_write.writerow(['Image Index'] +
                                           [d.name for d in Disease if d.value < 20])

            elif colenum == Disease.HasDisease.value:
                for row in csv_read:
                    if firstrow is False:
                        img_path = os.path.join(imgPath, row[0])
                        if position != '':
                            if row[6] == position:
                                if os.path.exists(img_path):
                                    if row[1] == "No Finding":
                                        csv_write.writerow([row[0], 0])
                                    else:
                                        csv_write.writerow([row[0], 1])
                        else:
                            if os.path.exists(img_path):
                                if row[1] == "No Finding":
                                    csv_write.writerow([row[0], 0])
                                else:
                                    csv_write.writerow([row[0], 1])

                    else:
                        firstrow = False
                        csv_write.writerow(['Image Index', Disease.HasDisease.name])

            else:
                diseasecol = ''
                for data in Disease:
                    if data.value == colenum:
                        diseasecol = data.name
                        break

                for row in csv_read:
                    if firstrow is False:
                        img_path = os.path.join(imgPath, row[0])
                        if os.path.exists(img_path):
                            diseasefindings = row[1].split('|')
                            if diseasecol in diseasefindings:
                                csv_write.writerow([row[0], 1])
                            else:
                                csv_write.writerow([row[0], 0])
                    else:
                        firstrow = False
                        csv_write.writerow(['Image Index', diseasecol])


def makecleanedcsv(topsavefolder, imageFolderPath, usedset, colenum, position):
    if usedset == 'sample':
        origcsvpath = os.path.join('set', 'sample_labels.csv')
    else:
        origcsvpath = os.path.join('set', 'Data_Entry_2017_v2020.csv')

    cleanpath = os.path.join(topsavefolder, 'Entry_cleaned_2020.csv')
    clean_then_save_csv(origcsvpath, cleanpath, imageFolderPath, colenum, position)
