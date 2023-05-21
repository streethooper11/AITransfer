# Source: https://github.com/EhabR98/Transfer-Learning-with-MobileNetV2/blob/main/README.md#1


import copy
import os

import keras.optimizers
from sklearn.model_selection import train_test_split
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import tensorflow as tf
import csv
from keras.callbacks import CSVLogger

from DiseaseEnum import Disease


def getDataGen(dataPath, imagePath, img_size, train_data_ratio, name, batch_size):
    validation_ratio = 1 - train_data_ratio

    train_datagen = ImageDataGenerator(
        rotation_range=30,
        horizontal_flip=True,
        rescale=1. / 255,
        validation_split=validation_ratio
    )

    df = pd.read_csv(dataPath)
    df[name] = df[name].astype('str')

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=df,
        directory=imagePath,
        x_col='Image Index',
        y_col=name,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True,
        seed=10,
        subset='training',
    )

    valid_gen = train_datagen.flow_from_dataframe(
        dataframe=df,
        directory=imagePath,
        x_col='Image Index',
        y_col=name,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True,
        seed=10,
        subset='validation',
    )

    return train_gen, valid_gen


def training_stage(dataPath, imagePath, savefolder,
                   outputNum=1, train_data_ratio=0.8, name=''):
    # Settable variables
    batch_size = 32
    epochs = 50
    learning_rate = 0.001

    img_size = (224, 224)

    train_gen, validation_gen = getDataGen(dataPath, imagePath, img_size,
                                           train_data_ratio, name, batch_size)

    df = pd.read_csv(dataPath)
    counts = df.shape[0]
    num_steps_train = (counts * train_data_ratio) // batch_size
    num_steps_val = (counts * (1 - train_data_ratio)) // batch_size

    input_shape = img_size + (3,)
    inputs = tf.keras.Input(shape=input_shape)

    x = preprocess_input(inputs)

    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    prediction_layer = tf.keras.layers.Dense(outputNum, activation='sigmoid')

    outputs = prediction_layer(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.5),
            tf.keras.metrics.Precision(thresholds=0.5, name='precision'),
            tf.keras.metrics.Recall(thresholds=0.5, name='recall'),
            tf.keras.metrics.TruePositives(thresholds=0.5, name='tp'),
            tf.keras.metrics.FalseNegatives(thresholds=0.5, name='fn'),
            tf.keras.metrics.FalsePositives(thresholds=0.5, name='fp'),
            tf.keras.metrics.TrueNegatives(thresholds=0.5, name='tn'),
            tf.keras.metrics.AUC(name='auc'),
        ],
    )

    csv_logger = CSVLogger(savefolder + 'log.csv', append=True, separator=';')
    model.fit(train_gen,
              epochs=epochs,
              steps_per_epoch=num_steps_train,
              validation_data=validation_gen,
              validation_steps=num_steps_val,
              verbose=2,
              callbacks=[csv_logger]
              )


def clean_then_save_csv(origFilePath, cleanFilePath, imgPath, colenum):
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


if __name__ == "__main__":
    setConst = '..\\set\\'
    saveConst = '..\\save\\'

    # Settable variables
    setname = 'sample'
    column = Disease.HasDisease

    imageFolderPath = setConst + setname + '\\train\\'

    if setname == 'sample':
        originalFilePath = setConst + 'sample_labels.csv'
    else:
        originalFilePath = setConst + 'Data_Entry_2017.csv'

    sf = saveConst + setname + '\\' + column.name + '\\'
    os.makedirs(sf, exist_ok=True)
    cleanedFilePath = sf + 'Entry_cleaned.csv'
    clean_then_save_csv(originalFilePath, cleanedFilePath, imageFolderPath,
                        colenum=column.value)

    training_stage(cleanedFilePath, imageFolderPath, sf,
                   outputNum=1, train_data_ratio=0.8, name=column.name)
