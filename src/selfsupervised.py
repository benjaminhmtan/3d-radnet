import os

gpu_id = 3
# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)

import pickle
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from utils.processing import CenterImage
from utils.format_labels import format_body_label, format_contrast_label, format_seq_label, format_view_label, format_stage_label
from utils.classifiers import mlp, mlp2, cnn3d, vgg3d, resnet3d
from utils.autoencoders import e1, d1, lean_classifier

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import ShuffleSplit, train_test_split

train_dir = "./train_data/"
output_dir = "./outputs/"
dataset = pd.read_csv("train_data_labels.csv")
num_samples = dataset["Scan Name"].count()

x = []
y_seq = []
y_view = []
y_ctrs = []
y_body = []
y_stage = []
labels = []
# Run through scans
for idx in range(len(dataset["Scan Name"])):
    crt_path = os.path.join(train_dir,dataset["Scan Name"][idx])
    with open(crt_path,"rb") as handle:
        crt_img = pickle.load(handle)
        handle.close()

    # Format image for input
    img_array = crt_img["img_array"]
    img_array = (img_array - np.min(img_array))/(np.max(img_array)-np.min(img_array)).astype("float32")
    img_array = CenterImage(img_array,(48,192,192)) # fix size with zero padding 
    # img_array = np.expand_dims(img_array,axis=-1)
    x.append(img_array)
    # if idx == 540:
    #     print(img_array[18:30,:,:])

    # y_seq.append(format_seq_label(dataset["Sequence"][idx]))
    # y_view.append(format_view_label(dataset["View"][idx]))
    # y_ctrs.append(format_contrast_label(dataset["Contrast"][idx]))
    # y_body.append(format_body_label(dataset["Body Coverage"][idx]))
    y_stage.append(format_stage_label(dataset["ajcc_pathologic_stage"][idx]))

x = np.array([x])
# y_seq = np.array([y_seq])
# y_view = np.array([y_view])
# y_ctrs = np.array([y_ctrs])
# y_body = np.array([y_body])
y_stage = np.array([y_stage])

x        =  x.reshape(num_samples, 48, 192, 192)
# y_seq  =  y_seq.reshape(num_samples, 5)
# y_view =  y_view.reshape(num_samples, 3)
# y_ctrs =  y_ctrs.reshape(num_samples, 2)
# y_body =  y_body.reshape(num_samples, 9)
y_stage = y_stage.reshape(num_samples, 13)
x = np.expand_dims(x, axis=-1)

# x_train, x_test, y_seq_train, y_seq_test, y_view_train, y_view_test, y_ctrs_train, y_ctrs_test, y_body_train, y_body_test, y_stage_train, y_stage_test = train_test_split(x, y_seq, y_view, y_ctrs, y_body, y_stage, test_size=0.2, random_state=42, stratify=y_stage, shuffle=True)
x_train, x_test, y_stage_train, y_stage_test = train_test_split(x, y_stage, test_size=0.2, random_state=42, stratify=y_stage, shuffle=True)

for curr_iter in range(25):
    # out_seq = keras.layers.Dense(5,activation="softmax",name="out_seq")
    # out_view = keras.layers.Dense(3,activation="softmax",name="out_view")
    # out_ctrs = keras.layers.Dense(2,activation="softmax",name="out_ctrs")
    # out_body = keras.layers.Dense(9,activation="sigmoid",name="out_body")
    out_stage = keras.layers.Dense(13,activation="softmax",name="out_stage")


    # output_layers = [out_seq, out_view, out_ctrs, out_body, out_stage]
    output_layers = [out_stage]
    
    model = lean_classifier(output_layers=output_layers).GetModel()
    model.summary()

    # loss_dict = {'out_seq': 'categorical_crossentropy', 'out_view': 'categorical_crossentropy', 'out_ctrs': 'categorical_crossentropy', 'out_body': 'binary_crossentropy', 'out_stage': 'categorical_crossentropy'}
    loss_dict = {'out_stage': 'categorical_crossentropy'}
    model.compile(loss=loss_dict, optimizer='sgd', metrics=['accuracy'])
        
    # out_dict = {'out_seq': y_seq_train, 'out_view': y_view_train, 'out_ctrs': y_ctrs_train, 'out_body': y_body_train, 'out_stage': y_stage_train}
    out_dict = {'out_stage': y_stage_train}

    model.fit(x_train, out_dict, epochs=35, batch_size=16, verbose=1, validation_split=0.15)
    # keras.models.save_model(model, "./models/basic/mlp")

    # Testing
    x_test_batches = np.array_split(x_test, 60)
    y_test_batches = np.array_split(y_stage_test, 60)

    correct_samples = 0
    true_labels = []
    predictions = []
    for x_batch, y_batch in zip(x_test_batches, y_test_batches):
        pred = model.predict_on_batch(x_batch)
        # pred = pred[-1]
        # pred = np.argmax(pred, axis=1)
        # y_batch = np.argmax(y_batch, axis=1)
        for plabel, tlabel in zip(pred, y_batch):
            true_labels.append(tlabel)
            predictions.append(plabel)
            # correct_samples = correct_samples + (plabel == tlabel)

    # acc = correct_samples/x_test.shape[0]
    # print("Test Accuracy: {}. ".format(acc))

    from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
    from collections import OrderedDict

    labels = np.arange(0,13,1,dtype=int)
    label_name = ['Stage I', 'Stage IA', 'Stage IB', 'Stage II', 'Stage IIA', 'Stage IIB', 'Stage III', 'Stage IIIA', 'Stage IIIB', 'Stage IIIC', 'Stage IV', 'Stage IVA', 'Stage IVB']
    columns=["N","AUC","ACC","SEN","SPEC","PPV","NPV"]

    y_true_labels = np.argmax(np.array(true_labels), axis=1)
    y_predictions = np.argmax(np.array(predictions), axis=1)

    cm = confusion_matrix(y_true_labels,y_predictions,labels=labels)
    print(cm)
    tp = 0
    for ind in labels:
        tp = tp + cm[ind,ind]

    total_acc = tp/np.sum(cm)

    performance = OrderedDict()

    for ind in labels:
        N = np.sum(cm[ind,:])
        
        TP = cm[ind,ind]
        TN = np.sum(cm) - np.sum(cm[ind,:]) - np.sum(cm[:,ind]) + TP
        FP = np.sum(cm[:,ind]) - TP
        FN = np.sum(cm[ind,:]) - TP

        if TP != 0:
            auc = roc_auc_score(np.array(true_labels)[:,ind],np.array(predictions)[:,ind])
            acc = (TP+TN)/(TP+TN+FP+FN)
            sen = TP/(TP+FN)
            spec = TN/(TN+FP)
            ppv = TP/(TP+FP)
            npv = TN/(TN+FN)   
        else:
            auc = 0
            sen = 0
            spec = 0
            ppv = 0
            npv = 0
        performance[label_name[ind]] = [N,auc,acc,sen,spec,ppv,npv]

    performance["Total Accuracy"] = [num_samples, None, total_acc, None, None, None, None]
    performance_pd = pd.DataFrame.from_dict(performance,orient='index',columns=columns)
    performance_pd.to_csv(os.path.join(output_dir,"basic_ss_cnn3d{}.csv".format(curr_iter)))
    print(performance_pd)
