import os
# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pickle
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utils.models import RadNet_resnet3d
from utils.processing import CenterImage
from utils.architectures import mlp_1000, cnn3d

import tensorflow as tf
from tensorflow import keras


### Functions
def format_body_label(labels):
    lab = np.zeros(9, dtype="uint8")
    lab_list = labels.split(';')
    
    for ind in range(len(lab_list)):
        crt = lab_list[ind]
        
        if "B" in crt:
            lab[0] = 1
        if "H" in crt:
            lab[1] = 1
        if "N" in crt:
            lab[2] = 1
        if "Lg" in crt:
            lab[3] = 1
        if "Br" in crt:
            lab[4] = 1
        if "Lv" in crt:
            lab[5] = 1
        if "K" in crt:
            lab[6] = 1
        if "I" in crt:
            lab[7] = 1
        if "P" in crt:
            lab[8] = 1
            
    return lab

def format_view_label(labels):
    lab = np.zeros(3, dtype="uint8")
    
    if "AX" in labels:
        lab[0] = 1
    if "COR" in labels:
        lab[1] = 1
    if "SAG" in labels:
        lab[2] = 1
        
    return lab

def format_seq_label(labels):
    lab = np.zeros(5, dtype="uint8")
    
    if "CT" in labels:
        lab[0] = 1
    if "T1 - SE" in labels:
        lab[1] = 1
    if "T2 - SE" in labels:
        lab[2] = 1
    if "T1 - FLAIR" in labels:
        lab[3] = 1
    if "T2 - FLAIR" in labels:
        lab[4] = 1
    
    return lab

def format_contrast_label(labels):
    lab = np.zeros(2, dtype="uint8")
    lab[labels] = 1
    return lab

### Load configs
with open("./configs/SETTINGS.json") as handle:
    params = json.load(handle)

test_dir = params["TEST_DIR"]
test_label = params["TEST_LABEL"]
output_dir = params["OUTPUT_DIR"]
train_dir = params["TRAIN_DIR"]
train_label = params["TRAIN_LABEL"]

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

def Main():
    
    print("#GPUs : ", len(tf.config.experimental.list_physical_devices('GPU')))
    # scan lists
    with open(train_label,"rb") as handle:
        train_dict = pickle.load(handle)
        handle.close()

    train_list = list(train_dict.keys())
    train_list = [img for img in train_list if "TCGA-" in img]

    x_train = []
    y_train_seq = []
    y_train_view = []
    y_train_ctrs = []
    y_train_body = []
    # Run through scans
    for ind in range(len(train_list)):
        crt_path = os.path.join(train_dir,train_list[ind])
        with open(crt_path,"rb") as handle:
            crt_img = pickle.load(handle)
            handle.close()

        # Format image for input
        img_array = crt_img["img_array"]
        img_array = (img_array - np.min(img_array))/(np.max(img_array)-np.min(img_array)).astype("float32")
        img_array = CenterImage(img_array,(48,192,192)) # fix size with zero padding 
        img_array = np.expand_dims(img_array,axis=-1)
        x_train.append(img_array)

        y_train_seq.append(format_seq_label(train_dict[train_list[ind]][1]))
        y_train_view.append(format_view_label(train_dict[train_list[ind]][2]))
        y_train_ctrs.append(format_contrast_label(train_dict[train_list[ind]][3]))
        y_train_body.append(format_body_label(train_dict[train_list[ind]][4]))

    x_train = np.array([x_train])
    y_train_seq = np.array([y_train_seq])
    y_train_view = np.array([y_train_view])
    y_train_ctrs = np.array([y_train_ctrs])
    y_train_body = np.array([y_train_body])
    
    x_train = x_train.reshape(len(train_list), 48, 192, 192)
    y_train_seq  = y_train_seq.reshape(len(train_list), 5)
    y_train_view = y_train_view.reshape(len(train_list), 3)
    y_train_ctrs = y_train_ctrs.reshape(len(train_list), 2)
    y_train_body = y_train_body.reshape(len(train_list), 9)

    out_seq = keras.layers.Dense(5,activation="softmax",name="out_seq")
    out_view = keras.layers.Dense(3,activation="softmax",name="out_view")
    out_ctrs = keras.layers.Dense(2,activation="softmax",name="out_ctrs")
    out_body = keras.layers.Dense(9,activation="sigmoid",name="out_body")
    # out_space = keras.layers.Dense(1,activation="linear",name="out_space")

    # model = cnn3d()
    model = RadNet_resnet3d()
    # model = model.GetModel()
    model.summary()

    model.compile(loss={'out_seq': 'categorical_crossentropy', 'out_view': 'categorical_crossentropy', 'out_ctrs': 'categorical_crossentropy', 'out_body': 'binary_crossentropy'}, optimizer='adam', metrics=['accuracy'])
    
    model.fit(x_train, {'out_seq': y_train_seq, 'out_view': y_train_view, 'out_ctrs': y_train_ctrs, 'out_body': y_train_body}, epochs=10, batch_size=16, verbose=1, validation_split=0.2)
    keras.models.save_model(model, "./models/cnn3d")
    # model = keras.models.load_model("./models/cnn3d")

    with open(test_label,"rb") as handle:
        test_dict = pickle.load(handle)
        handle.close()

    test_list = list(test_dict.keys())
    test_list = [img for img in test_list if "TCGA-" in img]

    # x_test = []
    # y_test = []
    pred_dict = {}
    # Run through scans
    for ind in range(len(test_list)):
        crt_path = os.path.join(test_dir,test_list[ind])
        with open(crt_path,"rb") as handle:
            crt_img = pickle.load(handle)
            handle.close()

        # Format image for input
        img_array = crt_img["img_array"]
        img_array = (img_array - np.min(img_array))/(np.max(img_array)-np.min(img_array)).astype("float32")
        img_array = CenterImage(img_array,(48,192,192)) # fix size with zero padding 
        img_array = np.expand_dims(img_array,axis=-1)

        x_test = np.array([img_array])
        x_test = x_test.reshape(1, 48, 192, 192, 1)
        pred = model.predict_on_batch(x_test)

            # Parse prediction
        y_seq = format_seq_label(test_dict[test_list[ind]][1])
        y_view = format_view_label(test_dict[test_list[ind]][2])
        y_ctrs = format_contrast_label(test_dict[test_list[ind]][3])
        y_body = format_body_label(test_dict[test_list[ind]][4])
        out_seq = {"prediction":list(pred[0][0]),"label":list(y_seq)}
        out_view = {"prediction":list(pred[1][0]),"label":list(y_view)}
        out_ctrs = {"prediction":list(pred[2][0]),"label":list(y_ctrs)}
        out_body = {"prediction":list(pred[3][0]),"label":list(y_body)}
        pred_dict[test_list[ind]] = [out_seq,out_view,out_ctrs,out_body]

    # Save prediction and labels for analysis later
    with open(os.path.join(output_dir,"test_results_dict"),"wb") as handle:
        pickle.dump(pred_dict,handle)
    print("\nFinished.")



if __name__=="__main__":
    Main()