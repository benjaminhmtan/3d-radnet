import numpy as np

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