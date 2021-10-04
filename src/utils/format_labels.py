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

def format_stage_label(label):
    lab = np.zeros(13, dtype="uint8")
    if 'Stage I' == label:
        lab[0] = 1
    if 'Stage IA' == label:
        lab[1] = 1
    if 'Stage IB' == label:
        lab[2] = 1
    if 'Stage II' == label:
        lab[3] = 1
    if 'Stage IIA' == label:
        lab[4] = 1
    if 'Stage IIB' == label:
        lab[5] = 1
    if 'Stage III' == label:
        lab[6] = 1
    if 'Stage IIIA' == label:
        lab[7] = 1
    if 'Stage IIIB' == label:
        lab[8] = 1
    if 'Stage IIIC' == label:
        lab[9] = 1
    if 'Stage IV' == label:
        lab[10] = 1
    if 'Stage IVA' == label:
        lab[11] = 1
    if 'Stage IVB' == label:
        lab[12] = 1
    
    return lab

def format_alt_stage_label(label):
    lab = np.zeros(4, dtype="uint8")
    if 'Stage I' == label:
        lab[0] = 1
    if 'Stage IA' == label:
        lab[0] = 1
    if 'Stage IB' == label:
        lab[0] = 1
    if 'Stage II' == label:
        lab[1] = 1
    if 'Stage IIA' == label:
        lab[1] = 1
    if 'Stage IIB' == label:
        lab[1] = 1
    if 'Stage III' == label:
        lab[2] = 1
    if 'Stage IIIA' == label:
        lab[2] = 1
    if 'Stage IIIB' == label:
        lab[2] = 1
    if 'Stage IIIC' == label:
        lab[2] = 1
    if 'Stage IV' == label:
        lab[3] = 1
    if 'Stage IVA' == label:
        lab[3] = 1
    if 'Stage IVB' == label:
        lab[3] = 1
    
    return lab

def format_project_label(label):
    lab = np.zeros(12, dtype="uint8")
    if 'BLCA' in label:
        lab[0] = 1
    if 'BRCA' in label:
        lab[1] = 1
    if 'COAD' in label:
        lab[2] = 1
    if 'ESCA' in label:
        lab[3] = 1
    if 'HNSC' in label:
        lab[4] = 1
    if 'KICH' in label:
        lab[5] = 1
    if 'KIRC' in label:
        lab[6] = 1
    if 'KIRP' in label:
        lab[7] = 1
    if 'LIHC' in label:
        lab[8] = 1
    if 'LUAD' in label:
        lab[9] = 1
    if 'LUSC' in label:
        lab[10] = 1
    if 'STAD' in label:
        lab[11] = 1
    
    return lab