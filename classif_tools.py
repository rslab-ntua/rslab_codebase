import numpy as np
import sklearn.metrics as metr
import pandas as pd


def confusion_matrix(pred, gt, fullpath, labels):


    # convert to int
    y_gt      = gt.astype('int') 
    y_pred    = pred.astype('int')

    # compute metrics
    cm      = metr.confusion_matrix  (y_gt, y_pred)
    kappa   = metr.cohen_kappa_score (y_gt, y_pred)
    OA      = metr.accuracy_score    (y_gt, y_pred)
    UA      = metr.precision_score   (y_gt, y_pred, average=None)
    #UA_avg  = metr.precision_score   (y_gt, y_pred, average='weighted')
    PA      = metr.recall_score      (y_gt, y_pred, average=None)
    #PA_avg  = metr.recall_score      (y_gt, y_pred, average='weighted')
      
    # confusion matrix with UA, PA
    sz1, sz2 = cm.shape
    cm_with_stats             = np.zeros((sz1+2,sz2+2))
    cm_with_stats[0:-2, 0:-2] = cm
    cm_with_stats[-1  , 0:-2] = np.round(UA,2)
    cm_with_stats[0:-2,   -1] = np.round(PA,2)
    cm_with_stats[-2  , 0:-2] = np.sum(cm, axis=0) 
    cm_with_stats[0:-2,   -2] = np.sum(cm, axis=1)
    
    # convert to list
    cm_list = cm_with_stats.tolist()
    
    # first row
    first_row = []
    first_row.extend (labels)
    first_row.append ('sum')
    first_row.append ('PA')
    
    # first col
    first_col = []
    first_col.extend(labels)   
    first_col.append ('sum')
    first_col.append ('UA')
    
    # fill rest of the text 
    idx = 0
    for sublist in cm_list:
        if   idx == sz1:
            cm_list[idx] = sublist
            sublist[-2]  = 'kappa:'
            sublist[-1]  = round(kappa,2)           
        elif idx == sz1+1:
            sublist[-2]  = 'OA:'
            sublist[-1]  = round(OA,2)
            cm_list[idx] = sublist            
        idx +=1
    
    # Convert to data frame
    df = pd.DataFrame(np.array(cm_list))
    df.columns = first_row
    df.index = first_col
    
    # Write to xls
    writer = pd.ExcelWriter(fullpath)
    df.to_excel(writer, 'Sheet 1')
    writer.save()
    
    return df


def load_data_multi_samples(data, labels, ratio, n_classes):
    
    cl_0 = np.where(labels==0)
    data_0 = data[cl_0,:]
    data_0 = data_0[0,:,:]
    rand_perm = np.random.permutation(data_0.shape[0])
    data_0 = data_0[rand_perm]
    s = data_0.shape[0] * ratio
    s = int(s)
    train_set = data_0[0:s,:]
    train_lab = np.zeros((s))
    other_set = data_0[s:data_0.shape[0],:]
    other_lab = np.zeros(other_set.shape[0])
    
    i = 1
    while i < n_classes:
        cl_i = np.where(labels==i)
        data_i = data[cl_i,:]
        data_i = data_i[0,:,:]
        rand_perm = np.random.permutation(data_i.shape[0])
        data_i = data_i[rand_perm]
        
        s = data_i.shape[0] * ratio
        s = int(s)
        train_set_i = data_i[0:s,:]
        train_lab_i = np.ones((s)) * i
        other_set_i = data_i[s:data_i.shape[0],:]
        other_lab_i = np.ones(other_set_i.shape[0]) * i
        
        
        train_set = np.concatenate((train_set, train_set_i), axis=0)
        train_lab = np.concatenate((train_lab, train_lab_i), axis=0)
        other_set = np.concatenate((other_set, other_set_i), axis=0)
        other_lab = np.concatenate((other_lab, other_lab_i), axis=0)
        
        
        
        i = i + 1
    
    rand_perm = np.random.permutation(train_lab.shape[0])
    train_set = train_set[rand_perm]
    train_lab = train_lab[rand_perm]
    
    rand_perm = np.random.permutation(other_lab.shape[0])
    other_set = other_set[rand_perm]
    other_lab = other_lab[rand_perm]

    return train_set, train_lab, other_set,other_lab