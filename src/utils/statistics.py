import torch

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay, multilabel_confusion_matrix, matthews_corrcoef
from scipy import stats

import matplotlib.pyplot as plt

from . import GlobalParameters

def _split_by_threshold(arr):
    arr = np.array(arr)

    if len(arr) != 1:
        a_pos = (arr >= GlobalParameters.positive_threshold).astype(int)
        a_pos[a_pos == 1] = 3
        
        a_neg = (arr <= GlobalParameters.negative_threshold).astype(int)
        a_neg[a_neg == 1] = 1
        
        both = a_pos + a_neg
        del a_pos, a_neg
        
        both[both == 0] = 2
        both -= 1

    
        return both
    else:
        return [(2 if arr[0] >= GlobalParameters.positive_threshold else (0 if arr[0] <= GlobalParameters.negative_threshold else 1))]
    

def generate(last_expected, last_out, images=True, matrix=True, model_file=""):
    if type(last_expected) == torch.Tensor:
        last_expected = last_expected.detach().cpu()
    if type(last_out) == torch.Tensor:
        last_out = last_out.detach().cpu()

    if len(last_expected) != 1:
        pearson = stats.pearsonr(last_expected, last_out)[0]

    if images and len(last_expected) != 1:
        RocCurveDisplay.from_predictions((last_expected >= GlobalParameters.positive_threshold), (last_out >= GlobalParameters.positive_threshold))
        plt.savefig(f'{GlobalParameters.output_directory}/ROC_AUC_Curve_{model_file}.png')

    last_expected = _split_by_threshold(last_expected)
    last_out = _split_by_threshold(last_out)

    if len(last_expected) != 1:
        accuracy = accuracy_score(last_expected, last_out)
    
    MCM = multilabel_confusion_matrix(last_expected, last_out)

    # TN, FP, FN, TP calculated by globally counting them across all classes 
    tn, fp, fn, tp = MCM[:, 0, 0], MCM[:, 0, 1], MCM[:, 1, 0], MCM[:, 1, 1]
    tn = tn.sum()
    fp = fp.sum()
    fn = fn.sum()
    tp = tp.sum()

    
    if len(last_expected) != 1:
        # roc
        try:
            roc_auc = roc_auc_score(np.array([(last_expected==0).astype(int), (last_expected==1).astype(int), (last_expected==2).astype(int)]), \
                                    np.array([(last_out==0).astype(int), (last_out==1).astype(int), (last_out==2).astype(int)]), multi_class='ovr')
        except ValueError:
            print("Warning: Only one class present for calculating ROC AUC score.")
            roc_auc = -1

        # sensitivity specificity
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        # matthews
        MCC = matthews_corrcoef(last_expected, last_out)

        if matrix:
            CM = confusion_matrix(last_expected, last_out, normalize='true', labels=[0,1,2]).round(2).tolist()
            CM = [str([str(column) + '0' if len(str(column)) == 3 else str(column) for column in row]).replace("'","") for row in CM]
    
            return pearson, accuracy, roc_auc, tn, fp, fn, tp, sensitivity, specificity, MCC, CM
    else:
        pearson, accuracy, roc_auc, sensitivity, specificity, MCC = np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN
        
        if matrix:
            CM = confusion_matrix(last_expected, last_out, normalize='true', labels=[0,1,2]).round(2).tolist()
            CM = [str([str(column) + '0' if len(str(column)) == 3 else str(column) for column in row]).replace("'","") for row in CM]
            
            return pearson, accuracy, roc_auc, tn, fp, fn, tp, sensitivity, specificity, MCC, CM
            

    return pearson, accuracy, roc_auc, tn, fp, fn, tp, sensitivity, specificity, MCC