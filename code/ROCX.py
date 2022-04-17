"""
该文件用于绘制二分类和三分类的ROC曲线，三分类可以按照模式改为多分类
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve,auc,roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
import joblib
from matplotlib.font_manager import FontProperties

'''三分类的ROC曲线'''

font_auto = FontProperties(fname='C:/Users/28411/Documents/WeChat Files/wxid_hlu3wvb0zpvp12/'
                                 'FileStorage/File/2021-09/TIMES.TTF',
                           size='xx-large',
                            stretch='expanded'
                           )
def three_auc_report(nb_classes, resultPath, Y_valid, X_test, clf_path, tp):

    if os.path.exists(clf_path) == False:
        os.makedirs(clf_path)
    plt.rc('font', family='Times New Roman')
    # Binarize the output

    Y_valid = label_binarize(Y_valid, classes=[i for i in range(nb_classes)])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    estimator = joblib.load(resultPath + '/model.pkl')
    Y_pred = estimator.predict_proba(X_test)

    for i in range(nb_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_valid[:, i], Y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_valid.ravel(), Y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nb_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= nb_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure(figsize=(10, 8), dpi=180)
    plt.plot(fpr["micro"], tpr["micro"], linestyle='--',
             label='micro-average AUC={0:0.4f}'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linewidth=2)

    plt.plot(fpr["macro"], tpr["macro"], linestyle='--',
             label='macro-average AUC={0:0.4f}'
                   ''.format(roc_auc["macro"]),
             color='navy', linewidth=2)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    m = ['HC', 'MCI', 'AD']
    for i, color in zip(range(nb_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, linestyle='-',
                 label='class {0}(AUC={1:0.4f})'
                       ''.format(m[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.rcParams['font.size'] = 16
    plt.xticks((np.arange(0, 1.1, step=0.1)), fontsize=16)
    plt.yticks((np.arange(0, 1.1, step=0.1)), fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    tit = ''
    for i in tp:
        if i == '_':
            tit += '+'
        else:
            tit += i.upper()
    plt.title(tit, fontsize=16)
    plt.legend(loc="lower right", prop=font_auto)
    # another_path = clf_path[:11] + '/three_roc' + '/' + tp
    # if os.path.exists(another_path) == False :
    #     os.makedirs(another_path)
    # plt.savefig(os.path.join(another_path, picName), dpi=600)
    # plt.savefig(os.path.join(clf_path, tp + ' ' +picName), dpi=600)
    '''保存至另一个'''

    if os.path.exists(clf_path) == False:
        os.makedirs(clf_path)
    plt.savefig(os.path.join(clf_path, tp), dpi=600)

    # plt.show()
    return

'''二分类的ROC曲线'''


def auc_report(n,y_test, y_pred_prob, clf_path,picName):
    if os.path.exists(clf_path) == False:
        os.makedirs(clf_path)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    auc_result = roc_auc_score(y_test, y_pred_prob)

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.plot(fpr, tpr, label='AUC is %0.3f' % auc_result)
    plt.rcParams['font.size'] = 8
    plt.legend(loc="lower right", fontsize=8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    plt.title(str(n) + 'ROC curve for two classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(fontsize=8)

    plt.savefig(os.path.join(clf_path, picName), dpi=600)
    plt.show()
    plt.clf()

    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(thresholds, index=i)})
    roc_t = roc.loc[(roc.tf - 0).abs().argsort()[:1]]
    threshold_value = roc_t['threshold'].values
    print(threshold_value)
    print('auc = ', auc_result)

    return auc_result, threshold_value