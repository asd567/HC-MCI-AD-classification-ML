import pandas
import pandas as pd
import os
import numpy as np
from sklearn import metrics

''' two classifier training '''

def compute_specificy(y, y_pred, P, N):
    TP=0
    FP=0
    FN=0
    TN=0
    for i in range(len(y)):
        if y[i] == P:
            if y_pred[i] == P:
                TP+=1
            if y_pred[i] == N:
                FN+=1
        if y[i] == N:
            if y_pred[i] == P:
                FP+=1
            if y_pred[i] == N:
                TN+=1
    return TN/(TN+FP)

def two_training():
    path = 'D:/project/pycharm/Xuanwu_pearsonr_lasso/test2/two'
    mt = ['t1', 'dti', 'pet', 't1_dti', 't1_pet', 'dti_pet', 't1_dti_pet']
    sub_classifier = ['ad_mci', 'ad_nc', 'mci_nc']
    model_class = ['SVM', 'logisticRegression', 'randomForest', 'knn', 'decisionTree']
    flag = True
    for s in sub_classifier:
        flag = False
        for model in model_class:
            for m in mt:
                y_test_path = os.path.join(path, m, s, 'train_test_split/y_train.csv')
                # 获取真实值
                y_test = pd.read_csv(y_test_path, sep=',')['Group']
                y_pred_path = os.path.join(path, m, s, 'result/{0}/train_pred.csv'.format(model))
                y_pred = pd.read_csv(y_pred_path, sep=',')['0']
                val = compute_specificy(y=y_test, y_pred=y_pred, P=1, N=0)
                x = val*100

                print('%.2f' % x)
                # pd.DataFrame(val).to_csv('C:/Users/28411/Desktop/pangyu/{0}.csv'.format(s))


def two_test():
    path = 'D:/project/pycharm/Xuanwu_pearsonr_lasso/test2/two'
    mt = ['t1', 'dti', 'pet', 't1_dti', 't1_pet', 'dti_pet', 't1_dti_pet']
    sub_classifier = ['ad_mci', 'ad_nc', 'mci_nc']
    model_class = ['decisionTree', 'knn', 'logisticRegression', 'randomForest', 'SVM']
    for m in mt:
        for s in sub_classifier:
            # os.remove(os.path.join(path, m, s, 'test_sepcificy.csv'))
            y_test_path = os.path.join(path, m, s, 'train_test_split/y_test.csv')
            # 获取真实值
            y_test = pd.read_csv(y_test_path, sep=',')['Group']
            # 获取每个模型的预测值
            for model in model_class:
                y_pred_path = os.path.join(path, m, s, 'result/{0}/pred.csv'.format(model))
                y_pred = pd.read_csv(y_pred_path, sep=',')['0']
                val = compute_specificy(y=y_test, y_pred=y_pred, P=1, N=0)
                x1 = pd.DataFrame([m, s, model, val], index=['motai', 'sub_classifier', 'model', 'sepcificy']).T
                x1.to_csv(os.path.join(path, m, s, 'test_sepcificy.csv'), sep=',', mode='a')

def two_independentVerification():
    path = 'Independent_feature_examine/IndependentVerification/two'
    mt = ['t1', 'dti', 'pet', 't1_dti', 't1_pet', 'dti_pet', 't1_dti_pet']
    sub_classifier = ['ad_mci', 'ad_nc', 'mci_nc']
    model_class = ['decisionTree', 'knn', 'logisticRegression', 'randomForest', 'SVM']
    for m in mt:
        for s in sub_classifier:
            # os.remove(os.path.join(path, m, s, 'independent_sepcificy.csv'))
            y_test_path = os.path.join(path, m, s, 'feature/X_independ.csv')
            # 获取真实值
            y_test = pd.read_csv(y_test_path, sep=',')['Group']
            # print(y_test)
            # 获取每个模型的预测值
            for model in model_class:
                y_pred_path = os.path.join(path, m, s, 'result/{0}/pred.csv'.format(model))
                y_pred = pd.read_csv(y_pred_path, sep=',')['0']
                val = compute_specificy(y=y_test, y_pred=y_pred, P=1, N=0)
                x1 = pd.DataFrame([m, s, model, val], index=['motai', 'sub_classifier', 'model', 'sepcificy']).T
                x1.to_csv(os.path.join(path, m, s, 'independent_sepcificy.csv'), sep=',', mode='a')

def three_computing(y, y_pred):
    confusion_matrix = metrics.confusion_matrix(y, y_pred)
    n_classes = confusion_matrix.shape[0]
    metrics_result = []
    for i in range(n_classes):
        ALL = np.sum(confusion_matrix)
        TP = confusion_matrix[i, i]
        FP = np.sum(confusion_matrix[:, i]) - TP
        FN = np.sum(confusion_matrix[i, :]) - TP
        TN = ALL-TP-FP-FN
        metrics_result.append([TP/(TP+FP), TP/(TP+FN), TN/(TN+FP)])
    return metrics_result

def three_independent():
    path = 'Independent_feature_examine/IndependentVerification/three'
    mt = ['t1', 'dti', 'pet', 't1_dti', 't1_pet', 'dti_pet', 't1_dti_pet']
    model_class = ['decisionTree', 'knn', 'logisticRegression', 'randomForest', 'SVM']
    for m in mt:
        y_test_path = os.path.join(path, m, 'feature/X_independ.csv')
        y_test = pd.read_csv(y_test_path, sep=',')['Group']
        for model in model_class:
            y_pred_path = os.path.join(path, m, 'result/{0}/pred.csv'.format(model))
            y_pred = pd.read_csv(y_pred_path, sep=',')['0']
            metrics_result = three_computing(y=y_test, y_pred=y_pred)
            pd.DataFrame(metrics_result).to_csv(os.path.join(path, m,
                                                             'result/{0}/independent_three.csv'.
                                                             format(model)), sep=',')

def three_training():
    path = 'test2/three'
    mt = ['t1', 'dti', 'pet', 't1_dti', 't1_pet', 'dti_pet', 't1_dti_pet']
    model_class = ['decisionTree', 'knn', 'logisticRegression', 'randomForest', 'SVM']
    for m in mt:
        y_test_path = os.path.join(path, m, 'train_test_split/y_train.csv')
        y_test = pd.read_csv(y_test_path, sep=',')['Group']
        for model in model_class:
            y_pred_path = os.path.join(path, m, 'result/{0}/train_pred.csv'.format(model))
            y_pred = pd.read_csv(y_pred_path, sep=',')['0']
            metrics_result = three_computing(y=y_test, y_pred=y_pred)
            pd.DataFrame(metrics_result).to_csv(os.path.join(path, m,
                                                             'result/{0}/train_specificy.csv'.
                                                             format(model)), sep=',')

def three_testing():
    path = 'test2/three'
    mt = ['t1', 'dti', 'pet', 't1_dti', 't1_pet', 'dti_pet', 't1_dti_pet']
    model_class = ['decisionTree', 'knn', 'logisticRegression', 'randomForest', 'SVM']
    for m in mt:
        y_test_path = os.path.join(path, m, 'train_test_split/y_test.csv')
        y_test = pd.read_csv(y_test_path, sep=',')['Group']
        for model in model_class:
            y_pred_path = os.path.join(path, m, 'result/{0}/pred.csv'.format(model))
            y_pred = pd.read_csv(y_pred_path, sep=',')['0']
            metrics_result = three_computing(y=y_test, y_pred=y_pred)
            pd.DataFrame(metrics_result).to_csv(os.path.join(path, m,
                                                             'result/{0}/test_specificy.csv'.
                                                             format(model)), sep=',')

if __name__ == '__main__':
    two_training()
    # two_test()
    # two_independentVerification()
    # three_independent()
    # three_training()
    # three_testing()