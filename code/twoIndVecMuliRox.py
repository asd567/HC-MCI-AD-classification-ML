import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve,auc,roc_auc_score
from matplotlib.font_manager import FontProperties

'''
测试集画roc曲线（综合）
'''

font_auto = FontProperties(fname='C:/Users/28411/Documents/WeChat Files/wxid_hlu3wvb0zpvp12/'
                                 'FileStorage/File/2021-09/TIMES.TTF',
                           size='xx-large',
                           stretch='expanded'
                           )

if __name__ == '__main__':

    sub_classifier = ['ad_mci', 'mci_nc', 'ad_nc']
    ml = ['decisionTree', 'knn', 'logisticRegression', 'randomForest', 'SVM']

    for j in sub_classifier:
        for k in ml:
            '''dti_pet'''
            y_test_file1 = 'IndependentVerification/two/dti_pet/{0}/result/{1}/y_ind_ver.csv'.format(j, k)
            y_pred_prob_file1 = 'IndependentVerification/two/dti_pet/{0}/result/{1}/pred_proba.csv'.format(j, k)
            y_pred_prob1 = pd.read_csv(y_pred_prob_file1, sep=',')
            y_test1 = pd.read_csv(y_test_file1, sep=',')
            '''t1_dti'''
            y_test_file2 = 'IndependentVerification/two/t1_dti/{0}/result/{1}/y_ind_ver.csv'.format(j, k)
            y_pred_prob_file2 = 'IndependentVerification/two/t1_dti/{0}/result/{1}/pred_proba.csv'.format(j, k)
            y_pred_prob2 = pd.read_csv(y_pred_prob_file2, sep=',')
            y_test2 = pd.read_csv(y_test_file2, sep=',')
            '''t1_pet'''
            y_test_file3 = 'IndependentVerification/two/t1_pet/{0}/result/{1}/y_ind_ver.csv'.format(j, k)
            y_pred_prob_file3 = 'IndependentVerification/two/t1_pet/{0}/result/{1}/pred_proba.csv'.format(j, k)
            y_pred_prob3 = pd.read_csv(y_pred_prob_file3, sep=',')
            y_test3 = pd.read_csv(y_test_file3, sep=',')
            '''t1'''
            y_test_file4 = 'IndependentVerification/two/t1/{0}/result/{1}/y_ind_ver.csv'.format(j, k)
            y_pred_prob_file4 = 'IndependentVerification/two/t1/{0}/result/{1}/pred_proba.csv'.format(j, k)
            y_pred_prob4 = pd.read_csv(y_pred_prob_file4, sep=',')
            y_test4 = pd.read_csv(y_test_file4, sep=',')
            '''dti'''
            y_test_file5 = 'IndependentVerification/two/dti/{0}/result/{1}/y_ind_ver.csv'.format(j, k)
            y_pred_prob_file5 = 'IndependentVerification/two/dti/{0}/result/{1}/pred_proba.csv'.format(j, k)
            y_pred_prob5 = pd.read_csv(y_pred_prob_file5, sep=',')
            y_test5 = pd.read_csv(y_test_file5, sep=',')
            '''pet'''
            y_test_file6 = 'IndependentVerification/two/pet/{0}/result/{1}/y_ind_ver.csv'.format(j, k)
            y_pred_prob_file6 = 'IndependentVerification/two/pet/{0}/result/{1}/pred_proba.csv'.format(j, k)
            y_pred_prob6 = pd.read_csv(y_pred_prob_file6, sep=',')
            y_test6 = pd.read_csv(y_test_file6, sep=',')
            '''t1_dti_pet'''
            y_test_file7 = 'IndependentVerification/two/t1_dti_pet/{0}/result/{1}/y_ind_ver.csv'.format(j, k)
            y_pred_prob_file7 = 'IndependentVerification/two/t1_dti_pet/{0}/result/{1}/pred_proba.csv'.format(j, k)
            y_pred_prob7 = pd.read_csv(y_pred_prob_file7, sep=',')
            y_test7 = pd.read_csv(y_test_file7, sep=',')

########################################################
            fpr1, tpr1, thresholds1 = roc_curve(y_test1.iloc[:, 1].values, y_pred_prob1.iloc[:, 2].values)
            auc_result1 = roc_auc_score(y_test1.iloc[:, 1].values, y_pred_prob1.iloc[:, 2].values)

            fpr2, tpr2, thresholds2 = roc_curve(y_test2.iloc[:, 1].values, y_pred_prob2.iloc[:, 1].values)
            auc_result2 = roc_auc_score(y_test2.iloc[:, 1].values, y_pred_prob2.iloc[:, 1].values)

            fpr3, tpr3, thresholds3 = roc_curve(y_test3.iloc[:, 1].values, y_pred_prob3.iloc[:, 1].values)
            auc_result3 = roc_auc_score(y_test3.iloc[:, 1].values, y_pred_prob3.iloc[:, 1].values)

            fpr4, tpr4, thresholds4 = roc_curve(y_test4.iloc[:, 1].values, y_pred_prob4.iloc[:, 1].values)
            auc_result4 = roc_auc_score(y_test4.iloc[:, 1].values, y_pred_prob4.iloc[:, 1].values)

            fpr5, tpr5, thresholds5 = roc_curve(y_test5.iloc[:, 1].values, y_pred_prob5.iloc[:, 1].values)
            auc_result5 = roc_auc_score(y_test5.iloc[:, 1].values, y_pred_prob5.iloc[:, 1].values)

            fpr6, tpr6, thresholds6 = roc_curve(y_test6.iloc[:, 1].values, y_pred_prob6.iloc[:, 1].values)
            auc_result6 = roc_auc_score(y_test6.iloc[:, 1].values, y_pred_prob6.iloc[:, 1].values)

            fpr7, tpr7, thresholds7 = roc_curve(y_test7.iloc[:, 1].values, y_pred_prob7.iloc[:, 1].values)
            auc_result7 = roc_auc_score(y_test7.iloc[:, 1].values, y_pred_prob7.iloc[:, 1].values)

            print(y_test7.iloc[:, 1].values)
            print(type(y_test7.iloc[:, 1].values))
            print(y_pred_prob7.iloc[:, 1].values)
            print(type(y_pred_prob7.iloc[:, 1].values))


            # figure
            plt.figure(figsize=(10, 8), dpi=180)
            plt.rc('font', family='Times New Roman')
            plt.plot([0, 1], [0, 1], 'k--', lw=1)
            plt.plot(fpr4, tpr4, 'co-', label='T1(AUC=%0.4f)' % auc_result4, linewidth=2)
            plt.plot(fpr5, tpr5, 'mo-', label='DTI(AUC=%0.4f)' % auc_result5, linewidth=2)
            plt.plot(fpr6, tpr6, 'yo-', label='PET(AUC=%0.4f)' % auc_result6, linewidth=2)
            plt.plot(fpr2, tpr2, 'go-', label='T1+DTI(AUC=%0.4f)' % auc_result2, linewidth=2)
            plt.plot(fpr3, tpr3, 'ko-', label='T1+PET(AUC=%0.4f)' % auc_result3, linewidth=2)
            plt.plot(fpr1, tpr1, 'bo-', label='DTI+PET(AUC=%0.4f)' % auc_result1, linewidth=2)
            plt.plot(fpr7, tpr7, 'ro-', label='T1+DTI+PET(AUC=%0.4f)' % auc_result7, linewidth=2)
            # plt.plot(fpr7, tpr7, 'go-',  linewidth=2)

            plt.xticks((np.arange(0, 1.1, step=0.1)), fontsize=16)
            plt.yticks((np.arange(0, 1.1, step=0.1)), fontsize=16)
            if k == 'decisionTree':
                sx = 'DT'
            elif k == 'knn':
                sx = 'KNN'
            elif k == 'logisticRegression':
                sx = 'LR'
            elif k == 'randomForest':
                sx = 'RF'
            else:
                sx = 'SVM'
            plt.title(sx)
            plt.xlabel('False Positive Rate', fontsize=16)
            plt.ylabel('True Positive Rate', fontsize=16)

            plt.legend(loc='lower right', prop=font_auto)
            clf_path = 'IndependentVerification/two/IndependentVerification_roc_result/{0}'.format(j)
            if os.path.exists(clf_path) == False:
                os.makedirs(clf_path)
            picName = k
            plt.tight_layout()
            plt.savefig(os.path.join(clf_path, picName), dpi=600)
            # # plt.show()
