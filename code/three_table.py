import pandas as pd
from sklearn.model_selection import cross_val_score,train_test_split
import os
import numpy as np
from sklearn.metrics import classification_report

'''进行模型的保存和加载'''
import joblib
''' 三分类 '''
def make_three_tables(resultPath, classifer, splitFile, lk, tp):
    estimator = joblib.load(resultPath+'/model.pkl')
    X_train = pd.read_csv(splitFile + '/X_train.csv').iloc[:, 2:]
    X_test = pd.read_csv(splitFile + '/X_test.csv').iloc[:, 2:]
    y_train = pd.read_csv(splitFile + '/y_train.csv').iloc[:, 1].values
    y_test = pd.read_csv(splitFile + '/y_test.csv').iloc[:, 1].values

    y_pred_test = estimator.predict(X_test)
    y_pred_train = estimator.predict(X_train)

    res_path = 'eva/{0}/{1}'.format(classifer, lk)
    if os.path.exists(res_path) == False:
        os.makedirs(res_path)
    file = open(res_path + '/model_score.txt', mode='a+')
    file.write('\n' + tp + ':')
    clf_rep = classification_report(y_test, y_pred_test, digits=6)
    file.write(str('\n测试集分类报告：\n'))
    file.write(clf_rep)

    train_rep = classification_report(y_train, y_pred_train, digits=6)
    file.write(str('训练集分类报告：\n'))
    file.write(train_rep)

    file.close()
    return None

if __name__ == '__main__':
    mt = ['t1', 'dti', 'pet', 't1_dti', 't1_pet', 'dti_pet', 't1_dti_pet']
    for i in mt:
        c = 3
        splitFile = 'test2/three/{0}/train_test_split'.format(i)
        resultSVMPath = 'test2/three/{0}/result/SVM'.format(i)
        resultKNNPath = 'test2/three/{0}/result/knn'.format(i)
        resultdecisionTreePath = 'test2/three/{0}/result/decisionTree'.format(i)

        resultlogisticRegressionPath = 'test2/three/{0}/result/logisticRegression'.format(i)
        resultrandomForestPath = 'test2/three/{0}/result/randomForest'.format(i)

        print(i)

        make_three_tables(resultPath=resultSVMPath, classifer=3, splitFile=splitFile, lk='svm', tp=i)
        # make_three_tables(resultPath=resultrandomForestPath, classifer=3, splitFile=splitFile, lk='RF', tp=i)
        # make_three_tables(resultPath=resultKNNPath, classifer=3, splitFile=splitFile, lk='knn', tp=i)
        # make_three_tables(resultPath=resultdecisionTreePath, classifer=3, splitFile=splitFile, lk='DT', tp=i)
        # make_three_tables(resultPath=resultlogisticRegressionPath, classifer=3, splitFile=splitFile, lk='LR',tp=i)
