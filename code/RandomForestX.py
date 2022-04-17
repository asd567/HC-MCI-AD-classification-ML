import FeatureProject
import ROCX
import os
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier #使用这个包构建决策树
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier  #使用这个包构建随机森林
from sklearn.metrics import precision_recall_curve #得到精度、召回率等
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
'''进行模型的保存和加载'''
import joblib

'''
函数参数说明：
classifer：用于说明是进行二分类还是进行三分类
splitFile：存放测试集、训练集的文件路径
resultPath：用于保存结果的文件路径,比如SVM就写到 ...... /result/SVM
'''

def random_tree_x(classifer,splitFile,resultPath, tp):
    if os.path.exists(resultPath) == False:  # 如果存放结果的目标路径不存在，则进行创建
        os.makedirs(resultPath)

    X_train = pd.read_csv(splitFile + '/X_train.csv').iloc[:, 2:].values
    X_test = pd.read_csv(splitFile + '/X_test.csv').iloc[:, 2:].values
    y_train = pd.read_csv(splitFile + '/y_train.csv').iloc[:, 1].values
    y_test = pd.read_csv(splitFile + '/y_test.csv').iloc[:, 1].values
    entropy_thresholds = np.linspace(0, 0.01, 50)
    gini_thresholds = np.linspace(0, 0.005, 50)

    '''n_estimators 不要超过20个，数据量小的话就少一些，数据量多就多一些'''
    estimator = RandomForestClassifier(n_estimators=10, max_depth=3).fit(X_train,y_train)  # 构建决策树

    y_pred_prob = estimator.predict_proba(X_test)
    y_pred = estimator.predict(X_test)

    '''训练集同样当成测试集输入并且进行测试'''
    train_pred_prob = estimator.predict_proba(X_train)
    train_pred = estimator.predict(X_train)

    pd.DataFrame(train_pred_prob).to_csv(resultPath + '/train_pred_proba.csv')
    pd.DataFrame(train_pred).to_csv(resultPath + '/train_pred.csv')
    df1 = pd.read_csv(splitFile + '/y_train.csv')
    df2 = pd.read_csv(resultPath + '/train_pred.csv', index_col='Unnamed: 0')
    df3 = pd.read_csv(resultPath + '/train_pred_proba.csv', index_col='Unnamed: 0')
    df4 = pd.concat(objs=[df1, df2, df3], axis=1)
    if classifer == 2:
        df4.columns = ['SubjID', 'Group', 'pred', 'train_pred_prob1', 'train_pred_prob2']
    if classifer == 3:
        df4.columns = ['SubjID', 'Group', 'pred', 'train_pred_prob1', 'train_pred_prob2', 'train_pred_prob3']
    df4.to_csv(resultPath + '/train_result_compare.csv')

    """
                将预测结果、预测结果概率存入/result/randomForest/result_compare.csv文件
    """
    pd.DataFrame(y_pred_prob).to_csv(resultPath + '/pred_proba.csv')
    pd.DataFrame(y_pred).to_csv(resultPath + '/pred.csv')
    df1 = pd.read_csv(splitFile + '/y_test.csv')
    df2 = pd.read_csv(resultPath + '/pred.csv', index_col='Unnamed: 0')
    df3 = pd.read_csv(resultPath + '/pred_proba.csv', index_col='Unnamed: 0')
    df4 = pd.concat(objs=[df1, df2, df3], axis=1)
    if classifer == 2:
        df4.columns = ['SubjID', 'Group', 'pred', 'pred_prob1', 'pred_prob2']
    if classifer == 3:
        df4.columns = ['SubjID', 'Group', 'pred', 'pred_prob1', 'pred_prob2', 'pred_prob3']
    df4.to_csv(resultPath + '/result_compare.csv')
    """
            模型得分
    """
    '''打开文件'''
    proba_result = open(resultPath + '/model_score.txt', mode='w')

    '''将交叉验证结果写入文件'''
    scores = cross_val_score(estimator, X_train, y_train, cv=5, scoring='accuracy')
    proba_result.write(str('训练集交叉验证结果：\n'))
    proba_result.writelines(str(scores))
    '''将模型测试评分结果写入文件'''
    score = estimator.score(X_test, y_test)  # 多分类单看一个score不恰当，应该看单独的
    proba_result.write(str('\n模型测试评分结果：'))
    proba_result.writelines(str(score))

    y_pred = estimator.predict(X_test)
    '''将分类报告写入文件'''
    clf_rep = classification_report(y_test, y_pred, digits=6)
    proba_result.write(str('\n分类报告：\n'))
    proba_result.write(clf_rep)
    '''将混淆矩阵写入文件'''
    cfu_mx = confusion_matrix(y_test, y_pred)
    proba_result.write(str('混淆矩阵：\n'))
    proba_result.writelines(str(cfu_mx))
    '''关闭文件'''


    '''训练集评分保存至文件'''
    score = estimator.score(X_train, y_train)  # 多分类单看一个score不恰当，应该看单独的
    proba_result.write(str('\n测试集模型测试评分结果：'))
    proba_result.writelines(str(score))

    y_pred = estimator.predict(X_train)
    '''将分类报告写入文件'''
    clf_rep = classification_report(y_train, y_pred, digits=6)
    proba_result.write(str('\n分类报告：\n'))
    proba_result.write(clf_rep)
    '''将混淆矩阵写入文件'''
    cfu_mx = confusion_matrix(y_train, y_pred)
    proba_result.write(str('混淆矩阵：\n'))
    proba_result.writelines(str(cfu_mx))
    '''关闭文件'''
    proba_result.close()

    if classifer == 3:
        # ROCX.three_auc_report(1,classifer, y_test, y_pred_prob, resultPath+'/ROC', picName='randomForest.png', tp=tp)
        joblib.dump(estimator, resultPath + '/model.pkl')
    # else:
    #     ROCX.auc_report(1,y_test, y_pred_prob[:,1], resultPath+'/ROC', picName='randomForest.png')
    #     joblib.dump(estimator, resultPath + '/model.pkl')