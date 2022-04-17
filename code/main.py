import SVMX
import KNNX
import LogisticRegressionX
import DecisionTreeX
import RandomForestX
import table
import ROCX
import warnings
warnings.filterwarnings("ignore")
import pandas as pd



def run(c, tp):
    '''如果是岭回归则不需要用到特征工程'''

    print('开始进行SVM')
    SVMX.svm_x(classifer=c, splitFile=splitFile, resultPath=resultSVMPath, tp=tp)
    print('开始进行KNN')
    KNNX.knn_x(classifer=c, splitFile=splitFile, resultPath=resultKNNPath, tp=tp)
    print('开始进行逻辑回归')
    LogisticRegressionX.logistic_reg(classifer=c, splitFile=splitFile,
                                     resultPath=resultlogisticRegressionPath, tp=tp)
    print('开始进行决策树')
    DecisionTreeX.decision_tree_x(classifer=c, splitFile=splitFile,
                                  resultPath=resultdecisionTreePath, tp=tp)
    print('开始进行随机森林')
    RandomForestX.random_tree_x(classifer=c, splitFile=splitFile,
                                resultPath=resultrandomForestPath, tp=tp)
    return

def make_table(resultPath, classifer, splitFile, op, lk, tp):
    table.make_table(resultPath=resultPath, classifer=classifer, splitFile=splitFile, op=op, lk=lk, tp=tp)

def make_three_tables(resultPath, classifer, splitFile, lk, tp):
    table.make_three_tables(resultPath, classifer, splitFile, lk, tp)

if __name__ == '__main__':
    """
    二类分类代码（完整可运行版本）
    """
    classifier = ['two']
    c = 2
    mt = ['t1', 'dti', 'pet', 't1_dti', 't1_pet', 'dti_pet', 't1_dti_pet']
    sub_classifier = ['ad_mci', 'ad_nc', 'mci_nc']

    for k in classifier:
        for j in mt:
            for i in sub_classifier:
                print(j, i)
                splitFile ='test2/' + k + '/' + j + '/' + i + '/train_test_split'
                resultSVMPath ='test2/' + k + '/' + j + '/' + i + '/result/SVM'
                resultKNNPath ='test2/' + k + '/' + j + '/' + i + '/result/knn'
                resultdecisionTreePath ='test2/' + k + '/' + j + '/' + i + '/result/decisionTree'
                resultlogisticRegressionPath ='test2/' + k + '/' + j + '/' + i + '/result/logisticRegression'
                resultrandomForestPath ='test2/' + k + '/' + j + '/' + i + '/result/randomForest'
                run(c, j)
                make_table(resultPath=resultSVMPath,classifer=2,splitFile=splitFile, lk=i, op='svm', tp=j)
                make_table(resultPath=resultrandomForestPath, classifer=2, splitFile=splitFile, lk=i, op='RF', tp=j)
                make_table(resultPath=resultKNNPath, classifer=2, splitFile=splitFile, lk=i, op='knn', tp=j)
                make_table(resultPath=resultdecisionTreePath, classifer=2, splitFile=splitFile, lk=i, op='DT', tp=j)
                make_table(resultPath=resultlogisticRegressionPath, classifer=2, splitFile=splitFile, lk=i, op='LR', tp=j)


    """
    二分代码单独测试（目前可以进行）
    """
    # c = 2
    mt = ['dti', 'dti_pet', 'pet', 't1', 't1_dti', 't1_dti_pet', 't1_pet']
    for i in mt:
        splitFile = 'test2/two/{0}/ad_nc/train_test_split'.format(i)

        resultSVMPath = 'test2/two/{0}/ad_nc/result/SVM'.format(i)
        resultKNNPath = 'test2/two/{0}/ad_nc/result/knn'.format(i)
        resultdecisionTreePath = 'test2/two/{0}/ad_nc/result/decisionTree'.format(i)

        resultlogisticRegressionPath = 'test2/two/{0}/ad_nc/result/logisticRegression'.format(i)
        resultrandomForestPath = 'test2/two/{0}/ad_nc/result/randomForest'.format(i)
        # run(c, tp=i)

        make_table(resultPath=resultSVMPath,classifer=2,splitFile=splitFile,op='ad_nc', lk='svm', tp=i)
        make_table(resultPath=resultrandomForestPath, classifer=2, splitFile=splitFile, op='ad_nc', lk='RF', tp=i)
        make_table(resultPath=resultKNNPath, classifer=2, splitFile=splitFile, op='ad_nc', lk='knn', tp=i)

    """
       三分类代码完整版
    # """
    # mt = ['t1', 'dti', 'pet', 't1_dti', 't1_pet', 'dti_pet', 't1_dti_pet']
    # for i in mt:
    #     c = 3
    #     splitFile = 'test2/three/'+i+'/train_test_split'
    #     resultSVMPath = 'test2/three/'+i+'/result/SVM'
    #     resultKNNPath = 'test2/three/'+i+'/result/knn'
    #     resultdecisionTreePath = 'test2/three/'+i+'/result/decisionTree'
    #
    #     resultlogisticRegressionPath = 'test2/three/'+i+'/result/logisticRegression'
    #     resultrandomForestPath = 'test2/three/'+i+'/result/randomForest'
    #     run(c, i)
    #     print(i)

    ''' test-ROC'''
    # Y_valid = pd.read_csv(splitFile + '/y_test.csv').iloc[:, 1].values
    # X_test = pd.read_csv(splitFile + '/X_test.csv').iloc[:, 2:]
    # ROCX.three_auc_report(nb_classes=3, resultPath=resultSVMPath, Y_valid=Y_valid, X_test=X_test,
    #                       clf_path='test2/three/three_class_test_ROC/SVM', tp=i)
    #
    # ROCX.three_auc_report(nb_classes=3, resultPath=resultKNNPath, Y_valid=Y_valid, X_test=X_test,
    #                       clf_path='test2/three/three_class_test_ROC/KNN', tp=i)
    #
    # ROCX.three_auc_report(nb_classes=3, resultPath=resultdecisionTreePath, Y_valid=Y_valid, X_test=X_test,
    #                       clf_path='test2/three/three_class_test_ROC/DT', tp=i)
    #
    # ROCX.three_auc_report(nb_classes=3, resultPath=resultlogisticRegressionPath, Y_valid=Y_valid, X_test=X_test,
    #                       clf_path='test2/three/three_class_test_ROC/LR',  tp=i)
    #
    # ROCX.three_auc_report(nb_classes=3, resultPath=resultrandomForestPath, Y_valid=Y_valid, X_test=X_test,
    #                       clf_path='test2/three/three_class_test_ROC/RF', tp=i)
    #
    # Y_valid = pd.read_csv(splitFile + '/y_train.csv').iloc[:, 1].values
    # X_test = pd.read_csv(splitFile + '/X_train.csv').iloc[:, 2:]
    # ROCX.three_auc_report(nb_classes=3, resultPath=resultSVMPath, Y_valid=Y_valid, X_test=X_test,
    #                       clf_path='test2/three/three_class_train_ROC/SVM', tp=i)
    #
    # ROCX.three_auc_report(nb_classes=3, resultPath=resultKNNPath, Y_valid=Y_valid, X_test=X_test,
    #                       clf_path='test2/three/three_class_train_ROC/KNN', tp=i)
    #
    # ROCX.three_auc_report(nb_classes=3, resultPath=resultdecisionTreePath, Y_valid=Y_valid, X_test=X_test,
    #                       clf_path='test2/three/three_class_train_ROC/DT', tp=i)
    #
    # ROCX.three_auc_report(nb_classes=3, resultPath=resultlogisticRegressionPath, Y_valid=Y_valid, X_test=X_test,
    #                       clf_path='test2/three/three_class_train_ROC/LR', tp=i)
    #
    # ROCX.three_auc_report(nb_classes=3, resultPath=resultrandomForestPath, Y_valid=Y_valid, X_test=X_test,
    #                       clf_path='test2/three/three_class_train_ROC/RF', tp=i)

    # make_three_tables(resultPath=resultSVMPath, classifer=3, splitFile=splitFile, lk='svm', tp=i)
    #     # make_three_tables(resultPath=resultrandomForestPath, classifer=3, splitFile=splitFile, lk='RF', tp=i)
    #     # make_three_tables(resultPath=resultKNNPath, classifer=3, splitFile=splitFile, lk='knn', tp=i)
    #     # make_three_tables(resultPath=resultdecisionTreePath, classifer=3, splitFile=splitFile, lk='DT', tp=i)
    #     # make_three_tables(resultPath=resultlogisticRegressionPath, classifer=3, splitFile=splitFile, lk='LR',tp=i)

    # """
    #    三分类代码单独测试（可以运行）
    # """
    # c = 3
    # filename = 't1_dti'
    # raw_file = 'test2/three/{0}/dti_nc_mci_ad_raw.csv'.format(filename)
    # standard_file = 'test2/three/{0}/feature/standard.csv'.format(filename)
    # pearsonr_file = 'test2/three/t1_dti/feature/pearsonr.csv'
    # lasso_file = 'test2/three/t1_dti/feature/lasso.csv'
    # splitFile = 'test2/three/t1_dti/train_test_split'
    #
    # resultSVMPath = 'test2/three/t1_dti/result/SVM'
    # resultKNNPath = 'test2/three/t1_dti/result/knn'
    # resultdecisionTreePath = 'test2/three/t1_dti/result/decisionTree'
    #
    # resultlogisticRegressionPath = 'test2/three/t1_dti/result/logisticRegression'
    # resultrandomForestPath = 'test2/three/t1_dti/result/randomForest'
    # run(c, 't1_dti')
    # make_three_tables(resultPath=resultSVMPath, classifer=3, splitFile=splitFile, lk='svm', tp='t1_dti')