''''
用于绘制分类的数据图，官方教程：
https://scikit-learn.org/stable/_images/sphx_glr_plot_classifier_comparison_001.png
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import joblib

filename = 'dti'
total = 'test2'
tp = 'ad_mci'
res = ['',  'SVM', 'logisticRegression', 'randomForest', 'knn', 'decisionTree']
names = ['SVM', 'logisticRegression', 'randomForest', 'knn', 'decisionTree']

'''开始进行画图'''
for k in names:
    figure = plt.figure(figsize=(10,8))

    data_file = '{0}/two/{1}/{2}/train_test_split/'.format(total, filename, tp)
    X_test = pd.read_csv(data_file + 'X_test.csv').iloc[:, 2:].values
    y_test = pd.read_csv(data_file + 'y_test.csv').iloc[:, 1].values
    pred_file = '{0}/two/{1}/{2}/result/{3}/'.format(total, filename, tp, k)
    y_pred = pd.read_csv(pred_file + 'pred.csv').iloc[:, 1].values
    print(y_pred)
    Z = pd.read_csv(pred_file + 'pred_proba.csv').iloc[:, 1:].values

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    plt.contourf(Z, alpha=0.8)
    y_ticks =[int(v) for v in np.linspace(0, len(Z)-1, num=len(Z))]
    print(y_ticks)
    plt.yticks(y_ticks)
    # print('Z[:, 0]')
    # print(Z[:, 0])

    plt.scatter(Z[:,0], y_ticks, cmap=cm_bright,
                edgecolors='k',
                s=500, marker='o')
    plt.title(k)

    plt.tight_layout()
    plt.show()


