import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.linear_model import LassoCV,RidgeCV,Ridge
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

'''
raw_file : 最原始的文件（没有进行特征工程，也没进行训练集和测试集划分）
feature_path : 用于存放训练集的标准化、pearsonr、lasso文件 存放测试集标准化后的文件
split_path : 存储X_train.csv X_test.csv y_train.csv y_test.csv 文件的路径
k : 数据类型为整型 用于lasso系数控制选择，目前为0
'''


def pre_reduce(raw_file, feature_path,split_path, k):
    '''先进行标签处理，将string类型变为int '''
    data = pd.read_csv(raw_file, index_col=0)
    labels = data['Group'].unique().tolist()
    data['Group'] = (data['Group']).apply(lambda n: labels.index(n))
    pd.DataFrame(data).to_csv(raw_file)
    print("原始数据大小：", data.shape)

    '''划分测试集和训练集'''
    data = pd.read_csv(raw_file)
    X = data.drop(['Group'], axis=1)
    y = data[['Group','SubjID']]
    X_train_x, X_test_x, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

    pd.merge(y_train, X_train_x).set_index('SubjID').to_csv(split_path + '/X_train.csv')
    pd.merge(y_test, X_test_x).set_index('SubjID').to_csv(split_path + '/X_test.csv')

    y_train.set_index('SubjID').to_csv(split_path + '/y_train.csv')
    y_test.set_index('SubjID').to_csv(split_path + '/y_test.csv')

    '''训练集进行标准化'''
    X_train = pd.read_csv(split_path + '/X_train.csv')
    # print(X_train)
    train_id = X_train['SubjID']
    train_data = X_train.iloc[:, 2:]
    ss = StandardScaler().fit(train_data)
    train_data_std = ss.transform(train_data)
    train_data_std = pd.DataFrame(train_data_std,  index=train_id, columns=train_data.columns)
    train_data_std.insert(0, 'Group', value=X_train['Group'].values)
    train_data_std.to_csv(feature_path + '/X_train_standard.csv')
   
    '''测试集 使用训练集的均值方差进行 标准化'''
    X_test = pd.read_csv(split_path + '/X_test.csv')
    test_ID = X_test['SubjID']
    test_data = X_test.iloc[:, 2:]
    X_test_all_std = ss.transform(test_data)
    X_test_all_std = pd.DataFrame(X_test_all_std, index=test_ID, columns=test_data.columns)
    X_test_all_std.insert(0, 'Group', value=X_test['Group'].values)
    X_test_all_std.to_csv(feature_path + '/X_test_standard.csv')

    '''使用pearsonr对训练集单变量选择 dti20% , 50% '''
    df2 = pd.read_csv(feature_path + '/X_train_standard.csv')
    names = df2.iloc[:, 2:].columns.values
    X = df2.iloc[:, 2:].values
    y = df2['Group'].values
    ccc = 0
    feature = []
    name_new = []
    for i in range(0, len(names)):
        r, p = pearsonr(df2.iloc[:, (i + 2)].values, y)
        if p < 0.05:
            feature.append(df2.iloc[:, (i + 2)])
            name_new.append(names[i])
            ccc += 1
    print('After pearsonr : ', ccc)
    feature = pd.DataFrame(feature).T
    df3 = pd.DataFrame(feature.values,index=train_id, columns=name_new)
    df3.insert(0, 'Group', value=df2.iloc[:, 1].values)
    df3.to_csv(feature_path + '/X_train_pearsonr.csv')

    '''使用lasso对训练集进行特征进一步提取，数量为训练集样本量的十到十五分之一'''
    df4 = pd.read_csv(feature_path + '/X_train_pearsonr.csv')
    X2 = df4.iloc[:, 2:].values
    y2 = df4.iloc[:, 1]
    '''alpha'''
    model = LassoCV().fit(X2, y2)
    # model = RidgeCV()
    # model = model.fit(X=X2, y=y2)
    c = []
    names = []
    features = []
    cnt = 0
    for i in range(0, len(model.coef_)):
        if abs(model.coef_[i]) > k:
            cnt = cnt + 1
            features.append(df4.iloc[:, (i + 2)])
            names.append(name_new[i])
            c.append(model.coef_[i])
    print(len(c))

    features = pd.DataFrame(features).T
    df5 = pd.DataFrame(features.values,index=train_id, columns=names)
    df5.insert(0, 'Group', value=df4.iloc[:, 1].values)
    df5.to_csv(split_path + '/X_train.csv')
    # 保存权重系数
    pd.DataFrame(c, index=names).to_csv(feature_path + '/coef.csv', sep=',')
    return ss


if __name__ == '__main__':
    '''二分类'''
    # classifier = ['two']
    # c = 2
    # mt = ['dti', 'dti_pet', 'pet', 't1', 't1_dti', 't1_dti_pet', 't1_pet']
    # sub_classifier = ['ad_mci', 'mci_nc', 'ad_nc']
    #
    # for k in classifier:
    #     for filename in mt:
    #         for tp in sub_classifier:
    #             raw_file = 'test2/two/' + filename + '/' + tp + '/raw.csv'
    #             feature_path ='test2/two/' + filename + '/' + tp + '/feature'
    #             split_path = 'test2/two/' + filename + '/' + tp + '/train_test_split'
    #             pre_reduce(raw_file=raw_file, feature_path=feature_path, split_path=split_path, k=0)
    ''''''
    # mt = ['dti', 'dti_pet', 'pet', 't1', 't1_dti', 't1_dti_pet', 't1_pet']
    # sub_classifier = ['ad_mci', 'mci_nc']
    # # pet 的 ad_mci 和 mci_nc
    # i = mt[1]
    # j = sub_classifier[0]
    # raw_file = 'test2/two/{0}/{1}/raw.csv'.format(i, j)
    # feature_path ='test2/two/{0}/{1}/feature'.format(i, j)
    # split_path = 'test2/two/{0}/{1}/train_test_split'.format(i, j)
    # k = 0
    # pre_reduce(raw_file=raw_file, feature_path=feature_path, split_path=split_path, k=k)

    # '''三分类'''
    mt = ['dti', 'dti_pet', 'pet', 't1', 't1_dti', 't1_dti_pet', 't1_pet']
    k=0.02
    for i in mt:
        raw_file = 'test2/three/' + i + '/raw.csv'
        feature_path = 'test2/three/' + i + '/feature'
        split_path = 'test2/three/' + i + '/train_test_split'
        pre_reduce(raw_file=raw_file, feature_path=feature_path, split_path=split_path, k=k)