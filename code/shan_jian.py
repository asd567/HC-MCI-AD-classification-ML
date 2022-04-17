import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


'''
coef_file 为 ridge_coef.csv  
test_file 为 X_test_raw.csv 
obj_file 为 X_test.csv
用于从测试集中抽取与训练集相关的特征
'''
def s_jian(coef_file, test_file, obj_file, split_path):
    coef = pd.read_csv(coef_file).iloc[:,0].values
    df1 = pd.read_csv(test_file)
    names = df1.iloc[:,2:].columns.values
    ID = df1['SubjID']
    group = df1['Group']
    features = []
    for i in range(0,len(coef)):
        for j in range(0,len(names)):
            if names[j] == coef[i]:
                features.append(df1[coef[i]].values)
                break
            j = j + 1
        i = i + 1
    df2 = pd.DataFrame(features).T
    df3 = pd.DataFrame(df2.values,columns=coef, index=ID)
    df3.insert(0,'Group',value=group.values)
    df3.to_csv(obj_file)
    return

if __name__ == '__main__':
    '''二分类'''
    mt = ['dti', 'dti_pet', 'pet', 't1', 't1_dti', 't1_dti_pet', 't1_pet']
    sub_classifier = ['ad_mci', 'mci_nc', 'ad_nc']
    for i in mt:
        for j in sub_classifier:
            coef_file = 'test2/two/' + i + '/' + j + '/feature/coef.csv'
            test_file = 'test2/two/'+ i + '/'+ j + '/feature/X_test_standard.csv'
            obj_file = 'test2/two/'+ i + '/'+ j + '/train_test_split/X_test.csv'
            split_path = ''
            s_jian(coef_file=coef_file, test_file=test_file, obj_file=obj_file, split_path=split_path)

    # '''三分类'''

    mt = ['dti', 'dti_pet', 'pet', 't1', 't1_dti', 't1_dti_pet', 't1_pet']
    for i in mt:
        print(i)
        coef_file = 'test2/three/' + i + '/feature/coef.csv'
        test_file = 'test2/three/' + i + '/feature/X_test_standard.csv'
        obj_file = 'test2/three/' + i + '/train_test_split/X_test.csv'
        s_jian(coef_file=coef_file, test_file=test_file, obj_file=obj_file, split_path='')


    #
    # filename = 't1_dti_pet'
    # tp = 'ad_mci'
    #
    # coef_file = 'test2/two/' + filename + '/' + tp + '/feature/coef.csv'
    # test_file = 'test2/two/' + filename + '/' + tp + '/feature/X_test_standard.csv'
    # obj_file = 'test2/two/' + filename + '/' + tp + '/train_test_split/X_test.csv'
    # s_jian(coef_file=coef_file, test_file=test_file, obj_file=obj_file)