import pandas as pd
import os
import matplotlib.pyplot as plt

def drawing(save_path,coef_file, filename, tp):
    df1 = pd.read_csv(coef_file)
    w = df1.iloc[:, 1].values
    # print(weight)
    for i in range(0, len(w)):
        df1.iloc[:, 1].values[i] = abs(w[i])
    df2 = df1.sort_values(by=['0'], ascending=True)
    # print(df2)
    weight = df2.iloc[:, 1].values
    name = df2.iloc[:, 0].values
    if len(name) >= 25:
        name = name[: 23]
        weight = weight[:23]
    plt.barh(y=name, width=weight, color='b', height=0.6)
    plt.title(filename + ' features about ' + tp)
    plt.tight_layout()
    pic_path = save_path + '/' + tp
    if os.path.exists(pic_path) == False:
        os.makedirs(pic_path)
    plt.savefig(pic_path + '/' + filename + '.png')
    plt.show()

def draw_save(save_path,coef_file, filename, tp):
    df1 = pd.read_csv(coef_file)

    weight = df1.iloc[:, 1].values
    name = df1.iloc[:, 0].values

    plt.barh(y=name, width=weight, color=['r', 'g', 'b', 'm', 'k'])
    plt.title(filename + ' features about ' + tp)
    plt.tight_layout()

    pic_path = save_path + '/' + tp
    if os.path.exists(pic_path) == False:
        os.makedirs(pic_path)
    plt.savefig(pic_path + '/' + filename + '.png')
    plt.show()

if __name__ == '__main__':
    '''绘制文章中的特征图片'''
    # coef_file = 'test2/two/dti/ad_mci/feature/coef.csv'
    # save_path = 'test2/two_visualized'
    #             # print(save_path +  '/' + filename)
    # drawing(save_path=save_path, coef_file=coef_file, filename='dti', tp='ad_mci')

    '''二分类可视化'''
    print('开始进行二分类特征可视化')
    classifier = ['two']
    c = 2
    mt = ['dti', 'dti_pet', 'pet', 't1', 't1_dti', 't1_dti_pet', 't1_pet']
    sub_classifier = ['ad_mci', 'mci_nc', 'ad_nc']

    for k in classifier:
        for filename in mt:
            for tp in sub_classifier:
                coef_file ='test2/two'+ '/' + filename + '/' + tp + '/feature/coef.csv'
                save_path = 'C:/Users/gxj/Desktop/fig/twoVisualized'
                drawing(save_path=save_path, coef_file=coef_file, filename=filename, tp=tp)
                # draw_save(save_path=save_path, coef_file=coef_file, filename=filename, tp=tp)

    '''三分类可视化'''
    print('开始进行三分类特征可视化')
    mt = ['dti', 'dti_pet', 'pet', 't1', 't1_dti', 't1_dti_pet', 't1_pet']
    for filename in mt:
        coef_file = 'test2/three' + '/' + filename + '/feature/coef.csv'
        save_path = 'C:/Users/gxj/Desktop/fig/threeVisualizedd'
        drawing(save_path=save_path, coef_file=coef_file, filename=filename, tp='')
        # draw_save(save_path=save_path, coef_file=coef_file, filename=filename,tp='')
    print('结束')
