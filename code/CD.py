import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

CD = np.float64(1.1482)

svm = [1,2,2,2,2,1,1]
lr = [2.5,1,1,1,4,2,2]
rf = [2.5,3,4,3,3,4,3]
knn = [5,4,5,5,5,3,5]
dt = [4,5,3,4,1,5,4]

matrix = np.array([svm, lr, rf, knn, dt])

rank_x = list(map(lambda x: np.mean(x), matrix))

name_y = ["SVM", "LR", "RF", "KNN", "DT"]
min_ = [x for x in rank_x - CD/2]
max_ = [x for x in rank_x + CD/2]

font_auto = FontProperties(fname='D:/Users/gxj/WeChect/file_rev/WeChat Files/wxid_hlu3wvb0zpvp12/FileStorage/File/2021-07/TIMES(1).TTF',
                           size='x-large'
                            # ,stretch='expanded'
                           )

plt.rc('font', family='Times New Roman')
plt.figure(figsize=(10, 8), dpi=180)
plt.title("Critical Difference",fontsize=16)
# svm
plt.scatter(rank_x[0],name_y[0],lw=2,color='#f32800',label='SVM average ranking value={0:0.4f}'.format(rank_x[0]))
plt.hlines(name_y[0],min_[0],max_[0],lw=2,color='#f32800')
# LR
plt.scatter(rank_x[1],name_y[1],lw=2,color='#f39800',label='LR average ranking value={0:0.4f}'.format(rank_x[1]))
plt.hlines(name_y[1],min_[1],max_[1],lw=2,color='#f39800')
# RF
plt.scatter(rank_x[2],name_y[2],lw=2, color='#32b16c', label='RF average ranking value={0:0.4f}'.format(rank_x[2]))
plt.hlines(name_y[2],min_[2],max_[2], lw=2, color='#32b16c')
# KNN
plt.scatter(rank_x[3],name_y[3],lw=2, color='#00a0e9', label='KNN average ranking value={0:0.4f}'.format(rank_x[3]))
plt.hlines(name_y[3],min_[3],max_[3], lw=2, color='#00a0e9')
# DT
plt.scatter(rank_x[4],name_y[4],lw=2, color='#8957a1', label='DT average ranking value={0:0.4f}'.format(rank_x[4]))
plt.hlines(name_y[4],min_[4],max_[4], lw=2, color='#8957a1')
plt.ylabel('Model', fontsize=16)
plt.xlabel('Ranking', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='lower right', prop=font_auto)
plt.savefig('CD.tif')
plt.show()

