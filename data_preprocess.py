import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np


data = pd.read_csv('./data/train.csv')
y = data.pop('target')
id = data.pop('id')

""" 有一些参数是没有参照的，很难说清一个范围，这种情况下我们使用学习曲线，看趋势 从曲线跑出的结果中选取一个更小的区间，再跑曲线
param_grid = {'n_estimators':np.arange(0, 200, 10)}

param_grid = {'max_depth':np.arange(1, 20, 1)}     param_grid = {'max_leaf_nodes':np.arange(25,50,1)}    
对于大型数据集，可以尝试从1000来构建，先输入1000，每100个叶子一个区间，再逐渐缩小范围

有一些参数是可以找到一个范围的，或者说我们知道他们的取值和随着他们的取值，模型的整体准确率会如何变化，这 样的参数我们就可以直接跑网格搜索 param_grid = {'criterion':['gini', 'entropy']}

param_grid = {'min_samples_split':np.arange(2, 2+20, 1)}

param_grid = {'min_samples_leaf':np.arange(1, 1+10, 1)}    param_grid = {'max_features':np.arange(5,30,1)} 

"""

rfc = RandomForestClassifier(
    n_estimators=100
    ,max_depth=None
    ,max_features='auto'
    ,random_state=90
    ,min_samples_leaf=1
    ,min_samples_split=2
    ,criterion='gini'
)

param_grid = {'n_estimators': np.arange(100, 101, 50)}
GS = GridSearchCV(rfc,param_grid,cv=5)
GS.fit(data,y)
f = open('./data/result.txt',mode='w')
f.write(str(GS.best_params_))
f.write(str(GS.best_score_))
f.close()
print(GS.best_params_)
print(GS.best_score_)