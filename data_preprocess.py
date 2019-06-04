import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost
from xgboost import XGBClassifier
import numpy as np

def normalize(series):
    u = series.mean()
    sigma = series.std()
    return (series - u)/sigma

df = pd.read_csv('./data/train.csv')
label_col_name = 'target'
#1.均衡否
print('label value_counts:\n', df[label_col_name].value_counts())
#2.重复否   一些不能重复的属性，如id, 若重复了，需要去重
if sum(df['id'].duplicated()) != 0:
    df['id'].drop_duplicates()
    print('属性 id 有重复，已去重。')

#3.空值否
# df.info(verbose=True)
# df.describe()

id = df.pop('id')
#4 normalize 1th
cols_except_lable_col = df.columns.drop(label_col_name)
df[cols_except_lable_col] = df[cols_except_lable_col].apply(lambda x:normalize(x))


corr = df.corr()
describe = corr[label_col_name].abs().describe()
corr = corr[corr[label_col_name]!=1]
items = corr[label_col_name].items()
corr.pop(label_col_name)
#去除同label低相关的属性
for k,v in items:
    if abs(v) < describe['25%']:
        df.pop(k)
#相关度高的属性去重
highly_correlated = dict()
for col in corr.columns:
    index = corr[col][corr[col]>0.9].index
    index = index.drop(col)
    highly_correlated[col] = index

for k,v,in highly_correlated.items():
    if k in df.columns:
        flag = False
        for i in v:
            if i in df.columns:
                flag = True
                break
        if flag:
            df.pop(k)

#构造新特征 x -> (x^2 -1)/2 (x^3 -1)/3  e^x logx
cols_except_lable_col = df.columns.drop(label_col_name)
for col in cols_except_lable_col:
    df[col + 'X^2'] = df[col].apply(lambda x: (x**2-1)/2)
    df[col + 'X^3'] = df[col].apply(lambda x: (x**3-1)/3)
    df[col + 'e^X'] = df[col].apply(lambda x: np.exp(x))
    df[col + 'logX'] = df[col].apply(lambda x: np.log(np.abs(x)))

#normalize 2th
cols_except_lable_col = df.columns.drop(label_col_name)
df[cols_except_lable_col] = df[cols_except_lable_col].apply(lambda x:normalize(x))

#model

label = df.pop(label_col_name)
X_train, X_test, y_train, y_test = train_test_split(df, label, test_size=0.33, random_state=42)

xgb = XGBClassifier()
xgb.fit(X_train,y_train)

score = xgb.score(X_test,y_test)
with open('./data/result.txt',mode='w') as f:
    f.write(str(score))
