# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import itertools
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier

plt.rcParams['font.sans-serif']=['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False    #用来正常显示负号

pd.set_option('display.width', 2000, 'display.max_rows', None,'display.max_columns', None)  # 设置数据显示
trd=pd.read_csv("../data/train.csv")    # 读取训练数据
tsd=pd.read_csv("../data/test.csv")     # 读取测试数据
trd.info()                            # 读取训练集信息
print(trd.describe())                 # 显示训练集数据特征
tsd.info()                            # 读取测试集信息
print(tsd.describe())                 # 显示测试集数据特征

# 数据数值化
def data_sd(trd):
    trd.loc[(trd.Cabin.notnull()), 'Cabin'] = 1
    trd.loc[(trd.Cabin.isnull()), 'Cabin'] = 0
    trd.loc[(trd['SibSp']>=3), 'SibSp'] = 3
    trd.loc[(trd['Parch']>=3),'Parch'] = 3
    trd.Sex[trd.Sex=="female"]=0
    trd.Sex[trd.Sex=="male"]=1
    trd.Embarked[trd.Embarked=="C"]=0
    trd.Embarked[trd.Embarked=="S"]=1
    trd.Embarked[trd.Embarked=="Q"]=2
    trd.Embarked[trd.Embarked.isnull()]=3
data_sd(trd)       # 训练数据规范化
data_sd(tsd)       # 测试数据规范化

# 随机森林填补缺失的年龄属性
def set_missing_ages(df):
    df1= df[['Age', 'Pclass', 'Fare', "Embarked",'Cabin','Parch', 'SibSp']][df.Fare.notnull()]  # 提取特征较显著的几个属性数据
    y = df1[df1.Age.notnull()].values[:, 0]    # 提取有年龄乘客的年龄数据
    x = df1[df1.Age.notnull()].values[:, 1:]   # 提取有年龄乘客的其它属性数据
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)  # 定义随机森林
    rfr.fit(x, y)     # 训练集（即有年龄乘客）数据导入随机森林，生成模型。
    predictedAges = rfr.predict(df1[df1.Age.isnull()].values[:, 1:])  # 空缺数据导入得到的模型，进行未知年龄结果预测。
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges    # 用得到的预测结果填补原缺失数据
    return df, rfr
trd, rfr = set_missing_ages(trd)                   # 调用年龄填补函数
trd.Age=trd.Age.astype(np.int32)                   # 年龄数据换为整数
tsd, rfr = set_missing_ages(tsd)                   # 调用年龄填补函数
tsd.Age=tsd.Age.astype(np.int32)                   # 年龄数据换为整数


# 年龄数据特征因子化
import sklearn.preprocessing as prc
def data_asd(trd):
    mmsc= prc.MinMaxScaler(feature_range=(0, 1))    # 年龄数据规范区间（0，1）
    T=np.array([trd.Age]).transpose()               # 年龄数据加维、数组化、取转置。才能顺利进行规范化操作。
    trd_d=mmsc.fit_transform(T).transpose()[0]      # 数据规范化，转置回来，取一维。
    trd["Age_mmsc"]=trd_d                           # 规范化的年龄数据拼接到原数据
data_asd(trd)
data_asd(tsd)

# 票价数据特征因子化
def data_fsd(trd):
    trd.Fare[trd.Fare.isnull()]=trd.Fare.mean()      # 空缺票价填充为平均值
    mmsc= prc.MinMaxScaler(feature_range=(0, 1))    # 票价数据规范区间（0，1）
    T=np.array([trd.Fare]).transpose()               # 票价数据加维、数组化、取转置。才能顺利进行规范化操作。
    trd_d=mmsc.fit_transform(T).transpose()[0]      # 数据规范化，转置回来，取一维。
    trd["Fare_mmsc"]=trd_d                           # 规范化的票价数据拼接到原数据
data_fsd(trd)
data_fsd(tsd)

# # dummies data
# def dummies_data(data_train):
#     dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
#     dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
#     dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
#     dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
#     data_train= pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
#     # data_train=data_train.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
#     return data_train
# trd=dummies_data(trd)
# train_df= trd.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# tsd=dummies_data(tsd)
# test_df= tsd.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# test_df.info()
# train_df.info()

score=[]                  # 记录评分的列表
temp0=[]                  # 记录当前选取的特征组合的评分
temp1=0                   # 记录当前选取组合的平均分数
temp2=0                   # 记录当前选取组合的分数标准差
z=["Pclass","Sex","Embarked","Age_mmsc","Cabin","Fare_mmsc",'SibSp','Parch']        # 用于生成特征组合的完整属性列表
for j in range(1,9):
    for i in itertools.combinations(z, j):                     # 取包含j个属性的特征组合
        i=list(i)

        # 交叉验证库，将训练集进行切分交叉验证取平均
        from sklearn import cross_validation
        from sklearn.model_selection import cross_val_score
        knc_kf=KNeighborsClassifier()                          # 定义一个K-邻近分类器
        x =trd[i]
        y =trd["Survived"]
        score=cross_val_score(knc_kf, x, y, cv=5)              # k为5的交叉验证分数，

        if (score.mean() > temp1 and score.std() < 0.016):     # 特征组合选取条件，标准差在指定范围，平均分最大
            temp0 = score
            temp1 = score.mean()
            temp2 = score.std()
            dict = {temp1: i}                      # 字典，key为平均分数，value为当前选取的特征组合
print(dict, temp0, temp1, temp2)

c =dict[temp1]    # ["Pclass","Sex","Embarked","Age_mmsc"]         # 最终选取的特征组合

# K-邻近算法建模
knc1 = KNeighborsClassifier()  # 定义一个K-邻近分类器
x_trd = trd[c]
y_trd = trd["Survived"]
knc1.fit(x_trd, y_trd)          # 训练模型
x_tsd = tsd[c]
y_tsd = knc1.predict(x_tsd)     # 进行预测
result = pd.DataFrame({'PassengerId': tsd['PassengerId'].values, 'Survived': y_tsd.astype(np.int32)})  # 更改预测结果格式
result.to_csv("../result/result_knc.csv", index=False)      # 输出结果


