# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False    #用来正常显示负号

pd.set_option('display.width', 2000, 'display.max_rows', None,'display.max_columns', None)  # 设置数据显示
trd=pd.read_csv("../data/train.csv")                   # 读取训练数据
tsd=pd.read_csv("../data/test.csv")                    # 读取测试数据
trd.info()                                  # 读取训练数据列信息
tsd.info()                                  # 读取测试数据列信息
print(trd.describe())                       # 显示测试数据特征
print(trd.describe())                       # 显示训练数据特征

# 两个Series，将一个索引处有值另一个为NaN的地方填充为0
def func1(Series1,Series2):
    for i in Series1.index:
        if i not in Series2.index:
            Series2[i]=0
    for i in Series2.index:
        if i not in Series1.index:
            Series2[1] = 0
    return Series1,Series2

# begin -*- 6.2属性与获救结果的关联统计 -*-
fig=plt.figure(figsize=(12,6))       # 定义图并设置画板尺寸
fig.set(alpha=0.2)                  # 设定图表颜色alpha参数
# fig.tight_layout()                  # 调整整体空白
plt.subplots_adjust(left=0.08,right=0.94,wspace =0.36, hspace =0.5)       # 调整子图间距

#1 各船舱等级的获救情况
ax1=fig.add_subplot(241)
ax1.set(title=u"各船舱等级乘客获救情况",xlabel=u"船舱等级",ylabel=u"人数")
ax1.set_title(u"各船舱等级乘客获救情况",fontdict={'fontsize':10})                # 设置标题字体大小
ax1.axis([0,4,0,600])
S0_Pclass= trd.Pclass[trd.Survived == 0].value_counts()
S1_Pclass= trd.Pclass[trd.Survived == 1].value_counts()
plt.xticks(rotation=90)
dfp1=pd.DataFrame({u'未获救':S0_Pclass, u'获救':S1_Pclass}).plot(ax=ax1,kind='bar', stacked=True,rot=1)
for i in S0_Pclass.index:                                                                   # 添加列标签
    plt.text(i-1.16,S0_Pclass[i]+S1_Pclass[i]+12,"{:.2f}".format(S1_Pclass[i]/(S0_Pclass[i]+S1_Pclass[i])))


#2 各船舱号乘客获救情况
ax2=fig.add_subplot(242)
ax2.set(title="各船舱号乘客获救情况",xlabel=u"船舱号",ylabel=u"人数")
ax2.set_title(u"各船舱号乘客获救情况",fontdict={'fontsize':10})                # 设置标题字体大小
ax2.axis([0,8,0,800])
trd2=trd.copy()
count=0
for i in trd2.Cabin.fillna("N").values:
    trd2.Cabin[count]=i[0]
    count+=1
S0_Cabin=trd2.Cabin[trd2.Survived==0].value_counts()
S1_Cabin=trd2.Cabin[trd2.Survived==1].value_counts()
dfp2=pd.DataFrame({"未获救":S0_Cabin,"获救":S1_Cabin}).plot(ax=ax2,kind="bar",stacked=True,rot=1)
S0_Cabin,S1_Cabin=func1(S0_Cabin,S1_Cabin)
S0_Cabin,S1_Cabin=S0_Cabin.sort_index(),S1_Cabin.sort_index()
count2=-0.5
for i in S0_Cabin.index:
    # print(i,S0_Cabin.index,S0_Cabin[i])
    # print(ax2.get_xticks())
    plt.text(count2,S0_Cabin[i]+S1_Cabin[i]+16,"{:.1f}".format(S1_Cabin[i]/(S0_Cabin[i]+S1_Cabin[i])))
    count2+=1

#3 各登船口的获救情况
ax3=fig.add_subplot(243)
ax3.set(title=u"各登船口乘客获救情况",xlabel=u"登船口",ylabel=u"人数")
ax3.set_title(u"各登船口乘客获救情况",fontdict={'fontsize':10})                # 设置标题字体大小
ax3.axis([0,3,0,800])
S0_Embarked= trd.Embarked[trd.Survived == 0].value_counts()
S1_Embarked= trd.Embarked[trd.Survived == 1].value_counts()
dfp2=pd.DataFrame({u'未获救':S0_Embarked, u'获救':S1_Embarked}).plot(ax=ax3,kind='bar', stacked=True,rot=1)
c=0
for i in S0_Embarked.index:                                                                   # 添加列标签
    plt.text(c-0.2,S0_Embarked[i]+S1_Embarked[i]+20,"{:.2f}"\
             .format(S1_Embarked[i]/(S0_Embarked[i]+S1_Embarked[i])))
    c+=1

#4 各船票价格乘客的获救情况
ax4=fig.add_subplot(244)
ax4.set(title="各船票价格乘客的获救情况",xlabel=u"票价",ylabel=u"获救率")
ax4.set_title(u"各船票价格乘客获救情况",fontdict={'fontsize':10})                # 设置标题字体大小
ax4.axis([0,300,0,1])
x=np.array(sorted(trd.Fare[trd.Fare.notnull()]))
y=[]
for i in x:
    y.append(trd.Fare[trd.Fare < i][trd.Survived == 1].count()/trd.Fare[trd.Fare < i].count())
y=np.array(y)
plt.plot(x,y,"--",linewidth=0.6)
    # ax4.set_xticks([])                                                   # 不显示x轴刻度

#5 各性别的获救情况
ax5=fig.add_subplot(245)
ax5.set(title=u"不同性别乘客获救情况",xlabel=u"性别",ylabel=u"人数")
ax5.set_title(u"不同性别乘客获救情况",fontdict={'fontsize':10})                # 设置标题字体大小
ax5.axis([0,5,0,700])
S0_Sex=trd.Sex[trd.Survived==0].value_counts()
S1_Sex=trd.Sex[trd.Survived==1].value_counts()
dfp3=pd.DataFrame({u'未获救':S0_Sex, u'获救':S1_Sex}).plot(ax=ax5,kind='bar', stacked=True,rot=0)
c=1
for i in S0_Sex.index:                                                                   # 添加列标签
    plt.text(c-0.15,S0_Sex[i]+S1_Sex[i]+16,"{:.2f}".format(S1_Sex[i]/(S0_Sex[i]+S1_Sex[i])))
    c-=1

#6 各年龄乘客的获救情况
ax6=fig.add_subplot(246)
ax6.set(title="各年龄乘客获救情况",xlabel=u"乘客年龄",ylabel=u"获救率")
ax6.set_title(u"各年龄乘客获救情况",fontdict={'fontsize':10})                # 设置标题字体大小
x6=np.array(sorted(trd.Age[trd.Age.notnull()]))
# print(x6)
y6=[]
for i6 in x6:
    y6.append(trd.Age[trd.Age<i6][trd.Survived==1].count()/trd.Age[trd.Age<i6].count())
plt.plot(x6,y6,"--",linewidth=0.6)
    # ax6.set_xticks([])                                                   # 不显示x轴刻度

#7 登船兄弟姐妹\配偶人数-乘客获救情况
ax7=fig.add_subplot(247)
ax7.set(title=u"登船兄弟姐妹\配偶人数-乘客获救情况",xlabel=u"登船兄弟姐妹\配偶人数",ylabel=u"人数")
ax7.set_title(u"登船兄弟姐妹\配偶人数-乘客获救情况",fontdict={'fontsize':10})                # 设置标题字体大小
ax7.axis([0,10,0,700])
S0_SibSp=trd.SibSp[trd.Survived==0].value_counts()
S1_SibSp=trd.SibSp[trd.Survived==1].value_counts()
dfp4=pd.DataFrame({"未获救":S0_SibSp,"获救":S1_SibSp}).plot(ax=ax7,kind="bar",stacked=True,rot=1)
S0_SibSp,S1_SibSp=func1(S0_SibSp,S1_SibSp)                      # 加起来
S0_SibSp=S0_SibSp.sort_index()                                  # 按照索引排序
S1_SibSp=S1_SibSp.sort_index()
c=0
for i in S0_SibSp.index:                                                                   # 添加列标签
    plt.text(c-0.3,S0_SibSp[i]+S1_SibSp[i]+16,"{:.2f}".format(S1_SibSp[i]/(S0_SibSp[i]+S1_SibSp[i])))
    c+=1

#8 登船父母\子女人数-乘客获救情况
ax8=fig.add_subplot(248)
ax8.set(title=u"登船父母\子女人数-乘客获救情况",xlabel=u"登船父母\子女人数",ylabel=u"人数")
ax8.set_title(u"登船父母\子女人数-乘客获救情况",fontdict={'fontsize':10})                # 设置标题字体大小
ax8.axis([0,10,0,800])
S0_Parch=trd.Parch[trd.Survived==0].value_counts()
S1_Parch=trd.Parch[trd.Survived==1].value_counts()
dfp8=pd.DataFrame({"未获救":S0_Parch,"获救":S1_Parch}).plot(ax=ax8,kind="bar",stacked=True,rot=0.5)
S0_Parch,S1_Parch=func1(S0_Parch,S1_Parch)                      # 加起来
S0_Parch=S0_Parch.sort_index()                                  # 按照索引排序
S1_Parch=S1_Parch.sort_index()
c=0
for i in S0_Parch.index:                                                                   # 添加列标签
    plt.text(c-0.3,S0_Parch[i]+S1_Parch[i]+16,"{:.2f}".format(S1_Parch[i]/(S0_Parch[i]+S1_Parch[i])))
    c+=1

plt.savefig('../result/数据初步分析.jpg')
plt.show()


