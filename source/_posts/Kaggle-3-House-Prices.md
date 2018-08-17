---
title: 'Kaggle(3): House Prices'
date: 2018-08-10 16:07:14
tags:
- Kaggle
- Python
- 数据挖掘
- 机器学习
---

# Base Model

## 问题描述

需要确保充分浏览题目要求才开始做，比如该题中的Evaluation提到用[RMSE](https://zh.wikipedia.org/wiki/%E5%9D%87%E6%96%B9%E6%A0%B9%E8%AF%AF%E5%B7%AE)即均方根误差去评估效果，而且需要提前将预测结果取对数

House Price的特征非常多，其中类别特征也很多，也有不少缺失值，乍一看去工作量非常大，实际也挺大：）

这个时候我们可以先偷懒要求低点，先建个**基模**，不需要准备那么充分才开始。

## 数据概况

可以发现属性很多，其中不少categorical variables，同时也有一定范围的缺失值

```py
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1460 entries, 0 to 1459
Data columns (total 81 columns):
Id               1460 non-null int64
MSSubClass       1460 non-null int64
MSZoning         1460 non-null object
LotFrontage      1201 non-null float64
LotArea          1460 non-null int64
Street           1460 non-null object
Alley            91 non-null object
...
FireplaceQu      770 non-null object
GarageType       1379 non-null object
GarageYrBlt      1379 non-null float64
GarageFinish     1379 non-null object
GarageCars       1460 non-null int64
GarageArea       1460 non-null int64
GarageQual       1379 non-null object
GarageCond       1379 non-null object
...
PoolQC           7 non-null object
Fence            281 non-null object
MiscFeature      54 non-null object
...
dtypes: float64(3), int64(35), object(43)
memory usage: 924.0+ KB
```

看起来挺棘手的，那就先不管那么多建个基模吧

## 缺失值

这里简单的对numerical variables进行填充，而对于categorical variables不进行填充，直接将缺失值作为一类。
```py
data['MSSubClass'] = data['MSSubClass'].astype(str)
# fillna for num
category_cols=data.select_dtypes(include=['object']).columns.values
for item in data.columns:
    if item not in category_cols:
        data[item] = data[item].fillna(data.mean()[item])
```

## Categorical Variables处理

如何处理类别属性？

有个普遍观点是树模型不太需要One-Hot Encoding，这里我都尝试了一遍，选择的是One-Hot encoding 和 [Label encoding](https://blog.csdn.net/sinat_29957455/article/details/79452141)

> 这里需要特别注意，对类别特征进行处理时，最好把训练样本和测试样本拼在一起，因为两份样本不一定拥有类别特征中的所有取值。可能会出现测试集中有训练集不存在的取值。同时放在一起分析也能发现那些在测试集中缺失但在训练集中完整的元素

## 建模调参

### RandomForest
起初选用的是RandomForest，要清楚[随机森林的参数](https://www.cnblogs.com/pinard/p/6160412.html)的具体含义，之后可通过[GridSearchCV或者RandomizedSearchCV](https://blog.csdn.net/juezhanangle/article/details/80051256)进行调参(前者耗时太大)。

```py
def GridSearch(data,label):
    score=['neg_mean_squared_error']
    tuned_parameters={'n_estimators':range(100,400,30),'max_features':np.arange(0.1,1,0.3),
                      'max_depth':range(10,100,10),'min_samples_split':range(2,10,2),
                      'min_samples_leaf':range(1,10,2)}
    clf=GridSearchCV(RandomForestRegressor(oob_score=True),tuned_parameters,cv=5,scoring='neg_mean_squared_error')
    clf.fit(data, label)
    best_esitmator=clf.best_estimator_
```

除此之外，也能可就一个参数进行遍历，画出精度折线图。

提交后，RF+onehotencoding误差为**0.14146**；RF+labelencoding误差为**0.14292**（看起来反而独热编码好点）

### Ridge

再试一下其他模型，比如Rigde岭回归
同样进行调参，提交后，误差为**0.13290**（labelencoding）

### Stacking

Stacking的概念具体[如此](https://blog.csdn.net/aliceyangxi1987/article/details/74857294)，建议结合下面的图片/代码和链接内的文字，有助于理解（除此之外，还有Bagging、Boosting和Blending等[ensemble方法](https://zhuanlan.zhihu.com/p/25836678)）

这里使用两层Stacking举例说明一下：
第一层两个学习器分别为RandomForest和Rigde，
第二层的学习器为LinearRegression。

{% qnimg Kaggle-3-House-Prices/RF_L1.png %}
{% qnimg Kaggle-3-House-Prices/Ridge_L1.png %}
{% qnimg Kaggle-3-House-Prices/L2.png %}

```py
def get_oof(clf,traindata,trainlabel,testdata,kf):
    oof_train=np.zeros((traindata.shape[0],))
    oof_test=np.zeros((testdata.shape[0],))
    oof_test_tmp=np.zeros((5,testdata.shape[0]))

    for i,(train_index,test_index) in enumerate(kf.split(traindata)):
        x_train=traindata.iloc[train_index,:]
        y_train=trainlabel.iloc[train_index]
        x_test=traindata.iloc[test_index,:]

        clf.fit(x_train,y_train)
        oof_train[test_index]=clf.predict(x_test)
        oof_test_tmp[i,:]=clf.predict(testdata)

    oof_test=oof_test_tmp.mean(axis=0)
    return oof_train.reshape(-1,1),oof_test.reshape(-1,1)
```

提交，误差为**0.13205**，还是有所提升的。

### Bagging

由于该问题是回归问题，所以使用Averageing即加权。
```py
def combine(traindata,testdata,label):
    clf1 = RandomForestRegressor(n_estimators=300, max_features=.3,oob_score=True)
    clf1.fit(traindata,label)
    prediction1=clf1.predict(testdata)

    clf2 = Ridge(30)
    clf2.fit(traindata,label)
    prediction2=clf2.predict(testdata)

    test=np.vstack((prediction1,prediction2))
    prediction=test.mean(axis=0)
    save(prediction,'submission_power_RF_Ridge.csv')
```

提交，误差为0.12548（onehot encoding）和0.13009（label encoding）

看来，在这里加权比stacking效果还好（可能因为stacking的第二层有点简陋）

----------

# 修改

这里顺便介绍一下[jupyter notebook](https://www.jqr.com/article/000251)，不错的工具

前面提到，HousePrice的主要问题在于特征工程，Kaggle上Pedro Marcelino[的一篇文章](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python/notebook)提供一个不错的处理思路，还是值得学习的。

这里选择性的记录与分享

## 主要流程
 1. 理解问题
 2. 因变量研究
 3. 变量联系
 4. 数据清洗：缺失值、离群点、类别属性
 5. 假设（数据转换）

## 预备工作（理解问题）

建议新建一个电子表格，记录各个属性的名称、类型、变量分割（位置/空间/建筑）、期望（即对房价的影响）和备注。

列出上述表格之后，我们就能先找准几个重要的属性进行研究（作者选中的是OverallQual、
YearBuilt、TotalBsmtSF和GrLivArea）

> 这里未必需要直接从复杂的表格中找出来，可以结合着下面的热力图挑选

## 分析目标（因变量房价研究）

通过``describe()``查看房价分布情况，再通过``sns.distplot(df_train['SalePrice']);``画出直方图，可以看出并不是一个正态。
{% qnimg Kaggle-3-House-Prices/1.png %}

## 变量联系

### 重要属性与房价的联系

```py
#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
```

{% qnimg Kaggle-3-House-Prices/2.png %}
正相关关系，
同上，也可画出TotalBsmtSF的
{% qnimg Kaggle-3-House-Prices/3.png %}
也是正相关，似乎还能看出指数关系

以上是数值属性的，下面可以通过box图，看房价与OverallQual这个类比属性的关系
```py
#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
```
{% qnimg Kaggle-3-House-Prices/4.png %}
{% qnimg Kaggle-3-House-Prices/5.png %}
貌似和OverallQual以及yearbuilt正相关

### 总体分析

#### 相关矩阵

```py
#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
```

{% qnimg Kaggle-3-House-Prices/6.png %}

通过这样的一个图，我们能够直观的看出属性的关系

之后，再从中挑选出几个与房价相关性最大的属性，重新画图
```py
#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
```

{% qnimg Kaggle-3-House-Prices/7.png %}

通过这个，我们可以进行更为细致的研究，如GarageArea和GarageCars是有因果关系的，TotalBsmtSF和1stFlrSF似乎也有相关性、TotRmsAbvGrd和GrLivArea也有相当关系，作者的做法是凡是遇到相关性大的属性，就只保留一个（我们可以尝试保留）

#### 相关散点图
```py
#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();
```
{% qnimg Kaggle-3-House-Prices/8.png %}

通过上面这个庞大的图，进行你的猜测，验证想法。

## 数据清洗

### 缺失值处理

> 作者删去了一大部分有缺失值的属性，接近20种属性，就我的看法其中不少是可以利用起来的。此外，作者没有考虑到train.csv中的NA不全意味着缺失值，也有可能是没有对应的物品，所以谈不了什么物品多大，“有无”也是可以利用的特征（这反映了之前预备工作的重要性，否则很容易将NA这一点遗漏）

```py
#missing data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
```
输出各属性的"缺失值比例"，结合相关性，选择性删去属性

### 异常值处理

将训练样本的房价标准化，容易发现个别离群值

## 建模准备工作（数据转换）

使数据符合统计假设

 - 正态性
 - 同方差性
 - 线性

当数据具备以上特征时，对于部分学习器有利，但并不绝对。

### 正态性

画出房价的直方图，以及P-P图
```py
#histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)
```

{% qnimg Kaggle-3-House-Prices/9.png %}
可以看出，并不符合正态分布。针对这种情况，常用log变换（这一点其实可以从比赛的evaluation的误差计算中看出）

变换后
{% qnimg Kaggle-3-House-Prices/10.png %}

以此类推，对最开始选定的几个重要属性都进行操作

### 同方差性

{% qnimg Kaggle-3-House-Prices/11.png %}

诸如上图SalePrice和GrLivArea的分布散点图，我们可以粗略认为具备同方差性，返回去看上面的图（未进行log变换前），可以发现是锥子形的。

### 类别属性

原文作者在类别属性上没做太大功夫，直接使用独热编码。

> 关于上面的修改，在相关性分析上，只着眼于线性关系，包括热力图，事实上，非线性关系也是需要关注的；此外，原文对于属性的删除显得过于随性

基于上述的数据处理后（注意学习器的参数得重新调整），提交后误差为0.12327


----------

# 总结

|    Model     | RMSE   |  categorical  |
| :--------:   | :-----:  | :----:  |
| RF        |   0.14292   |  label   |
| RF     | 0.14146 |   onehot     |
| Ridge     |    0.13290    |  label  |
| stacking  |    0.13205    |  label  |
| bagging     |    0.13009 |  label  |
| bagging  |    0.12548    |  onehot  |
| bagging_new  |    0.12327    |  onehot  |

当遇到棘手问题的时候，保持基模思想；
再者，主要思路就是
 - 问题描述/数据概况
 - 熟悉属性（基于背景列属性表格）
 - 分析目标
 - 变量联系（散点图/热力图辅助）
 - 数据清洗（离群/缺失）
 - 建模准备（数据转换）
 - 融合调参（ensemble和GridSearchCV）
