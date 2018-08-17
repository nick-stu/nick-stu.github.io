---
title:      "Kaggle(2): Titanic"
date:       2018-08-02
tags:
- 数据挖掘
- Python
- 机器学习
- Kaggle
---
分享两篇不错的文章给第一次入门的小伙伴  
[Kaggle入门介绍](https://zhuanlan.zhihu.com/p/25686876)  
[Kaggle入门指南](https://zhuanlan.zhihu.com/p/25742261)  

----------
# 初版：
总体流程：

 1. 数据清洗
 2. 数据预处理
 3. 训练模型及分类

## 数据清洗
文件通过read_csv读入成为dataFrame对象（[dataFrame的切片](https://blog.csdn.net/qiao8756/article/details/80598849)）  
首先，我们要删除冗余的属性，分析数据的属性
{% qnimg 11/1.png %}
可以明显观察到Name,PassengerId,Ticket,Cabin属性对于我们分类没有用处，所以删去
```py
# loadData
data=pd.read_csv('train.csv')

# 查看缺省值情况
# print(data.info())

# 删除冗余属性
del data['Name']
del data['Ticket']
del data['Cabin']
del data['PassengerId']
```

## 数据预处理
### 缺省值处理
可以通过data.info()查看缺省的情况，以下是train.csv的情况，可以发现Age和Embarked属性都存在小部分缺省值，而Cabin缺省值比例接近80%，即便前面没有删去该属性，在这一步的时候也应该删去。
``` 
<class 'pandas.core.frame.DataFrame'>  
RangeIndex: 891 entries, 0 to 890  
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
```
[机器学习如何处理缺省值](https://www.zhihu.com/question/26639110)
```py
# 填充缺省值
mean=int(trainData.mean()['Age'])
meanFare=trainData.mean()['Fare']
mode=EmbarkedMode(trainData['Embarked'])

trainData['Fare'] = trainData['Fare'].fillna(meanFare)
trainData['Age'] = trainData['Age'].fillna(mean)
trainData['Embarked'] = trainData['Embarked'].fillna(mode)

testData['Fare'] = testData['Fare'].fillna(meanFare)
testData['Age'] = testData['Age'].fillna(mean)
testData['Embarked'] = testData['Embarked'].fillna(mode)
```
### One Hot-encoding处理标称属性
诸如Sex、Embarked属性，因为是字符串，且并无大小联系。所以采用独热编码，而非数值映射。
>参考链接：  
[python One-hot coding 独热编码](https://blog.csdn.net/HeatDeath/article/details/78051218)  
[One-Hot Encoding](https://blog.csdn.net/pipisorry/article/details/61193868)

python使用``get_dummies``即可
## 训练模型
选择的算法是逻辑回归
>参考链接：  
[线性回归简单例子](https://zhuanlan.zhihu.com/p/37364327)  
[逻辑回归简单例子](https://zhuanlan.zhihu.com/p/37779484)  
[线性回归算法](https://www.cnblogs.com/babers/p/6727933.html)  
[逻辑回归算法](https://www.cnblogs.com/babers/p/6817317.html)  


```py
from sklearn.linear_model import LogisticRegression

# build model
model=LogisticRegression()
# train model
model.fit(trainData,trainLabel)
accuracy=model.score(testData,testLabel)
```

先选用了KFold（[交叉验证与网格搜索算法](https://www.davex.pw/2017/09/17/Cross-Validation/)）对train.csv的数据进行划分并测试，求平均得出accuracy.  
[sklearn库的Cross Validation](https://blog.csdn.net/Dream_angel_Z/article/details/47048285)  
最后观察accuracy情况不错，能够达到79%，将train.csv作为训练样本，test.csv作为测试样本开始正式的分类，最后提交，accuracy为75%.

----------
# 修改

>**注：**修改是基于这两篇博客。[前者](https://blog.csdn.net/han_xiaoyang/article/details/49797143)基于逻辑回归，[后者](https://zhuanlan.zhihu.com/p/28586467)虽然基于随机森林，但思路同样值得学习  
下面只阐述从博客中学到的知识点，整体流程请点击链接查看原文


## 数据分析
### 观察数据分布情况
数值型
``data_train.describe()``

{% qnimg 11/3.png %}
离散型
``data_train.describe(include=[np.object])``

{% qnimg 11/4.png %}


能够了解到最后获救人数的情况、乘客年龄分布、不同等级船舱人数情况、票价情况、性别分布等等一些信息
### 探究不同属性与Survived的关联
可以得出：Pclass越小，即上等舱的人容易获救；女生获救概率大；Embarked中C的获救概率较大；婴儿获救率大；票价越高存活率越大等等

此外，虽然子女或兄弟姐妹个数虽然从概率分布上看与Survived没有明显关联，但是后期可以合并这两特征构造一个诸如familySize的特征（这个猜测是合理的，也确实将看起来没有的属性利用起来了，这也算得上一种DataMining的嗅觉了，不过还是得通过统计获救概率来看看是否有明显区分，基于特征计算获救概率往往是一种不错的判断特征有用与否的简便方法）

以上，除了可以通过数字表示，还能通过图的形式表现出来(matplotlib,seaborn)，

## **特征处理**

### 标称/离散属性
大多做独热编码
### 缺省值处理
#### Cabin
不同于我之前的做法，这两篇博客都没有因为缺失值过多而直接丢弃Cabin属性，而是去观察有Cabin值和无Cabin值的获救概率情况，发现有Cabin值的获救概率大，那么之后可以将Cabin属性转化为“有/无”（经过转化，即便是缺失值过多的属性也被利用起来了）
#### Name
Name属性在这里也被利用起来了，通过提取名字中诸如“Lady、Countess、Master、Capt”等的有实际意义的字符串，统计Survived概率，发现Miss、Mrs、Master存活概率都较高，这也是一个不错的特征
#### Age
Age缺失较多，可以通过随机森林预测、计算对应船舱年龄中位数等方法填充。
这里也可以考虑将年龄分段处理，

> 通过统计特定属性上的获救概率分布情况，是一种简洁判别属性是否有效的方法

诸如上面，逐个分析属性如何处理

### 特征缩放（feature scaling）
注意属性之间的数值范围，如果有些属性值域为[-1,1]，而有些为[0,100]，有时候对于模型的训练是相当不利的，所以要做scaling（可仅针对一个属性）或者标准化，归一化等



## 训练模型
可以各种类型的都跑一遍，再调参

## 模型分析及改进
可以通过查看训练好的模型的参数分析  
``pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)})``  
或者基于特征处理作出假设，类似于将年龄分段离散化为类别属性、利用Cabin中的数值和字母

### Learning curves
过拟合/欠拟合 分析，判断我们现有模型处于哪种状态，便于我们作出调整，比如欠拟合的话，通常需要更多的feature  

{% qnimg 11/5.png %}

## 模型融合(Model ensemble)
意思是通过多个模型去分类，可以有两种方式，一种是同种算法不同训练样本，一种是不同算法相同训练样本（后者更常见）  

----------

{% qnimg 11/6.png %}

## **总结**

数据预处理时，要**看到问题的背景**，才能做出合理的处理。要更多地看到属性的分布以及与结果的联系，同时对于属性类型的转化往往也会带来出乎意料的效果

做了这道题才开始体会到Data Mining不同于iot的地方，iot的机器学习大都局限于信号，属性值全为数值型，在数据预处理方面较DataMining容易，模型选择上局限性也较大。

**完善后的流程：**

 1. 数据概况
 2. 数据分析（属性分布、属性与结果关联）
 3. 特征处理（关注缺省值）
 4. 训练模型
 5. 模型分析与改进