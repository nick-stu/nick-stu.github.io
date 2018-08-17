---
title:      "Kaggle(1): Digit Recognizer"
date:       2018-07-30 14:39:53
author:     "Nick"
tags:
- Kaggle
- Python
- 数据挖掘
- 机器学习
---
对于[Kaggle](https://www.kaggle.com)还不是特别熟悉，所以先[熟悉一下
](https://blog.csdn.net/u012162613/article/details/41929171)   

总体思路如博客所说，**装载训练样本，标签，测试样本**，这里面涉及[csv模块的应用](https://blog.csdn.net/u012162613/article/details/41915859)，伴随的是**数据预处理**的部分，例如把字符串转化为整数、二值化，在这其中伴随着array,mat等数据类型的处理，这方面还需要熟悉，最后就是进入**分类函数**，这里采用的是knn，具体实现就不多谈。最后存储结果。
### 调用sklearn
除了像上面一样自己实现分类算法外，还能[直接调用sklearn](https://blog.csdn.net/u012162613/article/details/41978235)，诸如KNN，SVM之类的都可直接使用
```py
from sklearn.neighbors import KNeighborsClassifier #调用scikit的knn算法包
knnClf=KNeighborsClassifier(n_neighbors=3) #可调整参数
knnClf.fit(trainData,ravel(trainLabel)) #训练模型，注意trainData每行为一样本，trainLabel为行向量
testLabel=knnClf.predict(testData) #分类器预测
```
样本数过大，中途出现了MemoryError  
总的来说，过程并不算陌生，基本上都是之前用matlab实现过的项目流程，需要在使用python相关工具上更为熟练。  
按格式提交后，就能看见结果和排名了
{% qnimg 9/1.png %}