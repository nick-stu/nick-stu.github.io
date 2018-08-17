---
title: SVM推导：优化目标函数/核函数
date: 2018-08-06 16:29:01
tags:
- SVM
- 机器学习
---
> **前言**：
分割线内的是相关知识说明，为了防止跳跃性过大，全文尽量以推导为驱动，中间夹杂相关公理和知识的说明，**注意**分割线内的说明所用的变量名可以会与外面不同。


> 参考资料：
[推导支持向量机 (SVM)](https://github.com/HaoMood/File/blob/master/%E4%BB%8E%E9%9B%B6%E6%8E%A8%E5%AF%BC%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA%28SVM%29.pdf)
[Support Vector Machine(一)](https://zhuanlan.zhihu.com/p/24638007)
[SVM-Support Vector Machine(二)](https://zhuanlan.zhihu.com/p/29865057)
[《机器学习》——周志华](https://book.douban.com/subject/26708119/)

{% qnimg SVM推导/1.jpg %}

## 构造原始优化函数

### 正确分类
我们将超平面定义为$w^Tx+b=0$ 
样本空间中点x到超平面(w,b)的距离可写为$$r=\frac{|w^Tx+b|}{||w||}$$
证明过程略，和高中学的点到平面的距离一样。

$y_i$为$x_i$的类，构造如下函数：
$$\begin{cases}
w^Tx_i+b \geq +1,\   y_i=+1\\\\  
w^Tx_i+b \leq -1,\   y_i=-1
\end{cases}$$
可整理为$y_i(w^T+b)\geq1$
距离超平面最近的两类点使等号成立。

### 最大间隔

设$x_1$和$x_2$分别为两类离边界最近的点，则间隔$\gamma=2/||w||$（利用上面的点到平面距离公式即可）
最大间隔也只与边界上的点有关，所以将这些点称为支持向量，这也是支持向量机的来源

### 初始优化函数

目标是找到最大间隔的划分超平面，即$$\max_{w,b}\frac{2}{||w||}\\\\
s.t.\  \ y_i(w^T+b)\geq1$$
可重写为$$\min_{w,b}\frac{||w||^2}{2}\\\\
s.t.\  \ y_i(w^T+b)\geq1$$


事实上，在此处我们已经能够用现成的优化计算包求解，不过还有更高效的方法。

## 拉格朗日乘子法（Lagrance multipliers）

----------

> 拉格朗日乘子法的基本思想是使**有约束优化问题**变成我们习惯的**无约束优化问题**

这里先举例一个等式约束的优化问题：假定x为d维向量，欲寻找x的某个取值$x^*$，使目标函数f(x)最小且同时满足g(x)=0的约束要求。

那么有如下结论:

 - 约束曲面上任一点x的梯度$\nabla g(x)$正交于约束曲面
 - $\nabla f(x^*)$正交于约束曲面


易知，$\nabla f(x^\*)$和$\nabla g(x^\*)$方向相同或相反
即存在$\lambda\not=0$使得$$\nabla f(x^\*)+\lambda \nabla g(x^\*)=0\tag{1}$$
那么$\lambda$称为拉格朗日乘子，这里定义拉格朗日函数$$L(x,\lambda)=f(x)+\lambda g(x)\tag{2}$$  
对x求偏导置零即可得式（1），对$\lambda$求偏导置零即可得约束条件，因而我们成功将原约束优化问题转化为对拉格朗日函数的无约束优化问题。

{% qnimg SVM推导/2.jpg %}

刚才考虑的是等式约束，现在考虑不等式约束。
分情况分析：
首先当最优点在g(x)<0的区域中，那么这里的约束条件就没有意义了，我们可以直接通过条件$\nabla f(x)=0 $来获得最优点，$\lambda = 0$  
当最优点在g(x)=0时，情况和上面的等式约束类似，不过两个梯度的方向一定是相反的（因为意味着上图内部的g(x)值更大，梯度方向是指向变大方向的），则$\lambda < 0$

则在约束$g(x) \leq 0$下最小化f(x)，可转化为在如下约束下最小化式(2):
$$\begin{cases}
g(x) \leq 0;\\\\
\lambda \geq 0;\\\\
\lambda g(x)=0
\end{cases}$$
> 上式称为KKT条件(Karush-Kuhn-Tucker)


----------

回到我们的优化函数，即$$\min_{w,b}\frac{||w||^2}{2}\\\\
s.t.\  \ y_i(w^T+b)\geq1$$
可转化为{% qnimg SVM推导/3.jpg %}


{% qnimg SVM推导/4.png %}
[图片来源](https://github.com/HaoMood/File/blob/master/%E4%BB%8E%E9%9B%B6%E6%8E%A8%E5%AF%BC%E6%94%AF%E6%8C%81%E5%90%91%E9%87%8F%E6%9C%BA%28SVM%29.pdf)
根据上图推导，我们的优化函数又可转化为{% qnimg SVM推导/5.png %}

## 对偶问题

----------

{% qnimg SVM推导/6.png %}
诸如上式的形式，我们把下面的式子称为上面的式子的对偶问题。  
我们需要证明他们有同样的最优解，若得证，那么我们可以将原始问题转化为其对偶问题。

{% qnimg SVM推导/7.png %}
[图片来源](https://zhuanlan.zhihu.com/p/24638007)
以上称为弱对偶性；若$d^\*=q^\*$则称为强对偶性，此时能由对偶问题获得主问题的最优下界。

{% qnimg SVM推导/8.png %}
[图片来源](https://zhuanlan.zhihu.com/p/24638007)
强对偶性的成立条件，成立之后只需要对拉格朗日函数求导并置零即可得到原变量和对偶变量的关系

----------

根据上面的对偶问题知识，我们的优化函数又可从{% qnimg SVM推导/5.png %}转化为{% qnimg SVM推导/9.png %}
 
那么这里先求解{% qnimg SVM推导/10.png %}

{% qnimg SVM推导/11.jpg %}
{% qnimg SVM推导/12.jpg %}


至此，优化函数中已不含w,b，解出拉格朗日乘子$\alpha$，求出w,b即可得到模型(稍后再解释如何解出拉格朗日乘子)
$$f(x)=w^Tx+b\\
=\sum_{i=1}^m \alpha_iy_ix_i^Tx+b$$

这里会发现我们的拉格朗日乘子数目会和训练样本一样多，事实上,在转化为拉格朗日函数式(6.8)时，有不等式约束，KKT条件为
$$\begin{cases}
\alpha_i \geq 0;\\\\
1-y_if(x_i)\leq 0;\\\\
a_i(1-y_if(x_i))=0
\end{cases}$$

我们注意上面的第三条式子，可以发现对任何训练样本总有$\alpha_i=0$或$y_if(x_i)=1$，即训练过程(计算f(x)过程)，只有边界点对计算过程有影响，即下图虚线上的点。
{% qnimg SVM推导/13.jpg %}
[图片来源](https://zhuanlan.zhihu.com/p/24638007)

> **我们到这里可以发现从公式、概念和名称上，我们在"支持向量"上达到了统一**

## 求解拉格朗日乘子（二次规划问题/SMO）

使用二次规划问题可以求解，但这会使问题的规模正比于训练样本数；而SMO(Sequential Minimal Optimization)的一种更为高效的方法.

> 关于这部分有空再拓展介绍一下

## 核函数

现在我们再拓展开来，如果遇到不能用一个超平面划分两类样本的超平面呢？
{% qnimg SVM推导/14.jpg %}

可以从上图看到解决方法**：将样本映射到一个更高维的特征空间，使得样本在新特征空间内线性可分**

则新模型为$f(x)=w^T\phi(x)+b$，优化函数为$$\max_\alpha \sum_{i=1}^m\alpha_i-\frac12\sum_{i=1}^m\sum_{j=1}^m\alpha_i\alpha_jy_iy_j\phi(x_i)^T\phi(x_j)\\\\ s.t. \sum_{i=1}^m\alpha_iy_i=0\\\\
\alpha_i \geq 0, i =1,2,3,...m$$
其中$\phi(x_i)^T\phi(x_j)$的计算，维数非常高，计算困难。  
所以采用$$\kappa(x_i,x_j)=\langle\phi(x_i),\phi(x_j)\rangle=\phi(x_i)^T\phi(x_j)$$
如此我们不必直接去计算高维空间中的内积，于是有$$\max_\alpha \sum_{i=1}^m\alpha_i-\frac12\sum_{i=1}^m\sum_{j=1}^m\alpha_i\alpha_jy_iy_j\kappa(x_i,x_j)\\\\ s.t. \sum_{i=1}^m\alpha_iy_i=0\\\\
\alpha_i \geq 0, i =1,2,3,...m$$
求解得$$f(x)=w^T\phi(x)+b\\
=\sum_{i=1}^m \alpha_iy_i\phi(x_i)^T\phi(x)+b\\
=\sum_{i=1}^m \alpha_iy_i\kappa(x,x_i)+b\tag{4}$$
$\kappa(.,.)$即为核函数(kernel function),式(4)称为支持向量展式  

那么在不知道$\phi(x)$的情况下，我们如何知道核函数？
{% qnimg SVM推导/16.jpg %}
{% qnimg SVM推导/15.png %}

即对于一个函数，若满足上述条件就为核函数，就能找到一个与其对应的映射$\phi$。换言之，一个核函数隐式定义了再生核希尔伯特空间(Reproducing Kernel Hilbert Space,RKHS)

可以说核函数的选取很重要，因为定义了如何映射到高维空间，决定了是否能够线性可分（对于不同情况如何选取有相关的经验）
常用的核函数有线性核、多项式核、高斯核(RBF核)、拉普拉斯核和Sigmoid核等。
也可通过函数组合得到新的核函数，这方面的组合可以查阅P129《机器学习》——周志华
