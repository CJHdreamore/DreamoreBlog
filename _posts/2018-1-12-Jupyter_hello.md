---
layout: post
author: 陈小耗
title: jupyter的首次尝试
categories: learning
tags: [DataScience][Jupyter]
---


```python
#encoding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

```


```python

train_df = pd.read_csv(r'E:\for_future\jupyter_code\HousePrice\train.csv')
test_df = pd.read_csv(r'E:\for_future\jupyter_code\HousePrice\test.csv')

```


```python
# How many variables in training data? What are they??
variables = train_df.columns
```


```python
variable_df = pd.DataFrame(train_df.columns[1:],
                           columns=['name'])
with open(r'E:\for_future\jupyter_code\HousePrice'
          r'\variable.csv','w',newline='') as f:
    variable_df.to_csv(path_or_buf=f,sep=',',
                       columns=['name','type',
                                'attribute','exp',
                                'actual','others']
                       )
 
    
    
```


```python
# 训练数据中的房价有什么表现?
train_df['SalePrice'].describe()
```




    count      1460.000000
    mean     180921.195890
    std       79442.502883
    min       34900.000000
    25%      129975.000000
    50%      163000.000000
    75%      214000.000000
    max      755000.000000
    Name: SalePrice, dtype: float64




```python
# 房价的大体分布如何？
sns.distplot(train_df['SalePrice']);

```


![png]({{site.baseurl}}/assets/img/HousePredict/output_5_0.png)



```python
# 看起来类似于一个正态分布，它的峰度和偏度是多少？
print('偏度： %f'% train_df['SalePrice'].skew())
print('峰度： %f'% train_df['SalePrice'].kurt())
```

    偏度： 1.882876
    峰度： 6.536282
    


```python
# 从excel表中选出我认为可能对房价影响较大的变量，包括数值型
# 变量和类别型变量
n_var = 'GrLivArea'
data = pd.concat([train_df['SalePrice'],train_df[n_var],
                  ],axis=1)
data.plot.scatter(x=n_var,y='SalePrice',ylim=(0,80e4));

```


![png]({{site.baseurl}}/assets/img/HousePredict/output_7_0.png)



```python
n_var = 'TotRmsAbvGrd' #楼上房间的个数
data = pd.concat([train_df['SalePrice'],train_df[n_var],
                  ],axis=1)
data.plot.scatter(x=n_var,y='SalePrice',ylim=(0,80e4));
```


![png]({{site.baseurl}}/assets/img/HousePredict/output_8_0.png)



```python
n_var = 'LotArea' #小区的面积
data = pd.concat([train_df['SalePrice'],train_df[n_var]],
                 axis=1)
data.plot.scatter(x=n_var,y='SalePrice',ylim=(0,80e4))

```




    <matplotlib.axes._subplots.AxesSubplot at 0x106345c0>




![png]({{site.baseurl}}/assets/img/HousePredict/output_9_1.png)



```python
n_var = 'TotalBsmtSF' #地窖的总面积
data = pd.concat([train_df['SalePrice'],train_df[n_var]],
                 axis=1)
data.plot.scatter(x=n_var,y='SalePrice',ylim=(0,80e4))

```




    <matplotlib.axes._subplots.AxesSubplot at 0xdc31c50>




![png]({{site.baseurl}}/assets/img/HousePredict/output_10_1.png)



```python
n_var = 'GarageArea' #车库的总面积
data = pd.concat([train_df['SalePrice'],train_df[n_var]],
                 axis=1)
data.plot.scatter(x=n_var,y='SalePrice',ylim=(0,80e4))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x108554a8>




![png]({{site.baseurl}}/assets/img/HousePredict/output_11_1.png)



```python
# 针对excel表格中类别型的变量，以boxplot来探究其离散分布
c_var = 'OverallQual'
data = pd.concat([train_df['SalePrice'],train_df[c_var]],
                 axis=1)
f,ax = plt.subplots(figsize=(8,6))
fig = sns.boxplot(x=c_var,y='SalePrice',data=data)
fig.axis(ymin=0,ymax=80e4);
```


![png]({{site.baseurl}}/assets/img/HousePredict/output_12_0.png)



```python
c_var = 'OverallCond'
data = pd.concat([train_df['SalePrice'],train_df[c_var]],
                 axis=1)
f,ax = plt.subplots(figsize=(8,6))
fig = sns.boxplot(x=c_var,y='SalePrice',data=data)
fig.axis(ymin=0,ymax=80e4);
```


![png]({{site.baseurl}}/assets/img/HousePredict/output_13_0.png)



```python
c_var = 'Utilities'
data = pd.concat([train_df['SalePrice'],train_df[c_var]],
                 axis=1)
f,ax = plt.subplots(figsize=(8,6))
fig = sns.boxplot(x=c_var,y='SalePrice',data=data)
fig.axis(ymin=0,ymax=80e4);
#哈哈，说明所有房子都是有水电气的。
```


![png]({{site.baseurl}}/assets/img/HousePredict/output_14_0.png)


#以上都是主观认识，首先通过先验来选择一些变量，再进一步可视化
#分析单一变量和房屋价格的关系：比如成正比的线性关系等



```python
# 利用heatmap来可视化各变量的correlation matrix
cor_matrix = train_df.corr()
f, ax = plt.subplots(figsize = (12,9))
sns.heatmap(cor_matrix,vmax=0.8,square=True);

```


![png]({{site.baseurl}}/assets/img/HousePredict/output_16_0.png)


- 上方的heatmap图给我们以启示：
1.TotalBsmtS和1stFlrSF,GarageYrBlt和YearBuilt,TotRsAbvGrd和GrLiveArea,GarageCars和GarageArea，GrLiveArea和TotRmsAbvGrd这些变量之间都有强相关性;
2.可以观察出哪些变量和SalePrice之间有强相关性：OverallQual,TotalBsmtSF,GrLiveArea,GargeArea等，其中相关性最强的是：OverallQual,GrLiveArea。
这和我们的主观认识也是非常相符的。



```python
# 更细致地显示出相关系数
k = 10
cols = cor_matrix.nlargest(k,'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot = True, square= True, fmt = '.2f',
                annot_kws = {'size':10},yticklabels = cols.values,
                xticklabels = cols.values)
plt.show()


```


![png]({{site.baseurl}}/assets/img/HousePredict/output_18_0.png)


从上面这幅heatmap中可以明显看到和SalePrice相关性最强的变量（除自身）:
OverallQaul,GrLivArea/TotRmsAbvGrd,GarageCars/GarageArea,TotalBsmtSF/1stFlrSF,FullBath,YearBuilt
对这些变量进行一个简单的说明：
- OverallQual:嗯，这个很自然嘛，房屋的总体评价影响房价；
- GrLivArea/TotRmsAbvGrd:这两个数值型的变量代表了房屋的面积，因它们俩具有强相关性，所以只用选取一个作为模型中的变量。
- GarageCars/GarageArea:描述车库的大小，任选其一
- TotalBsmtSF/1stFlrSF:地窖的总面积和一楼的总面积也是强相关量，任选其一
- FullBath:比较有趣，地面上的卫生间的数目竟然和房价有较强的相关性，难道是因为卫生间多暗示了房屋大？
- YearBuilt:嗯...房屋修建的时间，嗯，房屋的新旧对房价还是有影响的。



```python
#Amazing!Scatterplot!!
sns.set()
cols = ['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']
sns.pairplot(train_df[cols],size=2.5)
plt.show();
```


![png]({{site.baseurl}}/assets/img/HousePredict/output_20_0.png)


我要哭泣了！seaborn也太好用了吧！！pairplot简直神器啊！！！！！
以“SalePrice”为纵坐标，其他变量为横坐标观察这些散点图，可以抓住这些单一变量影响房价的趋势：
- OverallQual，GarageCars,FullBath都是离散型的变量，所以它们展现出来的趋势可能不明显
- GrLivArea和房价的关系可以说明显了：正比例，可能是指数上升关系。
- 地窖的总面积和房价亦有类指数上升的关系；
- 修建的时间和房价的关系初步看起来也是：房屋越新，房价越高。但这还需要进一步做**时间序列分析**
### 一些有趣的现象

以地窖总面积为横轴，地面生活区域面积为纵轴的图中可见，散点构成了一条边界线：大多数情况地面生活区域面积总是大于地窖总面积的，毕竟大多数购房者都是用于居住。



```python
# 处理丢失数据
total = train_df.isnull().sum().sort_values(ascending=False)
percent = (train_df.isnull().sum() / train_df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1,keys=['Total','Percent'])
missing_data.head(20)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PoolQC</th>
      <td>1453</td>
      <td>0.995205</td>
    </tr>
    <tr>
      <th>MiscFeature</th>
      <td>1406</td>
      <td>0.963014</td>
    </tr>
    <tr>
      <th>Alley</th>
      <td>1369</td>
      <td>0.937671</td>
    </tr>
    <tr>
      <th>Fence</th>
      <td>1179</td>
      <td>0.807534</td>
    </tr>
    <tr>
      <th>FireplaceQu</th>
      <td>690</td>
      <td>0.472603</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>259</td>
      <td>0.177397</td>
    </tr>
    <tr>
      <th>GarageCond</th>
      <td>81</td>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>GarageType</th>
      <td>81</td>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>81</td>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>GarageFinish</th>
      <td>81</td>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>GarageQual</th>
      <td>81</td>
      <td>0.055479</td>
    </tr>
    <tr>
      <th>BsmtExposure</th>
      <td>38</td>
      <td>0.026027</td>
    </tr>
    <tr>
      <th>BsmtFinType2</th>
      <td>38</td>
      <td>0.026027</td>
    </tr>
    <tr>
      <th>BsmtFinType1</th>
      <td>37</td>
      <td>0.025342</td>
    </tr>
    <tr>
      <th>BsmtCond</th>
      <td>37</td>
      <td>0.025342</td>
    </tr>
    <tr>
      <th>BsmtQual</th>
      <td>37</td>
      <td>0.025342</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>8</td>
      <td>0.005479</td>
    </tr>
    <tr>
      <th>MasVnrType</th>
      <td>8</td>
      <td>0.005479</td>
    </tr>
    <tr>
      <th>Electrical</th>
      <td>1</td>
      <td>0.000685</td>
    </tr>
    <tr>
      <th>Utilities</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



### 怎么看待和处理这些丢失变量？
- 原则：

1.如果一个变量描述的数据超过百分之十五都丢失了，那么就删去这个变量。

比如丢失变量列表中的前六项，并且在之前的相关性分析中显示这些变量与房价的相关性本就很低，因此删去这些变量是合理的。

2.丢失变量中关于车库的项，虽然它们丢失的比例不足百分之十五，但考虑到相关性分析中，可以选取GarageCar作为强相关性变量，故这些变量也可删去；

3.同理关于“地窖”，使用TotalBsmtSF变量，即可去除丢失量变量中关于地窖的项目。

4.对于“MasVnrArea”和“MasVnrType”这两项，它们与房价的相关性很低，故也可滤掉。

5.如果一个变量丢失的数据特别少，那么就保留变量而在表格中删去丢失样本的那一行，如：“Electrical”

综上：
- 删去列表中丢失样本>1的变量

- 保留丢失样本数为1的变量，并在原始表格中删去该行样本；

- 保留所有无丢失样本的变量



```python
#Dealing with missing data
#删去丢失样本数>1的列
train_select = train_df.drop((missing_data[missing_data['Total'] > 1 ]).index,1)
#删去“Electrical”变量丢失掉的那一行样本
train_select = train_select.drop(train_df.loc[train_df['Electrical'].isnull()].index,0)
#确认列表中的所有变量已无丢失情况
train_select.isnull().sum().max()

```




    0




```python
# 检测异常值:
# 1.从预测变量自身的分布中设定阈值：
saleprice_scaled = StandardScaler().fit_transform(train_select['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('下界取值：')
print (low_range)
print ('上界取值：')
print (high_range)
```

    下界取值：
    [[-1.83820775]
     [-1.83303414]
     [-1.80044422]
     [-1.78282123]
     [-1.77400974]
     [-1.62295562]
     [-1.6166617 ]
     [-1.58519209]
     [-1.58519209]
     [-1.57269236]]
    上界取值：
    [[ 3.82758058]
     [ 4.0395221 ]
     [ 4.49473628]
     [ 4.70872962]
     [ 4.728631  ]
     [ 5.06034585]
     [ 5.42191907]
     [ 5.58987866]
     [ 7.10041987]
     [ 7.22629831]]
    


```python
# 2.从变量和预测值间的关系来检测异常值
n_var = 'GrLivArea'
data = pd.concat([train_df['SalePrice'],train_df[n_var],
                  ],axis=1)
data.plot.scatter(x=n_var,y='SalePrice',ylim=(0,80e4));


```


![png]({{site.baseurl}}/assets/img/HousePredict/output_26_0.png)


从上图中，可以明显看出图右下方有两个异常值，它们不符合GrLivArea影响房价的总趋势，可能是这两个样本中的其他变量的影响过大。因此为确保该变量和房价的正确关系，剔除这两个异常值。
图中右上方有两个样本，它们可能属于SalePrice中上界区域，但由于它们基本上维持了GrlivArea和房价之间的增量关系，故可以保留。


```python
# 删除异常值
train_select.sort_values(by='GrLivArea',ascending = False)[:2]
train_select = train_select.drop(train_select[train_select['Id'] == 1299].index)
train_select = train_select.drop(train_select[train_select['Id'] == 524].index)
```


```python
n_var = 'GrLivArea'
data = pd.concat([train_select['SalePrice'],train_select[n_var],
                  ],axis=1)
data.plot.scatter(x=n_var,y='SalePrice',ylim=(0,80e4));
```


![png]({{site.baseurl}}/assets/img/HousePredict/output_29_0.png)



```python
# 使用多变量分析的一些假设检验：
# 房价的分布是否具有normality?
sns.distplot(train_select['SalePrice'],fit=norm)
# 进一步用概率图来确定
fig = plt.figure()
res = stats.probplot(train_select['SalePrice'], plot = plt)
```


![png]({{site.baseurl}}/assets/img/HousePredict/output_30_0.png)



![png]({{site.baseurl}}/assets/img/HousePredict/output_30_1.png)


从直方图和正态概率图中得知：房价数据呈现右偏的正态分布，可以使用“对数变换”来迫使其满足正态分布


```python
train_select['SalePrice'] = np.log(train_select['SalePrice'])
sns.distplot(train_select['SalePrice'],fit=norm);
fig = plt.figure()
res = stats.probplot(train_select['SalePrice'],plot=plt)
```


![png]({{site.baseurl}}/assets/img/HousePredict/output_32_0.png)



![png]({{site.baseurl}}/assets/img/HousePredict/output_32_1.png)


看！现在的房价数据已经基本上服从了正态分布。
接下来将对几个数值型变量“GrLivArea”,“TotalBsmTSF”进行normality分析。



```python
sns.distplot(train_select['GrLivArea'],fit = norm);
fig = plt.figure()
res = stats.probplot(train_select['GrLivArea'],plot = plt)
```


![png]({{site.baseurl}}/assets/img/HousePredict/output_34_0.png)



![png]({{site.baseurl}}/assets/img/HousePredict/output_34_1.png)



```python
train_select['GrLivArea'] = np.log(train_select['GrLivArea'])
sns.distplot(train_select['GrLivArea'],fit = norm);
fig = plt.figure()
res = stats.probplot(train_select['GrLivArea'],plot = plt)
```


![png]({{site.baseurl}}/assets/img/HousePredict/output_35_0.png)



![png]({{site.baseurl}}/assets/img/HousePredict/output_35_1.png)



```python
sns.distplot(train_select['TotalBsmtSF'],fit = norm);
fig = plt.figure()
res = stats.probplot(train_select['TotalBsmtSF'],plot = plt)
```


![png]({{site.baseurl}}/assets/img/HousePredict/output_36_0.png)



![png]({{site.baseurl}}/assets/img/HousePredict/output_36_1.png)


在这个情况下，变量有取值为0的情况，因此无法取对数来使其满足正态分布。对应为0的数据取对数后为-inf，无法绘制分布图。
为了保留TotalBsmtSF=0的信息，新增一个二值变量来反应房屋有无地窖。
再仅对有地窖情况的数据进行标准化处理。




```python
#新增一个变量反映房屋是否有地窖
train_select['HasBsmt'] = pd.Series(len(train_select['TotalBsmtSF']), index = train_select.index)
train_select['HasBsmt'] = 0
train_select.loc[train_select['TotalBsmtSF'] > 0,'HasBsmt'] = 1
```


```python
#对有地窖的数据进行标准化
train_select.loc[train_select['HasBsmt'] == 1,'TotalBsmtSF'] = np.log(train_select['TotalBsmtSF'])
```


```python
sns.distplot(train_select[train_select['TotalBsmtSF'] > 0]['TotalBsmtSF'],fit = norm);
fig = plt.figure()
res = stats.probplot(train_select[train_select['TotalBsmtSF'] > 0 ]['TotalBsmtSF'],plot = plt)
```


![png]({{site.baseurl}}/assets/img/HousePredict/output_40_0.png)



![png]({{site.baseurl}}/assets/img/HousePredict/output_40_1.png)



```python
# 检验同方差性
# GrLivArea & SalePrice
plt.scatter(train_select['GrLivArea'],train_select['SalePrice']);
```


![png]({{site.baseurl}}/assets/img/HousePredict/output_41_0.png)


可以对比一下，下图是正态化之前的GrLivArea与SalePrice的散点图


```python
plt.scatter(train_df['GrLivArea'],train_df['SalePrice']);
```


![png]({{site.baseurl}}/assets/img/HousePredict/output_43_0.png)


可见，正态化使得这两个变量之间具有了线性关系！这正是我们后续使用多变量分析假设的前提。


```python
# analyze "TotalBsmtSF" 
plt.scatter(train_select[train_select['TotalBsmtSF'] > 0 ]['TotalBsmtSF'],
            train_select[train_select['TotalBsmtSF'] > 0 ]['SalePrice']);
```


![png]({{site.baseurl}}/assets/img/HousePredict/output_45_0.png)



```python
#将类别型变量转化为dummy variables
train_select = pd.get_dummies(train_select)
train_select
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>...</th>
      <th>SaleType_ConLw</th>
      <th>SaleType_New</th>
      <th>SaleType_Oth</th>
      <th>SaleType_WD</th>
      <th>SaleCondition_Abnorml</th>
      <th>SaleCondition_AdjLand</th>
      <th>SaleCondition_Alloca</th>
      <th>SaleCondition_Family</th>
      <th>SaleCondition_Normal</th>
      <th>SaleCondition_Partial</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>8450</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>706</td>
      <td>0</td>
      <td>150</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>9600</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>978</td>
      <td>0</td>
      <td>284</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>11250</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>486</td>
      <td>0</td>
      <td>434</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>9550</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>216</td>
      <td>0</td>
      <td>540</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>14260</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>655</td>
      <td>0</td>
      <td>490</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>50</td>
      <td>14115</td>
      <td>5</td>
      <td>5</td>
      <td>1993</td>
      <td>1995</td>
      <td>732</td>
      <td>0</td>
      <td>64</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>20</td>
      <td>10084</td>
      <td>8</td>
      <td>5</td>
      <td>2004</td>
      <td>2005</td>
      <td>1369</td>
      <td>0</td>
      <td>317</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>60</td>
      <td>10382</td>
      <td>7</td>
      <td>6</td>
      <td>1973</td>
      <td>1973</td>
      <td>859</td>
      <td>32</td>
      <td>216</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>50</td>
      <td>6120</td>
      <td>7</td>
      <td>5</td>
      <td>1931</td>
      <td>1950</td>
      <td>0</td>
      <td>0</td>
      <td>952</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>190</td>
      <td>7420</td>
      <td>5</td>
      <td>6</td>
      <td>1939</td>
      <td>1950</td>
      <td>851</td>
      <td>0</td>
      <td>140</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>11</td>
      <td>20</td>
      <td>11200</td>
      <td>5</td>
      <td>5</td>
      <td>1965</td>
      <td>1965</td>
      <td>906</td>
      <td>0</td>
      <td>134</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>12</td>
      <td>60</td>
      <td>11924</td>
      <td>9</td>
      <td>5</td>
      <td>2005</td>
      <td>2006</td>
      <td>998</td>
      <td>0</td>
      <td>177</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>13</td>
      <td>20</td>
      <td>12968</td>
      <td>5</td>
      <td>6</td>
      <td>1962</td>
      <td>1962</td>
      <td>737</td>
      <td>0</td>
      <td>175</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>14</td>
      <td>20</td>
      <td>10652</td>
      <td>7</td>
      <td>5</td>
      <td>2006</td>
      <td>2007</td>
      <td>0</td>
      <td>0</td>
      <td>1494</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>15</td>
      <td>20</td>
      <td>10920</td>
      <td>6</td>
      <td>5</td>
      <td>1960</td>
      <td>1960</td>
      <td>733</td>
      <td>0</td>
      <td>520</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>16</td>
      <td>45</td>
      <td>6120</td>
      <td>7</td>
      <td>8</td>
      <td>1929</td>
      <td>2001</td>
      <td>0</td>
      <td>0</td>
      <td>832</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>17</td>
      <td>20</td>
      <td>11241</td>
      <td>6</td>
      <td>7</td>
      <td>1970</td>
      <td>1970</td>
      <td>578</td>
      <td>0</td>
      <td>426</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>18</td>
      <td>90</td>
      <td>10791</td>
      <td>4</td>
      <td>5</td>
      <td>1967</td>
      <td>1967</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>19</td>
      <td>20</td>
      <td>13695</td>
      <td>5</td>
      <td>5</td>
      <td>2004</td>
      <td>2004</td>
      <td>646</td>
      <td>0</td>
      <td>468</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>20</td>
      <td>20</td>
      <td>7560</td>
      <td>5</td>
      <td>6</td>
      <td>1958</td>
      <td>1965</td>
      <td>504</td>
      <td>0</td>
      <td>525</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21</td>
      <td>60</td>
      <td>14215</td>
      <td>8</td>
      <td>5</td>
      <td>2005</td>
      <td>2006</td>
      <td>0</td>
      <td>0</td>
      <td>1158</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>22</td>
      <td>45</td>
      <td>7449</td>
      <td>7</td>
      <td>7</td>
      <td>1930</td>
      <td>1950</td>
      <td>0</td>
      <td>0</td>
      <td>637</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23</td>
      <td>20</td>
      <td>9742</td>
      <td>8</td>
      <td>5</td>
      <td>2002</td>
      <td>2002</td>
      <td>0</td>
      <td>0</td>
      <td>1777</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24</td>
      <td>120</td>
      <td>4224</td>
      <td>5</td>
      <td>7</td>
      <td>1976</td>
      <td>1976</td>
      <td>840</td>
      <td>0</td>
      <td>200</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>25</td>
      <td>20</td>
      <td>8246</td>
      <td>5</td>
      <td>8</td>
      <td>1968</td>
      <td>2001</td>
      <td>188</td>
      <td>668</td>
      <td>204</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>26</td>
      <td>20</td>
      <td>14230</td>
      <td>8</td>
      <td>5</td>
      <td>2007</td>
      <td>2007</td>
      <td>0</td>
      <td>0</td>
      <td>1566</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>27</td>
      <td>20</td>
      <td>7200</td>
      <td>5</td>
      <td>7</td>
      <td>1951</td>
      <td>2000</td>
      <td>234</td>
      <td>486</td>
      <td>180</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>20</td>
      <td>11478</td>
      <td>8</td>
      <td>5</td>
      <td>2007</td>
      <td>2008</td>
      <td>1218</td>
      <td>0</td>
      <td>486</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>29</td>
      <td>20</td>
      <td>16321</td>
      <td>5</td>
      <td>6</td>
      <td>1957</td>
      <td>1997</td>
      <td>1277</td>
      <td>0</td>
      <td>207</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>30</td>
      <td>30</td>
      <td>6324</td>
      <td>4</td>
      <td>6</td>
      <td>1927</td>
      <td>1950</td>
      <td>0</td>
      <td>0</td>
      <td>520</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1430</th>
      <td>1431</td>
      <td>60</td>
      <td>21930</td>
      <td>5</td>
      <td>5</td>
      <td>2005</td>
      <td>2005</td>
      <td>0</td>
      <td>0</td>
      <td>732</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1431</th>
      <td>1432</td>
      <td>120</td>
      <td>4928</td>
      <td>6</td>
      <td>6</td>
      <td>1976</td>
      <td>1976</td>
      <td>958</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1432</th>
      <td>1433</td>
      <td>30</td>
      <td>10800</td>
      <td>4</td>
      <td>6</td>
      <td>1927</td>
      <td>2007</td>
      <td>0</td>
      <td>0</td>
      <td>656</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1433</th>
      <td>1434</td>
      <td>60</td>
      <td>10261</td>
      <td>6</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>0</td>
      <td>0</td>
      <td>936</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1434</th>
      <td>1435</td>
      <td>20</td>
      <td>17400</td>
      <td>5</td>
      <td>5</td>
      <td>1977</td>
      <td>1977</td>
      <td>936</td>
      <td>0</td>
      <td>190</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1435</th>
      <td>1436</td>
      <td>20</td>
      <td>8400</td>
      <td>6</td>
      <td>9</td>
      <td>1962</td>
      <td>2005</td>
      <td>0</td>
      <td>0</td>
      <td>1319</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1436</th>
      <td>1437</td>
      <td>20</td>
      <td>9000</td>
      <td>4</td>
      <td>6</td>
      <td>1971</td>
      <td>1971</td>
      <td>616</td>
      <td>0</td>
      <td>248</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1437</th>
      <td>1438</td>
      <td>20</td>
      <td>12444</td>
      <td>8</td>
      <td>5</td>
      <td>2008</td>
      <td>2008</td>
      <td>1336</td>
      <td>0</td>
      <td>596</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1438</th>
      <td>1439</td>
      <td>20</td>
      <td>7407</td>
      <td>6</td>
      <td>7</td>
      <td>1957</td>
      <td>1996</td>
      <td>600</td>
      <td>0</td>
      <td>312</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1439</th>
      <td>1440</td>
      <td>60</td>
      <td>11584</td>
      <td>7</td>
      <td>6</td>
      <td>1979</td>
      <td>1979</td>
      <td>315</td>
      <td>110</td>
      <td>114</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1440</th>
      <td>1441</td>
      <td>70</td>
      <td>11526</td>
      <td>6</td>
      <td>7</td>
      <td>1922</td>
      <td>1994</td>
      <td>0</td>
      <td>0</td>
      <td>588</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1441</th>
      <td>1442</td>
      <td>120</td>
      <td>4426</td>
      <td>6</td>
      <td>5</td>
      <td>2004</td>
      <td>2004</td>
      <td>697</td>
      <td>0</td>
      <td>151</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1442</th>
      <td>1443</td>
      <td>60</td>
      <td>11003</td>
      <td>10</td>
      <td>5</td>
      <td>2008</td>
      <td>2008</td>
      <td>765</td>
      <td>0</td>
      <td>252</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1443</th>
      <td>1444</td>
      <td>30</td>
      <td>8854</td>
      <td>6</td>
      <td>6</td>
      <td>1916</td>
      <td>1950</td>
      <td>0</td>
      <td>0</td>
      <td>952</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1444</th>
      <td>1445</td>
      <td>20</td>
      <td>8500</td>
      <td>7</td>
      <td>5</td>
      <td>2004</td>
      <td>2004</td>
      <td>0</td>
      <td>0</td>
      <td>1422</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1445</th>
      <td>1446</td>
      <td>85</td>
      <td>8400</td>
      <td>6</td>
      <td>5</td>
      <td>1966</td>
      <td>1966</td>
      <td>187</td>
      <td>627</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1446</th>
      <td>1447</td>
      <td>20</td>
      <td>26142</td>
      <td>5</td>
      <td>7</td>
      <td>1962</td>
      <td>1962</td>
      <td>593</td>
      <td>0</td>
      <td>595</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1447</th>
      <td>1448</td>
      <td>60</td>
      <td>10000</td>
      <td>8</td>
      <td>5</td>
      <td>1995</td>
      <td>1996</td>
      <td>1079</td>
      <td>0</td>
      <td>141</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1448</th>
      <td>1449</td>
      <td>50</td>
      <td>11767</td>
      <td>4</td>
      <td>7</td>
      <td>1910</td>
      <td>2000</td>
      <td>0</td>
      <td>0</td>
      <td>560</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1449</th>
      <td>1450</td>
      <td>180</td>
      <td>1533</td>
      <td>5</td>
      <td>7</td>
      <td>1970</td>
      <td>1970</td>
      <td>553</td>
      <td>0</td>
      <td>77</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1450</th>
      <td>1451</td>
      <td>90</td>
      <td>9000</td>
      <td>5</td>
      <td>5</td>
      <td>1974</td>
      <td>1974</td>
      <td>0</td>
      <td>0</td>
      <td>896</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1451</th>
      <td>1452</td>
      <td>20</td>
      <td>9262</td>
      <td>8</td>
      <td>5</td>
      <td>2008</td>
      <td>2009</td>
      <td>0</td>
      <td>0</td>
      <td>1573</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1452</th>
      <td>1453</td>
      <td>180</td>
      <td>3675</td>
      <td>5</td>
      <td>5</td>
      <td>2005</td>
      <td>2005</td>
      <td>547</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1453</th>
      <td>1454</td>
      <td>20</td>
      <td>17217</td>
      <td>5</td>
      <td>5</td>
      <td>2006</td>
      <td>2006</td>
      <td>0</td>
      <td>0</td>
      <td>1140</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1454</th>
      <td>1455</td>
      <td>20</td>
      <td>7500</td>
      <td>7</td>
      <td>5</td>
      <td>2004</td>
      <td>2005</td>
      <td>410</td>
      <td>0</td>
      <td>811</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>1456</td>
      <td>60</td>
      <td>7917</td>
      <td>6</td>
      <td>5</td>
      <td>1999</td>
      <td>2000</td>
      <td>0</td>
      <td>0</td>
      <td>953</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>1457</td>
      <td>20</td>
      <td>13175</td>
      <td>6</td>
      <td>6</td>
      <td>1978</td>
      <td>1988</td>
      <td>790</td>
      <td>163</td>
      <td>589</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>1458</td>
      <td>70</td>
      <td>9042</td>
      <td>7</td>
      <td>9</td>
      <td>1941</td>
      <td>2006</td>
      <td>275</td>
      <td>0</td>
      <td>877</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1458</th>
      <td>1459</td>
      <td>20</td>
      <td>9717</td>
      <td>5</td>
      <td>6</td>
      <td>1950</td>
      <td>1996</td>
      <td>49</td>
      <td>1029</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1459</th>
      <td>1460</td>
      <td>20</td>
      <td>9937</td>
      <td>5</td>
      <td>6</td>
      <td>1965</td>
      <td>1965</td>
      <td>830</td>
      <td>290</td>
      <td>136</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1457 rows × 222 columns</p>
</div>




```python

```
