## Kaggle泰坦尼克号生存预测挑战------数据分析
这是kaggle上Getting Started 的Prediction Competition，也是比较入门和简单的新人赛。

Datawhale这次学习活动，是以此赛题数据的内容进行数据分析的，我重新地回顾巩固一下这个比赛，我将分成三个部分：

- **[Kaggle泰坦尼克号生存预测挑战——数据分析](https://blog.csdn.net/weixin_43532000/article/details/108086363)**
- **Kaggle泰坦尼克号生存预测挑战——特征工程**
- **Kaggle泰坦尼克号生存预测挑战——模型建立、模型调参、融合**

## 先修知识
  - numpy的数据结构与基本用法
  - pandas的数据结构与基本用法
  - matplotlib基本用法
  


## 泰坦尼克号的沉没
&nbsp;&nbsp;&nbsp;&nbsp;1912年4月15日，在她的处女航中，被普遍认为“沉没”的RMS泰坦尼克号与冰山相撞后沉没。

&nbsp;&nbsp;&nbsp;&nbsp;不幸的是，船上没有足够的救生艇供所有人使用，导致2224名乘客和机组人员中的1502人死亡。虽然幸存有一些运气，但似乎有些人比其他人更有可能生存。

&nbsp;&nbsp;&nbsp;&nbsp;在这一挑战中，我们要求您建立一个预测模型来回答以下问题：“什么样的人更有可能生存？” 使用乘客数据（即姓名，年龄，性别，社会经济舱等）

&nbsp;&nbsp;&nbsp;&nbsp;**任务分析**：这是一个分类任务，建立模型预测幸存者

### 数据集
- 训练集：891*12,含891个样本，11+1个特征（一个是target）
- 测试集：418*11,含418个样本，11个特征

## Overview:
- PassengerId:乘客ID  --- **id号作用不大，考虑删除**
- Survived:存活 （target） ---- **标签：1代表存活 0 代表没有幸存**
- Pclass:舱位等级   ---- **分成三个等级 1 2 3**
- Name：姓名 --- **由于外国的姓氏有等级之分，有意义**
- Sex：性别 --- **女士优先**
- Age：年龄 --- **年轻人可能存活多**
- SibSp：泰坦尼克号上的兄弟姐妹/配偶数 --- **有待考查**
- Parch：泰坦尼克号上的父母/子女数量 --- **有待考查**
- Ticket：票据 --- **有待考查**
- Fare:票价 --- **票价高的可能待遇高**
- Cabin：船舱号 ---  **不同的船舱或许获得求生不同**
- Embarked：登船港口 ---- **C = Cherbourg；Q = Queenstown；S = Southampton**

### 导入相关的库


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings('ignore')
seed =12345
```

## 数据概览
1. 加载数据方法 ：pd.read_csv(),pd.read_table()
    - pd.read_csv()：读取以‘，’分割的文件到DataFrame,用于读取csv文件(csv用逗号符分隔字符段)
    - pd.read_table()：读取以‘\t’分割的文件到DataFrame,用于读取tsv文件(tsv用制表符分隔字符段)
    - 实质上两个方法都是通用的，函数中参数sep可以选定分隔符的类型
    
    ` 例如用read_csv()读取tsv文件，df = pd.read_csv(file_path,sep='\t')`
    
    

2. 处理大型文件或内存不足时，采用分块读取方式：
    - 使用参数chunksize指定文件块的大小（用于迭代）

     `df = pd.read_csv(file_path,chunksize = 100)
      for i in df:   ##用循环的方式即可迭代读取DataFrame 
        print(i) `
        

3. 更多常用参数![20180714205358426.png](attachment:20180714205358426.png)
     `


```python
##加载数据  不建议把特征名转成中文，在画图时可能出现乱码
train_df = pd.read_csv('data_train.csv')
test_df = pd.read_csv('data_test.csv')
##数据预览 查看前5行的数据
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



- DataFrame.info(): 可以对数据进行简要的概览,非空样本数，特征行的类型，特征数
- DataFrame.describe(): 输出数值型特征的一些统计量


```python
train_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB
    


```python
train_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>714.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
      <td>891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>446.000000</td>
      <td>0.383838</td>
      <td>2.308642</td>
      <td>29.699118</td>
      <td>0.523008</td>
      <td>0.381594</td>
      <td>32.204208</td>
    </tr>
    <tr>
      <th>std</th>
      <td>257.353842</td>
      <td>0.486592</td>
      <td>0.836071</td>
      <td>14.526497</td>
      <td>1.102743</td>
      <td>0.806057</td>
      <td>49.693429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.420000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>223.500000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>20.125000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.910400</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>446.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>668.500000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>38.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>891.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>6.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 11 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  418 non-null    int64  
     1   Pclass       418 non-null    int64  
     2   Name         418 non-null    object 
     3   Sex          418 non-null    object 
     4   Age          332 non-null    float64
     5   SibSp        418 non-null    int64  
     6   Parch        418 non-null    int64  
     7   Ticket       418 non-null    object 
     8   Fare         417 non-null    float64
     9   Cabin        91 non-null     object 
     10  Embarked     418 non-null    object 
    dtypes: float64(2), int64(4), object(5)
    memory usage: 36.0+ KB
    


```python
test_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>332.000000</td>
      <td>418.000000</td>
      <td>418.000000</td>
      <td>417.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1100.500000</td>
      <td>2.265550</td>
      <td>30.272590</td>
      <td>0.447368</td>
      <td>0.392344</td>
      <td>35.627188</td>
    </tr>
    <tr>
      <th>std</th>
      <td>120.810458</td>
      <td>0.841838</td>
      <td>14.181209</td>
      <td>0.896760</td>
      <td>0.981429</td>
      <td>55.907576</td>
    </tr>
    <tr>
      <th>min</th>
      <td>892.000000</td>
      <td>1.000000</td>
      <td>0.170000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>996.250000</td>
      <td>1.000000</td>
      <td>21.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.895800</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1100.500000</td>
      <td>3.000000</td>
      <td>27.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1204.750000</td>
      <td>3.000000</td>
      <td>39.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.500000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1309.000000</td>
      <td>3.000000</td>
      <td>76.000000</td>
      <td>8.000000</td>
      <td>9.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>



## 数据的探索性分析
 


```python
### 可以根据自己的需求封装一个函数
def _data_info(data,categorical_features):
    print('number of train examples = {}'.format(data.shape[0]))
    print('number of train Shape = {}'.format(data.shape))
    print('Features={}'.format(data.columns))
    print('\n--------输出类别特征的种类--------')
    for i in categorical_features:
        if i in list(data.columns):
            print("train："+i+":",list(data[i].unique()))
    print('\n--------缺失值--------')
    missing = data.isnull().sum()
    missing = missing[missing > 0]
    print(missing)
    missing.sort_values(inplace=True)
    missing.plot.bar()
    plt.show()
def data_info(data_train,data_test,categorical_features):
    print('--------训练集基本概况--------')
    _data_info(data_train,categorical_features)
    print('\n\n--------测试集基本概况--------')
    _data_info(data_test,categorical_features)

```

### 数据概况、类别特征、缺失值情况
- 训练集的样本数:891, 特征数：11+1（一个标签）
- 测试集的样本数:418, 特征数：11（一个标签）

  类别特征的情况：
    1. Survived (标签): 取值{0,1},对应未幸存和幸存
    2. Pclass: 取值{1,2,3},对应船舱级别
    3. Sex: 取值{male,female},对应性别
    4. Cabin: 船舱号
    5. Embarked:取值{S,C,Q}。对应登船港口


```python
data_info(train_df,test_df,['Survived','Pclass','Sex','Cabin','Embarked','SibSp','Parch'])
```

    --------训练集基本概况--------
    number of train examples = 891
    number of train Shape = (891, 12)
    Features=Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
           'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
          dtype='object')
    
    --------输出类别特征的种类--------
    train：Survived: [0, 1]
    train：Pclass: [3, 1, 2]
    train：Sex: ['male', 'female']
    train：Cabin: [nan, 'C85', 'C123', 'E46', 'G6', 'C103', 'D56', 'A6', 'C23 C25 C27', 'B78', 'D33', 'B30', 'C52', 'B28', 'C83', 'F33', 'F G73', 'E31', 'A5', 'D10 D12', 'D26', 'C110', 'B58 B60', 'E101', 'F E69', 'D47', 'B86', 'F2', 'C2', 'E33', 'B19', 'A7', 'C49', 'F4', 'A32', 'B4', 'B80', 'A31', 'D36', 'D15', 'C93', 'C78', 'D35', 'C87', 'B77', 'E67', 'B94', 'C125', 'C99', 'C118', 'D7', 'A19', 'B49', 'D', 'C22 C26', 'C106', 'C65', 'E36', 'C54', 'B57 B59 B63 B66', 'C7', 'E34', 'C32', 'B18', 'C124', 'C91', 'E40', 'T', 'C128', 'D37', 'B35', 'E50', 'C82', 'B96 B98', 'E10', 'E44', 'A34', 'C104', 'C111', 'C92', 'E38', 'D21', 'E12', 'E63', 'A14', 'B37', 'C30', 'D20', 'B79', 'E25', 'D46', 'B73', 'C95', 'B38', 'B39', 'B22', 'C86', 'C70', 'A16', 'C101', 'C68', 'A10', 'E68', 'B41', 'A20', 'D19', 'D50', 'D9', 'A23', 'B50', 'A26', 'D48', 'E58', 'C126', 'B71', 'B51 B53 B55', 'D49', 'B5', 'B20', 'F G63', 'C62 C64', 'E24', 'C90', 'C45', 'E8', 'B101', 'D45', 'C46', 'D30', 'E121', 'D11', 'E77', 'F38', 'B3', 'D6', 'B82 B84', 'D17', 'A36', 'B102', 'B69', 'E49', 'C47', 'D28', 'E17', 'A24', 'C50', 'B42', 'C148']
    train：Embarked: ['S', 'C', 'Q', nan]
    train：SibSp: [1, 0, 3, 4, 2, 5, 8]
    train：Parch: [0, 1, 2, 5, 3, 4, 6]
    
    --------缺失值--------
    Age         177
    Cabin       687
    Embarked      2
    dtype: int64
    


![png](output_18_1.png)


    
    
    --------测试集基本概况--------
    number of train examples = 418
    number of train Shape = (418, 11)
    Features=Index(['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch',
           'Ticket', 'Fare', 'Cabin', 'Embarked'],
          dtype='object')
    
    --------输出类别特征的种类--------
    train：Pclass: [3, 2, 1]
    train：Sex: ['male', 'female']
    train：Cabin: [nan, 'B45', 'E31', 'B57 B59 B63 B66', 'B36', 'A21', 'C78', 'D34', 'D19', 'A9', 'D15', 'C31', 'C23 C25 C27', 'F G63', 'B61', 'C53', 'D43', 'C130', 'C132', 'C101', 'C55 C57', 'B71', 'C46', 'C116', 'F', 'A29', 'G6', 'C6', 'C28', 'C51', 'E46', 'C54', 'C97', 'D22', 'B10', 'F4', 'E45', 'E52', 'D30', 'B58 B60', 'E34', 'C62 C64', 'A11', 'B11', 'C80', 'F33', 'C85', 'D37', 'C86', 'D21', 'C89', 'F E46', 'A34', 'D', 'B26', 'C22 C26', 'B69', 'C32', 'B78', 'F E57', 'F2', 'A18', 'C106', 'B51 B53 B55', 'D10 D12', 'E60', 'E50', 'E39 E41', 'B52 B54 B56', 'C39', 'B24', 'D28', 'B41', 'C7', 'D40', 'D38', 'C105']
    train：Embarked: ['Q', 'S', 'C']
    train：SibSp: [0, 1, 2, 3, 4, 5, 8]
    train：Parch: [0, 1, 3, 2, 4, 6, 5, 9]
    
    --------缺失值--------
    Age       86
    Fare       1
    Cabin    327
    dtype: int64
    


![png](output_18_3.png)


### 缺失值情况
了解了缺失值的情况后，可以对其进行简单的填充

1.训练集缺失值：
- Age  :       177
- Cabin:       687 
- Embarked:      2 

1.测试集缺失值：418
- Age:86
- Fare  : 1  
- Cabin : 327 


```python
#将数据合并一起处理,添加一个train特征用于区分训练集和测试集
train_df['train'] = 1
test_df['train'] = 0
data_df = pd.concat([train_df,test_df],sort=True).reset_index(drop=True)
## 删除PassengerId特征
data_df.drop('PassengerId',inplace=True,axis=1)

## 先将非数字的类别特征数字化
from sklearn import preprocessing
ler_sex = preprocessing.LabelEncoder()
ler_sex.fit(data_df['Sex'])
data_df['Sex'] = ler_sex.transform(data_df['Sex'])


```

### Embarked
缺失数量少，考虑使用众值进行填充


```python
data_df[data_df['Embarked'].isnull()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>train</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>61</th>
      <td>38.0</td>
      <td>B28</td>
      <td>NaN</td>
      <td>80.0</td>
      <td>Icard, Miss. Amelie</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>113572</td>
      <td>1</td>
    </tr>
    <tr>
      <th>829</th>
      <td>62.0</td>
      <td>B28</td>
      <td>NaN</td>
      <td>80.0</td>
      <td>Stone, Mrs. George Nelson (Martha Evelyn)</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>113572</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_df['Embarked'].fillna(data_df['Embarked'].mode()[0],inplace=True)
```


```python
## 填充完Embarker后，先将非数字的类别特征数字化
ler_Embarked = preprocessing.LabelEncoder()
ler_Embarked.fit(data_df['Embarked'])
data_df['Embarked'] = ler_Embarked.transform(data_df['Embarked'])
```

### Age
${177+86\over891+418}\approx 20\%$


缺失率约20%,考虑对齐进行填充，如果直接采用数据集的数据特征进行填充，效果可能可能不是很好

尝试结合其他聚合特征对Age进行填充，从相关性程度分析中可以看出,Pclass的程度较大， 


```python
abs(data_df.corr()['Age']).sort_values(ascending=False)
```




    Age         1.000000
    Pclass      0.408106
    SibSp       0.243699
    Fare        0.178740
    Parch       0.150917
    Embarked    0.080195
    Survived    0.077221
    Sex         0.063645
    train       0.018528
    Name: Age, dtype: float64



### 年龄分布


```python
y = data_df['Age']
plt.figure(1)
plt.title('Distribution of Age')
sns.distplot(y, kde=True)

## 不同性别的年龄分布，可以看出他们的分布趋于相同
plt.figure(2);
Age_Sex0 = data_df.loc[data_df['Sex']==0,'Age']
Age_Sex1 = data_df.loc[data_df['Sex']==1,'Age']
plt.title('Distribution of Age in Sex');
sns.distplot(Age_Sex0, kde=True);
sns.distplot(Age_Sex1, kde=True);
plt.legend(['Sex0','Sex1']);

```


![png](output_28_0.png)



![png](output_28_1.png)


### 不同Pclass中的年龄分布，从其中的分布确实能看出有一些差别


```python
Age_p1 = data_df.loc[data_df['Pclass']==1,'Age']
Age_p2 = data_df.loc[data_df['Pclass']==2,'Age']
Age_p3 = data_df.loc[data_df['Pclass']==3,'Age']
sns.distplot(Age_p1,kde=True,color='b')
sns.distplot(Age_p2,kde=True,color='green')
sns.distplot(Age_p3,kde=True,color='grey')
plt.title('Distribution of Age in Pclass')
plt.legend(['p1','p2','p3'])
```




    <matplotlib.legend.Legend at 0x1daa6aad348>




![png](output_30_1.png)



```python
age_by_pclass = data_df.groupby([ 'Pclass']).median()['Age']
for pclass in range(1, 4):
    print('Median age of passengers in Pclass {}: {}'.format(pclass,age_by_pclass[pclass]))
print('Median age of all passengers: {}'.format(data_df['Age'].median()))

# 根据Pclass填充Age值 选择中位数进行填充
data_df['Age'] = data_df.groupby(['Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
```

    Median age of passengers in Pclass 1: 39.0
    Median age of passengers in Pclass 2: 29.0
    Median age of passengers in Pclass 3: 24.0
    Median age of all passengers: 28.0
    

### Fare
Fare缺失的样本只有一个，可以考虑直接用数据集的统计特征填充

不过看到SibSp,Parch都等于0,说明购置的是单人票，Plass又是机舱等级，可以考虑聚合以上属性


```python
data_df[data_df['Fare'].isnull()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>train</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1043</th>
      <td>60.5</td>
      <td>NaN</td>
      <td>2</td>
      <td>NaN</td>
      <td>Storey, Mr. Thomas</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>NaN</td>
      <td>3701</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Pclass对价格的影响很大
abs(data_df.corr()['Fare']).sort_values(ascending=False)
```




    Fare        1.000000
    Pclass      0.558629
    Survived    0.257307
    Embarked    0.238005
    Parch       0.221539
    Age         0.202512
    Sex         0.185523
    SibSp       0.160238
    train       0.030831
    Name: Fare, dtype: float64




```python
## 聚合数据属性
print(data_df.groupby(['Pclass', 'Parch','SibSp','Embarked']).Fare.max()[3][0][0][0])
print(data_df.groupby(['Pclass', 'Parch','SibSp','Embarked']).Fare.min()[3][0][0][0])
print(data_df.groupby(['Pclass', 'Parch','SibSp','Embarked']).Fare.median()[3][0][0][0])
print(data_df.groupby(['Pclass', 'Parch','SibSp','Embarked']).Fare.mean()[3][0][0][0])
## 选择中位数进行填充
data_df['Fare'].fillna(data_df.groupby(['Pclass', 'Parch','SibSp','Embarked'])['Fare'].median()[3][0][0][0],inplace=True)
```

    18.7875
    4.0125
    7.2292
    7.923984210526318
    

### Cabin

Cabin缺失较多，如果没有很好的填充数据的方法时，建议将其直接删除。



```python
data_df.drop('Cabin',inplace=True,axis=1)
```

### 数据缺失填充完成

#### 继续分析数据


```python
train_data = data_df[data_df.train==1]
train_data['Survived'] = train_df['Survived']
train_data.drop('train',axis=1,inplace=True)
test_data = data_df[data_df.train==0]
test_data.drop(['Survived','train'],axis=1,inplace=True)
```

## 特征相关程度分析
#### 训练集


```python
train_data.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>1.000000</td>
      <td>-0.012936</td>
      <td>0.126256</td>
      <td>-0.172704</td>
      <td>-0.415037</td>
      <td>0.078710</td>
      <td>-0.244183</td>
      <td>-0.046230</td>
    </tr>
    <tr>
      <th>Embarked</th>
      <td>-0.012936</td>
      <td>1.000000</td>
      <td>-0.224719</td>
      <td>0.039798</td>
      <td>0.162098</td>
      <td>0.108262</td>
      <td>0.068230</td>
      <td>-0.167675</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>0.126256</td>
      <td>-0.224719</td>
      <td>1.000000</td>
      <td>0.216225</td>
      <td>-0.549500</td>
      <td>-0.182333</td>
      <td>0.159651</td>
      <td>0.257307</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>-0.172704</td>
      <td>0.039798</td>
      <td>0.216225</td>
      <td>1.000000</td>
      <td>0.018443</td>
      <td>-0.245489</td>
      <td>0.414838</td>
      <td>0.081629</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>-0.415037</td>
      <td>0.162098</td>
      <td>-0.549500</td>
      <td>0.018443</td>
      <td>1.000000</td>
      <td>0.131900</td>
      <td>0.083081</td>
      <td>-0.338481</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>0.078710</td>
      <td>0.108262</td>
      <td>-0.182333</td>
      <td>-0.245489</td>
      <td>0.131900</td>
      <td>1.000000</td>
      <td>-0.114631</td>
      <td>-0.543351</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>-0.244183</td>
      <td>0.068230</td>
      <td>0.159651</td>
      <td>0.414838</td>
      <td>0.083081</td>
      <td>-0.114631</td>
      <td>1.000000</td>
      <td>-0.035322</td>
    </tr>
    <tr>
      <th>Survived</th>
      <td>-0.046230</td>
      <td>-0.167675</td>
      <td>0.257307</td>
      <td>0.081629</td>
      <td>-0.338481</td>
      <td>-0.543351</td>
      <td>-0.035322</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
### 从幸存和性别成负相关程度较大
### 从幸存和Pclass 成负相关程度较大
### 从幸存和Fare 成负相关程度较大
train_data.corr()['Survived'].sort_values(ascending=False)
```




    Survived    1.000000
    Fare        0.257307
    Parch       0.081629
    SibSp      -0.035322
    Age        -0.046230
    Embarked   -0.167675
    Pclass     -0.338481
    Sex        -0.543351
    Name: Survived, dtype: float64




```python
plt.figure( figsize=(8, 8))
plt.title('Train Set Correlation HeatMap ',y=1,size=16)
sns.heatmap(train_data.corr(),square = True,  vmax=0.7,annot=True,cmap='Accent')
```




    <AxesSubplot:title={'center':'Train Set Correlation HeatMap '}>




![png](output_43_1.png)



```python
## 存活情况
plt.bar(['Not Survived','Survived'],train_data['Survived'].value_counts().values)
plt.title('Train_Set_Survived')
```




    Text(0.5, 1.0, 'Train_Set_Survived')




![png](output_44_1.png)


#### 测试集


```python
test_data.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>1.000000</td>
      <td>-0.134927</td>
      <td>0.348189</td>
      <td>-0.053829</td>
      <td>-0.525193</td>
      <td>-0.006212</td>
      <td>-0.077618</td>
    </tr>
    <tr>
      <th>Embarked</th>
      <td>-0.134927</td>
      <td>1.000000</td>
      <td>-0.257805</td>
      <td>0.054577</td>
      <td>0.227983</td>
      <td>0.076281</td>
      <td>0.052708</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>0.348189</td>
      <td>-0.257805</td>
      <td>1.000000</td>
      <td>0.230418</td>
      <td>-0.577504</td>
      <td>-0.192244</td>
      <td>0.172043</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>-0.053829</td>
      <td>0.054577</td>
      <td>0.230418</td>
      <td>1.000000</td>
      <td>0.018721</td>
      <td>-0.159120</td>
      <td>0.306895</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>-0.525193</td>
      <td>0.227983</td>
      <td>-0.577504</td>
      <td>0.018721</td>
      <td>1.000000</td>
      <td>0.108615</td>
      <td>0.001087</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>-0.006212</td>
      <td>0.076281</td>
      <td>-0.192244</td>
      <td>-0.159120</td>
      <td>0.108615</td>
      <td>1.000000</td>
      <td>-0.099943</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>-0.077618</td>
      <td>0.052708</td>
      <td>0.172043</td>
      <td>0.306895</td>
      <td>0.001087</td>
      <td>-0.099943</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure( figsize=(8, 8))
plt.title('Test Set Correlation HeatMap ',y=1,size=16)
sns.heatmap(test_data.corr(),square = True,  vmax=0.7,annot=True,cmap='Accent')
```




    <AxesSubplot:title={'center':'Test Set Correlation HeatMap '}>




![png](output_47_1.png)


### 连续性数据分布的情况


```python
continue_features = ['Age', 'Fare']
survived = train_data['Survived'] == 1

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(15, 15))
plt.subplots_adjust(right=1.5)

for i, feature in enumerate(continue_features):    
    sns.distplot(train_data[~survived][feature], label='Not Survived', hist=True, color='#e74c3c', ax=axs[0][i])
    sns.distplot(train_data[survived][feature], label='Survived', hist=True, color='#2ecc71', ax=axs[0][i])
    
    sns.distplot(train_data[feature], label='Training Set', hist=False, color='#e74c3c', ax=axs[1][i])
    sns.distplot(test_data[feature], label='Test Set', hist=False, color='#2ecc71', ax=axs[1][i])
    
    axs[0][i].set_xlabel('')
    axs[1][i].set_xlabel('')
    
    for j in range(2):        
        axs[i][j].tick_params(axis='x', labelsize=20)
        axs[i][j].tick_params(axis='y', labelsize=20)
    
    axs[0][i].legend(loc='upper right', prop={'size': 20})
    axs[1][i].legend(loc='upper right', prop={'size': 20})
    axs[0][i].set_title('Distribution of Survival in {}'.format(feature), size=20, y=1.05)

axs[1][0].set_title('Distribution of {} Feature'.format('Age'), size=20, y=1.05)
axs[1][1].set_title('Distribution of {} Feature'.format('Fare'), size=20, y=1.05)
plt.show()
```


![png](output_49_0.png)


### 类别特征分布情况
- Embarked ：取值为0时的，幸存率较高
- Sex: 取值为0时，幸存率较高
- Pclass :幸存率由1到3 递减
- SibSp和Parch的幸存率大致分布一致，可以考虑将他们合并成一个Family特征


```python
Categorical_features = ['Embarked', 'Parch','SibSp','Sex', 'Pclass']

fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(20, 20))
plt.subplots_adjust(right=1.5, top=1.25)

for i, feature in enumerate(Categorical_features, 1):    
    plt.subplot(2, 3, i)
    sns.countplot(x=feature, hue='Survived', data=train_data)
    
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    
    plt.xlabel('{}'.format(feature), size=20, labelpad=15)
    plt.ylabel('Passenger Count', size=20, labelpad=15)    
    plt.legend(['Not Survived', 'Survived'], loc='upper center')
    plt.title('Count of Survival in {} Feature'.format(feature), size=20, y=1.05)
plt.show()


fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(15, 15))
plt.subplots_adjust(right=1.5, top=1.25)
for i, feature in enumerate(Categorical_features, 1):    
    plt.subplot(2, 3, i)
    sns.pointplot(feature,y='Survived',data=train_data)
    
    plt.tick_params(axis='x', labelsize=20)
    plt.tick_params(axis='y', labelsize=20)
    
    plt.xlabel('{}'.format(feature), size=20, labelpad=15)
    plt.ylabel('Passenger Count', size=20, labelpad=15)    
    plt.title('Rate of Survival in {} Feature'.format(feature), size=20, y=1.05)
plt.show()



```


![png](output_51_0.png)



![png](output_51_1.png)


### 对姓名特征进行分析


```python
train_data['Title'] = train_data['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}

train_data['Title'] = train_data['Title'].map(Title_Dictionary)
sns.barplot(x="Title", y="Survived", data=train_data)
```




    <AxesSubplot:xlabel='Title', ylabel='Survived'>




![png](output_53_1.png)


### 保存CSV文件


```python
train_data.to_csv('./train.csv',index=False)
test_data.to_csv('./test.csv',index=False)
```

# 特征工程
特征工程很重要：数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已。
![122.jpg](attachment:122.jpg)


```python
#将数据合并一起处理,添加一个train特征用于区分训练集和测试集
train_data['train'] = 1
test_data['train'] = 0
data_all = pd.concat([train_data,test_data],sort=True).reset_index(drop=True)
```

### 对Fare的各个区间进行分析
对Fare进行区间划分之后，可以明显从区间的生存率看出，船票越高的游客幸存的机会更大


```python
## 提取幸存的和未幸存的
survived = train_data['Survived'] == 1
max_fare = train_data['Fare'].max()
print('max_fare:',max_fare)
Fare_survived=train_data[survived]['Fare']
Fare_Notsurvived=train_data[~survived]['Fare']


## 幸存 数据分桶：由于分布不均，在不同的区间选择了不同的划分
cut_list = list(range(0,100,10))
cut_list.extend(list(range(100,610,100)))
Fare_survived_cut=pd.cut(Fare_survived,cut_list,right=False)
Fare_survived_counts=Fare_survived_cut.value_counts(sort= False)

## 未幸存 数据分桶：由于分布不均，在不同的区间选择了不同的划分
Fare_Notsurvived_cut=pd.cut(Fare_Notsurvived,cut_list,right=False)
Fare_Notsurvived_counts=Fare_Notsurvived_cut.value_counts(sort=False)

## 幸存+未幸存
Passenger_Fare_counts = Fare_survived_counts+Fare_Notsurvived_counts
RateOfFare_survived_counts = Fare_survived_counts/Passenger_Fare_counts

plt.figure(figsize=(10,10))
plt.subplots_adjust(right=2,hspace =0.5)
plt.subplot(2,2,1)
Fare_survived_counts.plot.bar()
plt.title('Distribution of Survived in Fare interval ',size=15)
plt.ylabel('Passenger Count',size=15)
plt.subplot(2,2,2)
Fare_Notsurvived_counts.plot.bar()
plt.title('Distribution of Unsurvived in Fare interval ',size=15)
plt.ylabel('Passenger Count',size=15)

plt.subplot(2,2,3)
Passenger_Fare_counts.plot.bar()
plt.title('Distribution of All Passenger in Fare interval ',size=15)
plt.ylabel('Passenger Count',size=15)

plt.subplot(2,2,4)
RateOfFare_survived_counts.plot.bar(color='r')
plt.title('Distribution of Rate_survived in Fare interval ',size=15)
plt.ylabel('Rate Survived',size=15)
plt.show()
```

    max_fare: 512.3292
    


![png](output_59_1.png)


### 对Age的各个区间进行分析
对Age划分区间进行分析后，发现幼儿的生存机会似乎更大


```python
## 提取幸存的和未幸存的
survived = train_data['Survived'] == 1
max_age = train_data['Age'].max()
print('max_age:',max_age)
Age_survived=train_data[survived]['Age']
Age_Notsurvived=train_data[~survived]['Age']


## 幸存 
## 未成年进行了细分
cut_list = [0,4,8,12,18,30,40,55,65,80]

Age_survived_cut=pd.cut(Age_survived,cut_list,right=True)
Age_survived_counts=Age_survived_cut.value_counts(sort=False)

## 未幸存 
Age_Notsurvived_cut=pd.cut(Age_Notsurvived,cut_list,right=True)
Age_Notsurvived_counts=Age_Notsurvived_cut.value_counts(sort=False)

## 幸存+未幸存
Passenger_Age_counts = Age_survived_counts+Age_Notsurvived_counts
RateOfAge_survived_counts = Age_survived_counts/Passenger_Age_counts

plt.figure(figsize=(10,10))
plt.subplots_adjust(right=2,hspace =0.5)
plt.subplot(2,2,1)
Age_survived_counts.plot.bar()
plt.title('Distribution of Survived in Age interval ',size=15)
plt.ylabel('Passenger Count',size=15)
plt.subplot(2,2,2)
Age_Notsurvived_counts.plot.bar()
plt.title('Distribution of Unsurvived in Age interval ',size=15)
plt.ylabel('Passenger Count',size=15)

plt.subplot(2,2,3)
Passenger_Age_counts.plot.bar()
plt.title('Distribution of All Passenger in Age interval ',size=15)
plt.ylabel('Passenger Count',size=15)

plt.subplot(2,2,4)
RateOfAge_survived_counts.plot.bar(color='r')
plt.title('Distribution of Rate_survived in Age interval ',size=15)
plt.ylabel('Rate Survived',size=15)
plt.show()
```

    max_age: 80.0
    


![png](output_61_1.png)


### Fare ，Age
对连续的特征进行离散化，得到粗粒度的特征

对Fare，Age分桶操作


```python
cut_list = list(range(0,100,10))
cut_list.extend(list(range(100,700,100)))
Fare_cut=pd.cut(data_all['Fare'],cut_list,labels=False,right=False)
data_all['Fare_bin']=Fare_cut
cut_list = [0,4,8,12,18,30,40,55,65,80]
Age_cut=pd.cut(data_all['Age'],cut_list,labels=False)
data_all['Age_bin']=Age_cut
data_all[['Age','Age_bin','Fare','Fare_bin']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Age_bin</th>
      <th>Fare</th>
      <th>Fare_bin</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>4</td>
      <td>7.2500</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>5</td>
      <td>71.2833</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>4</td>
      <td>7.9250</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.0</td>
      <td>5</td>
      <td>53.1000</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>5</td>
      <td>8.0500</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### SibSp、Parch

SibSp和Parch的幸存率大致分布一致，可以考虑将他们合并成一个Family(SibSp+Parch+1)特征

- 可以看出Family ==1 时，幸存率很低
- Family == 2 ,3, 4 的幸存率高
- Family == 5 ,6 ,7 的幸存率很低
- Family > 7 几乎没有幸存


```python
survived = train_data['Survived'] == 1
train_data['Family']=train_data['SibSp']+train_data['Parch']+1
sns.countplot(x='Family', hue='Survived', data=train_data)
plt.title('Count of Survival in Family Feature',size=15)
```




    Text(0.5, 1.0, 'Count of Survival in Family Feature')




![png](output_65_1.png)


### SibSp,Parch
SibSp和Parch的幸存率大致分布一致，可以考虑将他们合并成一个Family(SibSp+Parch+1)特征
- Family值为1时，即代表单身人士：single
- Family值为2,3,4时，即代表小型的家庭：small
- Family值为5,6,7时，即代表中型的家庭：medium
- Family值大于7时，代表大型的家庭:Large


```python
data_all['Family']=data_all['SibSp']+data_all['Parch']+1
Family_category_map = {1: 'Single', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Medium', 8: 'Large', 11: 'Large'}
data_all['Family_category'] = data_all['Family'].map(Family_category_map)
```


```python
data_all[['Family_category','Family']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Family_category</th>
      <th>Family</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Small</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Small</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Single</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Small</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Single</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Name
从名字中可以看出一些乘船人的身份


```python
data_all['Title'] = data_all['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}
data_all['Title'] = data_all['Title'].map(Title_Dictionary)
data_all[['Name','Title']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Braund, Mr. Owen Harris</td>
      <td>Mr</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Heikkinen, Miss. Laina</td>
      <td>Miss</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>Mrs</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Allen, Mr. William Henry</td>
      <td>Mr</td>
    </tr>
  </tbody>
</table>
</div>




```python
data_all.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Embarked</th>
      <th>Fare</th>
      <th>Name</th>
      <th>Parch</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Survived</th>
      <th>Ticket</th>
      <th>Title</th>
      <th>train</th>
      <th>Fare_bin</th>
      <th>Age_bin</th>
      <th>Family</th>
      <th>Family_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>2</td>
      <td>7.2500</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0.0</td>
      <td>A/5 21171</td>
      <td>Mr</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>Small</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38.0</td>
      <td>0</td>
      <td>71.2833</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>PC 17599</td>
      <td>Mrs</td>
      <td>1</td>
      <td>7</td>
      <td>5</td>
      <td>2</td>
      <td>Small</td>
    </tr>
    <tr>
      <th>2</th>
      <td>26.0</td>
      <td>2</td>
      <td>7.9250</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>STON/O2. 3101282</td>
      <td>Miss</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>Single</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35.0</td>
      <td>2</td>
      <td>53.1000</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1.0</td>
      <td>113803</td>
      <td>Mrs</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>Small</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35.0</td>
      <td>2</td>
      <td>8.0500</td>
      <td>Allen, Mr. William Henry</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0.0</td>
      <td>373450</td>
      <td>Mr</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>Single</td>
    </tr>
  </tbody>
</table>
</div>



### Ticket
提取Ticket特征,一张Ticket可以供多人使用


```python
Ticket_Count = dict(data_all['Ticket'].value_counts())
data_all['TicketGroup'] = data_all['Ticket'].apply(lambda x:Ticket_Count[x])
sns.barplot(x='TicketGroup', y='Survived', data=data_all)
```




    <AxesSubplot:xlabel='TicketGroup', ylabel='Survived'>




![png](output_73_1.png)



```python
## 处理类别特征
def process_category_feature(data,category_feature=None):
    for feature in category_feature:   
        # onehot
        feature_dummies = pd.get_dummies(data[feature], prefix=feature)
        data = pd.concat([data, feature_dummies],axis=1)
        ## 删除
        data.drop(feature,axis=1,inplace=True)
    return data
```


```python
data_all=process_category_feature(data_all,category_feature=['Pclass','Embarked','Title','Family_category','Fare_bin','Age_bin','Sex'])
```


```python
data_all.drop(['Name','Ticket','SibSp','Parch'],axis=1,inplace=True)
```


```python
#对数据进行归一化处理
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
##Age
age_scale = scaler.fit(data_all['Age'].values.reshape(-1, 1))
data_all['Age'] = age_scale.transform(data_all['Age'].values.reshape(-1, 1))
##Fare
fare_scale = scaler.fit(data_all['Fare'].values.reshape(-1, 1))
data_all['Fare'] = scaler.transform(data_all['Fare'].values.reshape(-1, 1))
```


```python
train_data = data_all[data_all['train']==1]
test_data = data_all[data_all['train']==0]
y=train_df['Survived'].values
train_data.drop(['train','Survived'],axis=1,inplace=True)
test_data.drop(['train','Survived'],axis=1,inplace=True)
```


```python
train_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 891 entries, 0 to 890
    Data columns (total 43 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   Age                     891 non-null    float64
     1   Family                  891 non-null    int64  
     2   Fare                    891 non-null    float64
     3   Pclass_1                891 non-null    uint8  
     4   Pclass_2                891 non-null    uint8  
     5   Pclass_3                891 non-null    uint8  
     6   Embarked_0              891 non-null    uint8  
     7   Embarked_1              891 non-null    uint8  
     8   Embarked_2              891 non-null    uint8  
     9   Title_Master            891 non-null    uint8  
     10  Title_Miss              891 non-null    uint8  
     11  Title_Mr                891 non-null    uint8  
     12  Title_Mrs               891 non-null    uint8  
     13  Title_Officer           891 non-null    uint8  
     14  Title_Royalty           891 non-null    uint8  
     15  Family_category_Large   891 non-null    uint8  
     16  Family_category_Medium  891 non-null    uint8  
     17  Family_category_Single  891 non-null    uint8  
     18  Family_category_Small   891 non-null    uint8  
     19  Fare_bin_0              891 non-null    uint8  
     20  Fare_bin_1              891 non-null    uint8  
     21  Fare_bin_2              891 non-null    uint8  
     22  Fare_bin_3              891 non-null    uint8  
     23  Fare_bin_4              891 non-null    uint8  
     24  Fare_bin_5              891 non-null    uint8  
     25  Fare_bin_6              891 non-null    uint8  
     26  Fare_bin_7              891 non-null    uint8  
     27  Fare_bin_8              891 non-null    uint8  
     28  Fare_bin_9              891 non-null    uint8  
     29  Fare_bin_10             891 non-null    uint8  
     30  Fare_bin_11             891 non-null    uint8  
     31  Fare_bin_14             891 non-null    uint8  
     32  Age_bin_0               891 non-null    uint8  
     33  Age_bin_1               891 non-null    uint8  
     34  Age_bin_2               891 non-null    uint8  
     35  Age_bin_3               891 non-null    uint8  
     36  Age_bin_4               891 non-null    uint8  
     37  Age_bin_5               891 non-null    uint8  
     38  Age_bin_6               891 non-null    uint8  
     39  Age_bin_7               891 non-null    uint8  
     40  Age_bin_8               891 non-null    uint8  
     41  Sex_0                   891 non-null    uint8  
     42  Sex_1                   891 non-null    uint8  
    dtypes: float64(2), int64(1), uint8(40)
    memory usage: 62.6 KB
    


```python
print('train_data Shape',train_data.shape)
print('test_data Shape',test_data.shape)
```

    train_data Shape (891, 43)
    test_data Shape (418, 43)
    


```python
from sklearn.model_selection import train_test_split
features=train_data.values
##将数据分成 训练集和测试集 ：比例为 8：2
X_train, X_test, Y_train, Y_test = train_test_split(features, y, test_size=0.2, random_state=seed)
```


```python
from sklearn.ensemble import RandomForestClassifier
model_all = RandomForestClassifier(criterion='gini', 
                                           n_estimators=1100,
                                           max_depth=5,
                                           min_samples_split=4,
                                           min_samples_leaf=5,
                                           max_features='auto',
                                           oob_score=True,
                                           random_state=42,
                                           n_jobs=-1,
                                           verbose=1)
```


```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics as metrics
##记录准确率
train_accuary=0
val_accuary=0
#决策树模型
model_all=LinearSVC(dual=False)
#使用  10折交叉验证
n=10
sk=StratifiedKFold(n_splits=n,shuffle=True,random_state=seed) 
for train,val in sk.split(X_train,Y_train):
    X_tr=X_train[train]
    y_tr=Y_train[train]
    X_val=X_train[val]
    y_val=Y_train[val]
    model_all.fit(X_tr,y_tr)
    pre_train=model_all.predict(X_tr)    
    pre_val=model_all.predict(X_val)
    train_accuary+=(metrics.accuracy_score(y_tr,pre_train))
    val_accuary+=(metrics.accuracy_score(y_val,pre_val))

##输出平均的
print('train_avg_accuary:%0.3f'%(train_accuary/n))
print('val_avg_accuary:%0.3f'%(val_accuary/n))
```

    train_avg_accuary:0.844
    val_avg_accuary:0.819
    


```python
pre=model_all.predict(test_data)
```


```python
submit = pd.DataFrame({'PassengerId':np.array(range(892,1310)),'Survived':pre})
submit.to_csv('submit.csv',index=False)
```


```python

```
