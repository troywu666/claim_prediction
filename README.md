# 索赔情况预测

便签：LightGBM+LR

[TOC]

## 1、背景和目标

背景：

目标：根据所提供的汽车保单持有人的数据建立机器学习模型，分析保单持有人是否会在次年提出索赔。

## 2、分析方法确定

* 该项目中的预测问题为二分类问题，与互联网的广告点击率预测问题相类似
* 数据均已做过脱敏处理，无法通过业务知识对数据特征进行处理

* GBDT集成方法为不断拟合残差的机器学习方法，在数学意义上是直接相加，通过增加LR部分可为其提供权重，因而采用GBDT的方法进行预测

## 3、数据观察与预处理

### 3.1、数据特征分类

```python
values = list(train.columns)[2: ]
cate1 = []
cate2 = []
cate_con_or_ord = []
cat_cols = []
bin_cols = []

for col in train.columns:
    cols = col.split('_')
    if len(cols) == 3:
        cate1.append(cols[1])
        cate2.append('continuous or ordinal')
        cate_con_or_ord.append(col)
    if len(cols) == 4:
        cate1.append(cols[1])
        cate2.append(cols[3])
        if cols[3] == 'cat':
            cat_cols.append(col)
        if cols[3] == 'bin':
            bin_cols.append(col)
columns_df = pd.DataFrame({'category_1': cate1, 'category_2': cate2}, index = values)
```

### 3.2、缺失值数据处理

* 查看缺失值数据分布情况

```python
import missingno as msno

train_null_values = []
train_col_missing = []

for col in values:
    if len(train[col][train[col].isnull()].index):
        train_null_values.append((col, len(train[col][train[col].isnull()].index)))
        train_col_missing.append(col)
for i in sorted(train_null_values, key = lambda x: x[1], reverse = True):
    print(i)
msno.matrix(train[train_col_missing], color=(0.42, 0.1, 0.05))
```

![png](README\output1.png)

* 分析缺失数据间的相关性

```python
msno.heatmap(df = train[train_col_missing])
```

![png](README\output2.png)

* 同理，将test数据集的缺失值情况进行可视化

![png](README\output3.png)

![png](README\output4.png)

### 3.3、各变量相关性分析

```python
import cufflinks as cf
train[values].corr().iplot(kind = 'heatmap', text = train[values].corr().values)
```

![png](README\newplot1.png)

**可见calc类的数据没有相关性**

```python
corr_values = values.copy()
for col in values:
    if 'calc' in col:
        corr_values.remove(col)
train[corr_values].corr().iplot(kind = 'heatmap', colorscale = 'spectral')
```

![png](README\newplot2.png)

### 3.4、各类特征分析

#### 3.4.1、二分类特征分析

##### 3.4.1.1、训练集二分类空值占比

```python
import cufflinks as cf

train_bin_zero_list = []
train_bin_one_list = []
for col in bin_cols:
    temp = train[col].value_counts()
    zero = temp[0]
    one = temp[1]
    train_bin_zero_list.append(zero)
    train_bin_one_list.append(one)
    
df = pd.DataFrame({'zero_counts': train_bin_zero_list, 'one_counts': train_bin_one_list}, index = bin_cols)
df.iplot(kind = 'bar', barmode = 'stack')
```

![png](README\newplot3.png)

































