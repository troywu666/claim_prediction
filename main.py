#!/usr/bin/env python
# coding: utf-8

# # 数据分析
# ## 载入数据

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import seaborn as sns
import lightgbm as lgb


# In[2]:


train = pd.read_csv('./train/train.csv', na_values = -1)
test = pd.read_csv('./test/test.csv', na_values = -1)

train['target'].value_counts()##数据不均衡


# ## 特征分类

# In[3]:


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
columns_df


# In[4]:


test[cate_con_or_ord].info()


# In[5]:


train[cate_con_or_ord].info()


# ## 处理空值数据

# In[6]:


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


# In[7]:


msno.heatmap(df = train[train_col_missing])


# In[8]:


import missingno as msno

test_col_missing = []
test_null_values = []
for col in values:
    if len(test[col][test[col].isnull()].index):
        test_col_missing.append(col)
        test_null_values.append((col, len(test[col][test[col].isnull()].index)))
for i in sorted(test_null_values, key = lambda x: x[1], reverse = True):
    print(i)
    
msno.matrix(test[test_col_missing], color = (0.2, 0.2, 0.2))
msno.heatmap(test[test_col_missing])


# ## 相关性分析

# In[38]:


import seaborn as sns
plt.figure(figsize = (32, 24))
sns.heatmap(train[values].corr())##calc数据没有相关性


# In[10]:


corr_values = values.copy()
for col in values:
    if 'calc' in col:
        corr_values.remove(col)
plt.figure(figsize = (12, 9))
sns.heatmap(train[corr_values].corr(), annot = False, 
            cmap = sns.diverging_palette(200, 10, as_cmap=True))


# ## 各类特征分析
# ### 二分类特征分析
# #### 训练集二分类空值占比

# In[11]:


import plotly.offline as pltoff
import plotly.graph_objs as go

train_bin_zero_list = []
train_bin_one_list = []
for col in bin_cols:
    temp = train[col].value_counts()
    zero = temp[0]
    one = temp[1]
    train_bin_zero_list.append(zero)
    train_bin_one_list.append(one)

train_trace1 = go.Bar(x = bin_cols,
               y = train_bin_zero_list,
               name = 'zero_counts')
train_trace2 = go.Bar(x = bin_cols,
               y = train_bin_one_list,
               name = 'one_counts')
train_bin_plot_data = [train_trace1, train_trace2]
layout = go.Layout(barmode = 'stack')
fig = go.Figure(layout = layout, data = train_bin_plot_data)
pltoff.iplot(fig, filename = 'stack-bar')
##ps_ind_14与ps_ind_10_bin,ps_ind_11_bin,ps_ind_12_bin,ps_ind_13_bin相关，
#而这4项二分类取零较多，ps_ind_14空值较多


# #### 测试集空值占比

# In[12]:


import plotly.offline as pltoff
import plotly.graph_objs as go

test_bin_zero_list = []
test_bin_one_list = []

for col in bin_cols:
    temp = test[col].value_counts()
    test_bin_zero_list.append(temp[0])
    test_bin_one_list.append(temp[1])

test_trace1 = go.Bar(x = bin_cols,
                    y = test_bin_zero_list,
                    name = 'zero counts')
test_trace2 = go.Bar(x = bin_cols,
                    y = test_bin_one_list,
                    name = 'one counts')
test_bin_plot_data = [test_trace1, test_trace2]
layout = go.Layout(barmode = 'stack')
fig = go.Figure(data = test_bin_plot_data, layout = layout)
pltoff.iplot(fig, filename = 'stack-bar')


# #### 训练集各特征相应的target取值占比

# In[13]:


train_1 = train[train.target == 1]
train_0 = train[train.target == 0]

k = 0
plt.figure(figsize = (32,24))
for col in bin_cols:
    k += 1
    temp0 = train_0[col].value_counts()
    bin_zero_t0 = temp0[0]
    bin_one_t0 = temp0[1]
    temp1 = train_1[col].value_counts()
    bin_zero_t1 = temp1[0]
    bin_one_t1 = temp1[1]
    one_list = (bin_zero_t1 / (bin_zero_t0 + bin_zero_t1), bin_one_t1 / (bin_one_t0 + bin_one_t1))
    plt.subplot(5, 4, k) 
    plt.ylabel('proportion of target = 1')
    plt.title(col)
    plt.bar(('feature: 0', 'feature: 1'), one_list, width = 0.6, label = 'target: 1')


# ### 多分类特征分析

# #### 特征对target的影响

# In[14]:


train_cate = train.fillna(-1).copy()
plt.figure(figsize = (32, 24))
k = 0
for col in cat_cols:
    k += 1
    names = []
    cate_t0 = []
    cate_t1 = []
    one_list = []
    for i in train_cate[col].unique():
        names.append(str(i))
        target = train_cate.loc[train_cate[col] == i, ['target']]
        if len(target['target'].unique()) == 2:
            cate_t0.append(target['target'].value_counts(sort = False)[0])
            cate_t1.append(target['target'].value_counts(sort = False)[1])
        if len(target['target'].unique()) == 1 and target['target'].unique()[0] == 1: 
            cate_t0.append(0)
            cate_t1.append(target['target'].value_counts()[0])
        if len(target['target'].unique()) == 1 and target['target'].unique()[0] == 0:
            cate_t0.append(target['target'].value_counts()[0])
            cate_t1.append(0)
        
    for i in zip(cate_t0, cate_t1):
        one_list.append(i[1] / (i[0] + i[1]))
    plt.subplot(5, 3, k)
    plt.title(col)
    plt.ylabel('target1 counts')
    plt.bar(names, one_list)
##ps_car_08_cat实际上是二分类特征


# #### 训练数据多分类特征取值数量占比

# In[15]:


k = 0
plt.figure(figsize = (32, 24))
for col in cat_cols:
    k += 1
    names = list(train[col].value_counts().index.astype('str'))
    counts = list(train[col].value_counts().values.astype('float64'))
    if col in train_col_missing:
        names.append('NaN')
        counts.append(len(train[col]) - sum(counts))
    plt.subplot(5, 3, k)
    plt.title(col)
    plt.ylabel('feature counts')
    plt.bar(names, counts)


# #### 测试数据多分类特征取值数量占比

# In[16]:


k = 0
plt.figure(figsize = (32, 24))
for col in cat_cols:
    k += 1
    names = list(test[col].value_counts().index.astype('str'))
    counts = list(test[col].value_counts().values.astype('float64'))
    if col in test_col_missing:
        names.append('NaN')
        counts.append(len(test[col]) - sum(counts))
    plt.subplot(5, 3, k)
    plt.title(col)
    plt.ylabel('feature counts')
    plt.bar(names, counts)


# ### 顺序与连续特征分析

# In[17]:


list1 = []
for col in cate_con_or_ord:
    list1.append((col, len(train[col].unique())))
for i in sorted(list1, key = lambda x: x[1], reverse = True):
    print(i)


# #### 顺序或连续训练集数据的直方图

# In[18]:


conti_cols = ['ps_car_13', 'ps_reg_03', 'ps_car_14', 'ps_car_12']
conti_or_ord = cate_con_or_ord.copy()
for i in conti_cols:
    conti_or_ord.remove(i)
    
k = 0
plt.figure(figsize = (32, 24))
for col in conti_or_ord:
    k += 1
    plt.subplot(5, 5, k)
    plt.title(col)
    plt.ylabel('counts')
    plt.hist(train[col].dropna())


# #### 顺序或连续测试集数据的直方图

# In[39]:


conti_cols = ['ps_car_13', 'ps_reg_03', 'ps_car_14', 'ps_car_12']
conti_or_ord = cate_con_or_ord.copy()
for i in conti_cols:
    conti_or_ord.remove(i)
    
k = 0
plt.figure(figsize = (32, 24))
for col in conti_or_ord:
    k += 1
    plt.subplot(5, 5, k)
    plt.title(col)
    plt.ylabel('counts')
    plt.hist(test[col].dropna())


# #### 训练集顺序数据的带KDE曲线图的直方图

# In[19]:


k = 0
plt.figure(figsize = (32, 24))
for col in conti_cols:
    k += 1
    plt.subplot(2, 2, k)
    plt.title(col)
    plt.ylabel('counts')
    sns.distplot(tuple(train[col].dropna()))


# #### 测试集顺序数据的带KDE曲线图的直方图

# In[40]:


k = 0
plt.figure(figsize = (32, 24))
for col in conti_cols:
    k += 1
    plt.subplot(2, 2, k)
    plt.title(col)
    plt.ylabel('counts')
    sns.distplot(tuple(test[col].dropna()))


# #### 顺序或连续数据对target的影响

# In[20]:


##第一种写法
train_rank = train[conti_or_ord].copy()
train_rank = train_rank.dropna()
train_rank['target'] = train['target']
train_rank1 = train_rank[train_rank['target'] == 1]
train_rank0 = train_rank[train_rank['target'] == 0]

k = 0
plt.figure(figsize = (32, 24))
for col in conti_or_ord:
    k += 1
    trace1 = []
    trace0 = []
    names = []
    one_list = []
    for i in train_rank[col].unique():
        names.append(i)
        trace1.append(len(train_rank1[train_rank1[col] == i]))
        trace0.append(len(train_rank0[train_rank0[col] == i]))
    for (x, y) in zip(trace0, trace1):
        one_list.append(y / (x + y))
    plt.subplot(6, 4, k)
    plt.title(col)
    plt.ylabel('counts')
    plt.bar(names, one_list)


# In[21]:


##第二种写法
train_rank = train[conti_or_ord].copy()
train_rank = train_rank.dropna()
train_rank['target'] = train['target']
train_rank1 = train_rank[train_rank['target'] == 1]
train_rank0 = train_rank[train_rank['target'] == 0]

k = 0
plt.figure(figsize = (32, 24))
for col in conti_or_ord:
    k += 1
    one_list = []
    names = train_rank[col].value_counts().index
    trace1 = train_rank1[col].value_counts()[names].values
    trace0 = train_rank0[col].value_counts()[names].values
    for (x, y) in zip(trace0, trace1):
        one_list.append(y / (x + y))
    plt.subplot(6, 4, k)
    plt.title(col)
    plt.ylabel('counts')
    plt.bar(names, one_list)


# #### 对不同target取值的连续变量做KDE密度曲线图

# In[22]:


train_rank_count = train[conti_cols].copy()
train_rank_count['target'] = train['target']
train_rank_count = train_rank_count.dropna()
train_rank0_count = train_rank_count[train_rank_count['target'] == 1]
train_rank1_count = train_rank_count[train_rank_count['target'] == 0]


k = 0
plt.figure(figsize = (32, 24))
for col in conti_cols:
    k += 1
    plt.subplot(2, 2, k)
    sns.kdeplot(train_rank0_count[col])
    sns.kdeplot(train_rank1_count[col])


# #### 有缺失数据的顺序或连续数据对target取值的影响

# In[23]:


train_missing_cont_ord = list(set(train_col_missing).intersection(set(cate_con_or_ord)))
train_rank_missing_cont_ord = train[train_missing_cont_ord].copy()
train_rank_missing_cont_ord['target'] = train['target']
train_rank0_missing_cont_ord = train_rank_missing_cont_ord[
    train_rank_missing_cont_ord['target'] == 0]
train_rank1_missing_cont_ord = train_rank_missing_cont_ord[
    train_rank_missing_cont_ord['target'] == 1]

k = 0
plt.figure(figsize = (32, 24))
for col in train_missing_cont_ord:
    k += 1
    rank0_col_missing = train_rank0_missing_cont_ord[col].isnull().sum()
    rank0_col_unmissing = len(train_rank0_missing_cont_ord) - rank0_col_missing
    rank1_col_missing = train_rank1_missing_cont_ord[col].isnull().sum()
    rank1_col_unmissing = len(train_rank1_missing_cont_ord) - rank1_col_missing
    one_list = (rank1_col_missing / (rank1_col_missing + rank0_col_missing), 
                   rank1_col_unmissing / (rank1_col_unmissing + rank0_col_unmissing))
    plt.subplot(2, 2, k)
    plt.title(col)
    plt.ylabel('counts')
    plt.bar(('missing feature', 'unmissing feature'), one_list)


# # 数据处理
# ## 缺失数据处理

# In[24]:


train_clean = train.copy()
for col in cat_cols:
    train_clean[col].fillna(-1, inplace = True)
train_clean['ps_reg_03_miss'] = np.zeros(len(train))
train_clean.loc[train_clean['ps_reg_03'].isnull(), 'ps_reg_03_miss'] = 1
train_clean['ps_car_14_miss'] = np.zeros(len(train))
train_clean.loc[train_clean['ps_car_14'].isnull(), 'ps_car_14_miss'] = 1

test_clean = test.copy()
for col in cat_cols:
    test_clean[col].fillna(-1, inplace = True)
test_clean['ps_reg_03_miss'] = np.zeros(len(test))
test_clean.loc[test_clean['ps_reg_03'].isnull(), 'ps_reg_03_miss'] = 1
test_clean['ps_car_14_miss'] = np.zeros(len(test))
test_clean.loc[test_clean['ps_car_14'].isnull(), 'ps_car_14_miss'] = 1

for col in ['ps_car_11', 'ps_car_12']:
    train_clean[col].fillna(train_clean[col].median(), inplace = True)
    test_clean[col].fillna(test_clean[col].median(), inplace = True)


# ## 特征工程
# ### 缺失数据特征构建

# In[25]:


train_clean['missing_total_0'] = np.zeros(len(train_clean))
train_clean['missing_total_1'] = np.zeros(len(train_clean))
train_clean['missing_total_2'] = np.zeros(len(train_clean))

train_clean.loc[train_clean['ps_ind_04_cat'] == -1, 'missing_total_0'] += 1
train_clean.loc[train_clean['ps_car_01_cat'] == -1, 'missing_total_0'] += 1
train_clean.loc[train_clean['ps_ind_02_cat'] == -1, 'missing_total_0'] += 1

train_clean.loc[train_clean['ps_car_03_cat'] == -1, 'missing_total_1'] += 1
train_clean.loc[train_clean['ps_car_05_cat'] == -1, 'missing_total_1'] += 1

train_clean.loc[train_clean['ps_ind_05_cat'] == -1, 'missing_total_2'] += 1
train_clean.loc[train_clean['ps_car_07_cat'] == -1, 'missing_total_2'] += 1


# #### 验证新建特征对target的影响

# In[26]:


k = 0
plt.figure(figsize = (32, 24))
temp0 = train_clean[train_clean.target == 0]
temp1 = train_clean[train_clean.target == 1]
for col in ['missing_total_0', 'missing_total_1', 'missing_total_2']:
    k += 1
    names = []
    one_list = []
    for fea in train_clean[col].unique():
        names.append(fea)
        temp0_fea = len(temp0[temp0[col] == fea])
        temp1_fea = len(temp1[temp1[col] == fea])
        one_list.append(temp1_fea / (temp0_fea + temp1_fea))
    plt.subplot(1, 3, k)
    plt.ylabel('counts')
    plt.title(col)
    plt.bar(names, one_list)


# In[27]:


test_clean['missing_total_0'] = np.zeros(len(test_clean))
test_clean['missing_total_1'] = np.zeros(len(test_clean))
test_clean['missing_total_2'] = np.zeros(len(test_clean))

test_clean.loc[test_clean['ps_ind_04_cat'] == -1, 'missing_total_0'] += 1
test_clean.loc[test_clean['ps_car_01_cat'] == -1, 'missing_total_0'] += 1
test_clean.loc[test_clean['ps_ind_02_cat'] == -1, 'missing_total_0'] += 1

test_clean.loc[test_clean['ps_car_03_cat'] == -1, 'missing_total_1'] += 1
test_clean.loc[test_clean['ps_car_05_cat'] == -1, 'missing_total_1'] += 1

test_clean.loc[test_clean['ps_ind_05_cat'] == -1, 'missing_total_2'] += 1
test_clean.loc[test_clean['ps_car_07_cat'] == -1, 'missing_total_2'] += 1


# ### 二分类特征构建

# In[28]:


train_clean['bin_plus_0'] = train_clean['ps_ind_06_bin'] + train_clean['ps_ind_07_bin'] + train_clean['ps_ind_08_bin'] + train_clean['ps_ind_09_bin']
train_clean['bin_plus_1'] = train_clean['ps_ind_16_bin'] + train_clean['ps_ind_17_bin'] + train_clean['ps_ind_18_bin']
train_clean['bin_plus_0'].value_counts()
##必然有一个特征无效，此时需要选择一个特征从train_clean和test_clean数据集中剔除


# In[29]:


train_clean['bin_plus_1'].value_counts()


# In[30]:


train_clean['bin_plus_0'] = train_clean['ps_ind_06_bin'] + train_clean['ps_ind_07_bin'] + train['ps_ind_08_bin']
train_clean['bin_plus_0'].value_counts()


# In[31]:


train_clean.drop('ps_ind_09_bin', axis = 1)
test_clean.drop('ps_ind_09_bin', axis = 1)

test_clean['bin_plus_0'] = test_clean['ps_ind_06_bin'] + test_clean['ps_ind_07_bin'] + test_clean['ps_ind_08_bin'] 
test_clean['bin_plus_1'] = test_clean['ps_ind_16_bin'] + test_clean['ps_ind_17_bin'] + test_clean['ps_ind_18_bin']


# In[32]:


plt.figure(figsize = (32, 24))
sns.heatmap(train_clean[['ps_ind_01', 'ps_ind_02_cat', 'ps_ind_03',
       'ps_ind_04_cat', 'ps_ind_05_cat', 'ps_ind_06_bin', 'ps_ind_07_bin',
       'ps_ind_08_bin', 'ps_ind_10_bin', 'ps_ind_11_bin',
       'ps_ind_12_bin', 'ps_ind_13_bin', 'ps_ind_14', 'ps_ind_15',
       'ps_ind_16_bin', 'ps_ind_17_bin', 'ps_ind_18_bin', 'ps_reg_01',
       'ps_reg_02', 'ps_reg_03', 'ps_car_01_cat', 'ps_car_02_cat',
       'ps_car_03_cat', 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat',
       'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat', 'ps_car_10_cat',
       'ps_car_11_cat', 'ps_car_11', 'ps_car_12', 'ps_car_13', 'ps_car_14',
       'ps_car_15', 'ps_reg_03_miss', 'ps_car_14_miss',
       'missing_total_0', 'missing_total_1', 'missing_total_2', 'bin_plus_0',
       'bin_plus_1']].corr(), annot = False, 
            cmap = sns.diverging_palette(200, 10, as_cmap=True))


# In[33]:


k = 0
temp0 = train_clean[train_clean['target'] == 0]
temp1 = train_clean[train_clean['target'] == 1]

plt.figure(figsize = (32, 24))
for col in ['bin_plus_0', 'bin_plus_1']:
    k += 1
    names = []
    one_list = []
    for fea in temp0[col].unique():
        names.append(fea)
        temp0_fea = len(temp0[temp0[col] == fea])
        temp1_fea = len(temp1[temp1[col] == fea])
        one_list.append(temp1_fea / (temp0_fea + temp1_fea))
    plt.subplot(1, 2, k)
    plt.title(col)
    plt.ylabel('counts')
    plt.bar(names, one_list)


# ### 多分类特征构建

# In[34]:


cat_cols.remove('ps_car_08_cat')
cat_cols.remove('ps_car_11_cat')


# #### 将多分类特征留存一份副本，当成顺序或连续变量，用中值填充到空值

# In[35]:


for col in cat_cols:
    name = '%s_rank'%(col)
    train_clean[name] = train_clean[col].replace(-1, np.nan)
    train_clean[name] = train_clean[name].fillna(train_clean[name].median())
    test_clean[name] = test_clean[col].replace(-1, np.nan)
    test_clean[name] = test_clean[name].fillna(test_clean[name].median())


# In[37]:


#### 把所有原本的多分类特征，根据贝叶斯概率转化为连续型特征
'''
y = train_clean.target
y1 = train_clean[train_clean.target == 1]
for col in cat_cols:
    cat1 = train_clean[col]
    cat2 = train_clean.loc[y == 1, col]
    cat3 = test_clean[col]
    valuec1 = cat1.value_counts()
    valuec2 = cat2.value_counts()
    catdict = {}
    for i in valuec1.index:
        ni = valuec1[i]
        if i in valuec2.index:
            ni1 = valuec2[i]
        else:
            ni1 = 0
        lam = 1 / (np.exp(2 - 0.1 * ni))
        smoothing = lam * ni1 / ni + (1 - lam) * len(y1) / len(y)
        catdict[i] = smoothing
    print(catdict)
    list1 = []
    list2 = []
    for i in cat1:
        list1.append(catdict[i])
    for i in cat3:
        list2.append(catdict[i])
    train_clean[col] = list1
    test_clean[col] = list2
'''


# In[45]:


train_clean.drop(['ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 
                  'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 
                  'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14', 'ps_calc_15_bin',
                  'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin', 
                  'ps_calc_20_bin'], axis = 1)
test_clean.drop(['ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 
                  'ps_calc_06', 'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 
                  'ps_calc_11', 'ps_calc_12', 'ps_calc_13', 'ps_calc_14', 'ps_calc_15_bin',
                  'ps_calc_16_bin', 'ps_calc_17_bin', 'ps_calc_18_bin', 'ps_calc_19_bin', 
                  'ps_calc_20_bin'], axis = 1)


# In[46]:


import pickle
with open('values.pkl', 'wb') as f:
    pickle.dump(train_clean, f)
    pickle.dump(test_clean, f)


# # 创建模型

# ## 载入数据

# In[1]:


import pickle
with open('values.pkl', 'rb') as f:
    train_clean = pickle.load(f)
    test_clean = pickle.load(f)


# In[4]:


import lightgbm as lgb
from sklearn.model_selection import train_test_split
y = train_clean['target']
x = train_clean.drop(['id', 'target'], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size = 0.2, random_state = 100)
lgb_train = lgb.Dataset(x_train, y_train)
lgb_eval = lgb.Dataset(x_test, y_test, reference = lgb_train)


# In[23]:


params = {
    'task': 'train',
    'boosting': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 16,
    'num_iterations': 25,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1
         }
gbm = lgb.train(params, lgb_train, num_boost_round = 100, valid_sets = lgb_train)
y_train_pred = gbm.predict(x_train, pred_leaf = True)


# In[36]:


transformed_training_matrix = np.zeros([len(y_train_pred),
                                        len(y_train_pred[0]) * params['num_leaves']], 
                                       dtype = np.int64)
for i in range(0, len(y_train_pred)):
    temp = np.arange(len(y_train_pred[0])) * params['num_leaves'] + np.array(y_train_pred[i])
    transformed_training_matrix[i][temp] += 1


# In[25]:


y_test_pred = gbm.predict(x_test, pred_leaf = True)
transformed_testing_matrix = np.zeros([len(y_test_pred),
                                        len(y_test_pred[0]) * params['num_leaves']], 
                                       dtype = np.int64)
for i in range(0, len(y_test_pred)):
    temp = np.arange(len(y_test_pred[0])) * params['num_leaves'] + np.array(y_test_pred[i])
    transformed_testing_matrix[i][temp] += 1


# In[29]:


from sklearn.linear_model import LogisticRegression
lm = LogisticRegression(penalty = 'l2', C = 0.05)
lm.fit(X = transformed_training_matrix, y = y_train)
y_pred_test = lm.predict_proba(transformed_testing_matrix)


# In[37]:


from sklearn.metrics import log_loss
log_loss(y_test, y_pred_test)

