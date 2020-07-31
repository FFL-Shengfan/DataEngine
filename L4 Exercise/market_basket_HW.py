import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from efficient_apriori import apriori


# header=None，不将第一行作为head
dataset = pd.read_csv('./Market_Basket_Optimisation.csv', header = None) 
# shape为(7501,20)
print(dataset.shape)

# 将数据存放到transactions中
transactions = []
for i in range(0, dataset.shape[0]):
    temp = []
    for j in range(0, 20):
        if str(dataset.values[i, j]) != 'nan':
           temp.append(str(dataset.values[i, j]))
    transactions.append(temp)
#print(transactions)
# 挖掘频繁项集和频繁规则
itemsets, rules = apriori(transactions, min_support=0.02,  min_confidence=0.4)
print("频繁项集：", itemsets)
print("关联规则：", rules)

#mlxtend
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#将dataset转换成一列
df = dataset[dataset.columns[1:]].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1
)
#print(df)
#one-hot编码
basket_hot_encoded=df.str.get_dummies(sep=',')
#print(basket_hot_encoded)
itemsets=apriori(basket_hot_encoded,use_colnames=True,min_support=0.02)
itemsets=itemsets.sort_values(by="support",ascending=False)
print('-'*20,'频繁项集','-'*20)
print(itemsets)
#最大显示列数
pd.options.display.max_columns=100
#提升度
rules=association_rules(itemsets,metric='lift',min_threshold=2)
rules=rules.sort_values(by='lift',ascending=False)
print('-'*20,'关联规则','-'*20)
print(rules)
