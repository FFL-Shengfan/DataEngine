import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#导入表格，进行数据处理
def ChartImport():
	dataset=pd.read_csv('./订单表.csv',encoding='gbk')
	df=dataset[['订单日期','客户ID','产品名称']]
	#print(df)
	#同一客户在同一日期买入作为一个订单
	df['Index']=df['客户ID'].map(str)+'-'+df['订单日期']
	group=df[['Index','产品名称']]
	group=group.groupby(['Index'])['产品名称'].apply(lambda x:x.str.cat(sep='/')).reset_index()
	hot_encoded_df=group.drop('产品名称',1).join(group.产品名称.str.get_dummies(sep='/'))
	hot_encoded_df.set_index(['Index'],inplace=True)
	print(hot_encoded_df.head())
	return(hot_encoded_df)

def Cal_Apriori(dataset):	
	itemsets = apriori(dataset,use_colnames=True, min_support=0.1)
	itemsets = itemsets.sort_values(by="support" , ascending=False) 
	print('-'*20, '频繁项集', '-'*20)
	print(itemsets)
	rules =  association_rules(itemsets, metric='lift', min_threshold=1)
	rules = rules.sort_values(by="lift" , ascending=False) 
	print('-'*20, '关联规则', '-'*20)
	print(rules)


def main():
	pd.options.display.max_columns=100
	dataset=ChartImport()
	Cal_Apriori(dataset)


main()

