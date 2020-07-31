from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

#数据规范化
def DataStandard(Train):
	le=LabelEncoder()
	Train['fueltype']=le.fit_transform(Train['fueltype'])
	Train['aspiration']=le.fit_transform(Train['aspiration'])
	Train['doornumber']=le.fit_transform(Train['doornumber'])
	Train['carbody']=le.fit_transform(Train['carbody'])
	Train['drivewheel']=le.fit_transform(Train['drivewheel'])
	Train['enginelocation']=le.fit_transform(Train['enginelocation'])
	Train['enginetype']=le.fit_transform(Train['enginetype'])
	Train['cylindernumber']=le.fit_transform(Train['cylindernumber'])
	Train['fuelsystem']=le.fit_transform(Train['fuelsystem'])
	train_x=Train[['symboling','fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','wheelbase','carlength','carwidth','carheight','curbweight','enginetype','cylindernumber','enginesize','fuelsystem','boreratio','stroke','compressionratio','horsepower','peakrpm','citympg','highwaympg','price']]
	#规范化到[0,1]空间
	min_max_scaler=preprocessing.MinMaxScaler()
	train_x=min_max_scaler.fit_transform(train_x)
	pd.DataFrame(train_x).to_csv('temp.csv',index=False)
	#print(train_x)
	return(train_x)

#手肘法查看分成几组较为合适
def ElbowKmeans(train_x):
	sse = []
	for k in range(1, 21):
		# kmeans算法
		kmeans = KMeans(n_clusters=k)
		kmeans.fit(train_x)
		# 计算inertia簇内误差平方和
		sse.append(kmeans.inertia_)
	print(sse)
	x = range(1, 21)
	plt.xlabel('K')
	plt.ylabel('SSE')
	plt.plot(x, sse, 'o-')
	plt.show()

def Kmeans(train_x,x):
	kmeans = KMeans(n_clusters=x)
	kmeans.fit(train_x)
	predict_y = kmeans.predict(train_x)
	dataset=pd.read_csv('CarPrice_Assignment.csv', encoding='gbk')
	result = pd.concat((dataset,pd.DataFrame(predict_y)),axis=1)
	result.rename({0:u'聚类结果'},axis=1,inplace=True)
	#按照分类排序
	result=result.sort_values(by='聚类结果',ascending=True)
	print(result)
	result.to_csv("CarType_cluster_result.csv",index=False)

def main():
	pd.options.display.max_columns=100
	data = pd.read_csv('CarPrice_Assignment.csv', encoding='gbk')
	#print(data.head())
	train_x=DataStandard(data)
	#print(data.head())
	ElbowKmeans(train_x)
	#根据总车型数为205个，加上分析手肘法计划分成10个类
	Kmeans(train_x,10)


main()