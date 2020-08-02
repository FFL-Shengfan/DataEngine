from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from lxml import etree
from nltk.tokenize import word_tokenize


def create_word_cloud(f):
	print('根据词频，开始生成词云!')
	#f = remove_stop_words(f)
	cut_text = word_tokenize(f)
	#print(cut_text)
	cut_text = " ".join(cut_text)
	print(cut_text)
	wc = WordCloud(
		max_words=100,
		width=2000,
		height=1200,
    )

	wordcloud = wc.generate(cut_text)
	# 写词云图片
	wordcloud.to_file("wordcloud.jpg")
	# 显示词云文件
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.show()


def main():
	data=pd.read_csv('./Market_Basket_Optimisation.csv',header=None)
	all_word=''
	#print(data)
	for row in range(0,data.shape[0]):
		for column in range(0,20):
			if str(data.values[row,column])!='nan':
				#print(str(data.values[row,column]))
				all_word=all_word+''.join (' '+str(data.values[row,column]))
	print(all_word)
	#remove_stop_words(all_word)
	create_word_cloud(all_word)

main()