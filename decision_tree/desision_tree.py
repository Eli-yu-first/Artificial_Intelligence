import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import tree

film_data = open('tree_data.csv','rt')
reader = csv.reader(film_data)

headers = next(reader)
# print(headers)

feature_list = [] # 特征值
result_list = [] # 结果（最后一列）

for row in reader:
    result_list.append(row[-1])

    # 去掉首位两列，特征集只保留'type','country' , 'gross'，将headers与row压缩成一个字典
    feature_list.append(dict(zip(headers[1:-1],row[1:-1])))

vec = DictVectorizer() # 将 dict类型的list数据转换成 numpy array（特征向量化）
dummyX = vec.fit_transform(feature_list).toarray() # 做扁平化处理
dummyY = preprocessing.LabelBinarizer().fit_transform(result_list)
# print(dummyX)
# country  |  gnoss | type
# 0,0,0,0  | 0,0  | 0,0,0
# print(dummyY)

clf = tree.DecisionTreeClassifier(criterion='entropy',random_state=0)
clf = clf.fit(dummyX,dummyY)
print('clf=',str(clf))


# 数据可视化 1
import pydotplus
dot_data = tree.export_graphviz(clf,
                                feature_names=vec.get_feature_names(),
                                filled=True,
                                rounded=True, # 用圆角框
                                special_characters=True,
                                out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('film.pdf')

##开始预测
A = ([[e,e,0,1,0,1,0,1,0]])#日本(4)-低票房(2)-动画片(3)
# B = ([[e，0，1，0，0，1，0，1，0]])#法国(4)-低票房(2)-动画片(3))
# C = ([[1，0，0，0，1，0， 1，0，0]]）#美国(4)-高票房(2)-动作片(3)
predict_result = clf.predict(A)

# 数据可视化 2
# import graphviz
# from sklearn.tree import export_graphviz
# model = clf
# dot_data = tree.export_graphviz(model,
#                                 feature_names=vec.get_feature_names(),
#                                 filled=True,
#                                 rounded=True,
#                                 special_characters=True,
#                                 out_file=None)
# graph = graphviz.Source(dot_data) # 将决策树模型可视化
# graph.render('ddd.pdf') # 生成决策树可视化PDF文件

