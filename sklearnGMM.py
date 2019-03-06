#!/usr/bin/python
#coding:utf8
import numpy as np
import os
import sys
import random
import logging
import math
import copy
from sklearn import preprocessing
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn import metrics



feature_to_use_number=[5,3,6,4,30,29,33,34,35,38,12,39,25,23,26]
num_feature_choose=(4,5,6,9,22,29,30)     #cfs gen10 0.47936
#num_feature_choose=(4,5,11,13,15,29,36)     #0.36
#num_feature_choose=(29,10,21,4,5,30,9)      #GR 8 0.439
#num_feature_choose=(0,4,5,6,9,10,15,22,29,30)  #cfs gen 0.46533
#num_feature_choose=(4,24,29,30)                #cfs bestfirst

extension_feature_choose=(4,5,29,22,28,33,32,34)

#符号特征预处理
def AddFeature(num_feature,str_feature):
    le = preprocessing.LabelEncoder()
    le.fit(str_feature)
    b = le.transform(str_feature)
    all_feature = np.c_[num_feature, b]
    return all_feature

#one-hot编码
def Onehot(str_feature):
    le = preprocessing.LabelEncoder()
    str = le.fit_transform(str_feature)
    one_hot = preprocessing.OneHotEncoder(sparse=False).fit(str.reshape(-1, 1))
    one_hot = one_hot.transform(str.reshape(-1, 1))
    # print "one_hot:",one_hot
    return one_hot

#特征提取
def extract_features(f):
    str_feature1 = np.genfromtxt(f, dtype=str, delimiter=",", usecols=1, comments="@")
    f.seek(0)
    str_feature2 = np.genfromtxt(f, dtype=str, delimiter=",", usecols=2, comments="@")
    f.seek(0)
    str_feature3 = np.genfromtxt(f, dtype=str, delimiter=",", usecols=3, comments="@")
    f.seek(0)
    type = np.genfromtxt(f, dtype=str, delimiter=",", usecols=-1, comments="@")
    f.seek(0)
    #print "str1:", str_feature1
    #print "str2:", str_feature2
    #print "type:", type
    f.seek(0)
    num_feature1 = np.genfromtxt(f, dtype=float, delimiter=",", usecols=num_feature_choose, comments="@")
    f.seek(0)
    num_feature2 = np.genfromtxt(f, dtype=float, delimiter=",", usecols=extension_feature_choose,comments="@")
    return str_feature1,str_feature2,str_feature3,num_feature1,num_feature2,type

#特征归一化等处理
def normalized_feature(all_feature):
    min_max_scaler = preprocessing.MinMaxScaler()
    normalized_feature = min_max_scaler.fit_transform(all_feature)  # 最大最小归一化
    #print "1:", normalized_feature

    #normalized_feature = preprocessing.scale(normalized_feature)  # 标准化特征
    #print "2:", normalized_feature

    #max_abs_scaler = preprocessing.MaxAbsScaler()
    #normalized_feature = max_abs_scaler.fit_transform(normalized_feature) #绝对值归一化

    #normalized_feature = preprocessing.normalize(normalized_feature, norm="l2")  # 正则化样本数据
    #normalized_feature=np.around(normalized_feature,8)
    return normalized_feature

#特征归一化等处理2
def normalized_feature1(all_feature):
    min_max_scaler = preprocessing.MinMaxScaler()
    normalized_feature = min_max_scaler.fit_transform(all_feature)  # 最大最小归一化
    #print "1:", normalized_feature

    normalized_feature = preprocessing.scale(normalized_feature)  # 标准化特征
    #print "2:", normalized_feature

    #max_abs_scaler = preprocessing.MaxAbsScaler()
    #normalized_feature = max_abs_scaler.fit_transform(normalized_feature) #绝对值归一化

    normalized_feature = preprocessing.normalize(normalized_feature, norm="l2")  # 正则化样本数据
    #normalized_feature=np.around(normalized_feature,8)
    return normalized_feature

#距离计算公式
def Minkowski(x, y):
  x = np.array(x)
  y = np.array(y)
  output = np.square((abs(x - y) ** 0.5).sum())
  return output
def Euclidean(x,y):
  x = np.array(x)
  y = np.array(y)
  output = (np.square(x - y).sum())**0.5
  return output
def Manhattan(x,y):
  x = np.array(x)
  y = np.array(y)
  output = sum(abs(x - y))
  return output



def distEclud(vecA, vecB):
    '''
    输入：向量A和B
    输出：A和B间的欧式距离
    '''
    return np.sqrt(sum(np.power(vecA - vecB, 2)))

def newCent(L):
    '''
    输入：有标签数据集L
    输出：根据L确定初始聚类中心
    '''
    centroids = []
    label_list = np.unique(L[:,-1])
    for i in label_list:
        L_i = L[(L[:,-1])==i]
        cent_i = np.mean(L_i,0)
        centroids.append(cent_i[:-1])
    return np.array(centroids)

def semi_kMeans(L, U, distMeas=distEclud, initial_centriod=newCent):
    '''
    输入：有标签数据集L（最后一列为类别标签）、无标签数据集U（无类别标签）
    输出：聚类结果
    '''
    dataSet = np.vstack((L[:,:-1],U))#合并L和U
    label_list = np.unique(L[:,-1])
    k = len(label_list)           #L中类别个数
    m = np.shape(dataSet)[0]

    clusterAssment = np.zeros(m)#初始化样本的分配
    centroids = initial_centriod(L)#确定初始聚类中心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#将每个样本分配给最近的聚类中心
            minDist = np.inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i] != minIndex: clusterChanged = True
            clusterAssment[i] = minIndex
    return clusterAssment


'''
标记扩展
'''

def MarkExtension1(sample_index,sample_set_type,extension,extension_sample_set,attack_type,radius):
  exsample_index = copy.deepcopy(sample_index)
  exsample_set_type = copy.deepcopy(sample_set_type)
  c=[]
  for index,value in extension:
      distance=[]
      for detector in extension_sample_set:
          d=Euclidean(detector,value)
          distance.append(d)
      c.append(min(distance))
      if min(distance)<=radius:
          exsample_index.append(index)
          a=[]
          b=np.where(np.array(distance)==min(distance))
          for i in b[0]:
            j=exsample_set_type[i]
            a.append(j)
          exsample_set_type.append(Counter(a).most_common()[0][0])
      else:
          continue
  n=0
  F=0
  for i in exsample_index:
      if i in exsample_index and i not in sample_index:
          n +=1
          if attack_type[i] != exsample_set_type[exsample_index.index(i)]:
              F +=1
          else:
              continue
      else:
          continue
  F=F/float(n)
  return exsample_index,exsample_set_type,n,F

def MarkExtension(sample_index,sample_set_type,extension,extension_sample_set,attack_type,radius):
  exsample_index = copy.deepcopy(sample_index)
  exsample_set_type = copy.deepcopy(sample_set_type)
  for index,value in extension:
      distance=[]
      for detector in extension_sample_set:
          d=Euclidean(detector,value)
          distance.append(d)
      c=np.where(np.array(distance)<=radius)
      if c[0]!=[]:
          exsample_index.append(index)
          a=(np.array(exsample_set_type))[c].tolist()
          exsample_set_type.append(Counter(a).most_common()[0][0])
      else:
          continue
  n=0
  F=0
  for i in exsample_index:
      if i in exsample_index and i not in sample_index:
          n +=1
          if attack_type[i] != exsample_set_type[exsample_index.index(i)]:
              F +=1
          else:
              continue
      else:
          continue
  F=F/float(n)
  return exsample_index,exsample_set_type,n,F
'''
为每簇添加标记
'''


def AddSymbol(lines,exsample_index, exsample_set_type,attacki,attackiindex,attack_pre_type,sample_index,f):
  attack_set=zip(attackiindex,attacki)
  F=0
  for i,j in attack_set:
    if i in exsample_index:
      if exsample_set_type[exsample_index.index(i)]==j:
        if j==attack_pre_type:
          lines[i].append("+1")
        else:
          lines[i].append("-1")
      else:
        F=F+1
    else:
      continue
  for i in attackiindex:
    lines[i]=",".join(lines[i])
    f.write(lines[i])
    f.write("\n")
  a=0
  for i in attackiindex:
    if i in exsample_index and i not in sample_index:
      a=a+1
    else:
      continue
  print "a:",a
  print "F:",F
  F=F/float(a)
  return F



if __name__ == '__main__':
    with open(r"C:\Users\Administrator\Desktop\yichangtestnou2r.arff", "r") as f1:
        test_str_feature1,test_str_feature2,test_str_feature3,test_num_feature,extensiontest_num_feature,type1=extract_features(f1)
    with open(r"C:\Users\Administrator\Desktop\yichangtestnou2r.arff", "r") as f2:
        lines0 = f2.readlines()
        j = 0
        for line in lines0:
            lines0[j] = line.strip().split(",")
            j = j + 1
    with open(r"C:\Users\Administrator\Desktop\samplenormal.txt", "r") as f3:
        sample_str_feature1, sample_str_feature2,sample_str_feature3,sample_num_feature, extensionsample_num_feature,sample_type=extract_features(f3)
    with open(r"C:\Users\Administrator\Desktop\samplenormal.txt", "r") as f4:
        lines = f4.readlines()
        j = 0
        for line in lines:
            lines[j] = line.strip().split(",")
            j = j + 1

    num_feature = np.vstack((sample_num_feature, test_num_feature))
    extension_num_feature = np.vstack((extensionsample_num_feature, extensiontest_num_feature))

    str_feature1 = np.hstack((sample_str_feature1, test_str_feature1))
    print "str1:", str_feature1
    str_feature2 = np.hstack((sample_str_feature2, test_str_feature2))
    print "str2:", str_feature2
    str_feature3 = np.hstack((sample_str_feature3, test_str_feature3))
    print "str3:", str_feature3


    extension_feature = AddFeature(extension_num_feature,str_feature3)
    extension_feature = AddFeature(extension_feature,str_feature2)
    extension_feature = AddFeature(extension_feature,str_feature1)
    extension_feature = normalized_feature(extension_feature)
    print "标记扩展特征:",extension_feature

    '''one_hot1 = Onehot(str_feature1)
    all_feature = np.hstack((num_feature, one_hot1))
    one_hot2 = Onehot(str_feature2)
    all_feature = np.hstack((all_feature, one_hot2))
    one_hot3 = Onehot(str_feature3)
    all_feature = np.hstack((all_feature, one_hot3))
    all_feature = normalized_feature(all_feature)
    pca = PCA(n_components=10)
    data1 = pca.fit_transform(all_feature)
    #print "比率:", pca.explained_variance_ratio_
    #print "方差:", pca.explained_variance_
'''
    #data = num_feature
    data = AddFeature(num_feature,str_feature3)
    data = AddFeature(data,str_feature2)
    data = AddFeature(data,str_feature1)
    data = normalized_feature(data)
    type = np.hstack((sample_type,type1))
    print "总的数据特征集：", data
    lines.extend(lines0)                          #这儿可能有bug,多了一行？
    print "总的数据：", len(lines)


    '''gmm = GaussianMixture(n_components=4, n_init=10, reg_covar=1e-5, max_iter=1000, random_state=10)
    gmm = gmm.fit(data)
    pre1 = gmm.predict(data)
    print pre1
    print metrics.adjusted_rand_score(type, pre1)'''
    kmeans=KMeans(n_clusters=4,max_iter=1000,n_init=10,random_state=10)
    kmeans=kmeans.fit(data)
    pre=kmeans.predict(data)
    print metrics.adjusted_rand_score(type,pre)

    # 标记数据确认，总测试数据确认和索引获取
    sample_set = data[range(len(sample_type))]
    extension_sample_set = extension_feature[range(len(sample_type))]
    sample_set_type = sample_type.tolist()
    sample_index = range(len(sample_set_type))
    print "采样的标记：", sample_set
    print "采样的标记目录：", sample_index
    print "采样的标记数量：", len(sample_set)
    print "采样的攻击类型：", sample_set_type

    extension_set_index = range(len(sample_index), len(data))
    extension_set = data[range(len(sample_index), len(data))]
    mark_extension_set = extension_feature[range(len(sample_index), len(data))]
    extension_set_type = type1.tolist()
    attack_type = copy.deepcopy(sample_set_type)  # 类型列表形式
    attack_type.extend(extension_set_type)
    print "attack_type:", attack_type
    attack_type_array = np.array(attack_type)  # 类型数组形式
    print "未标记的数据目录：", extension_set_index
    print "未标记的数据：", extension_set
    extension = zip(extension_set_index, extension_set)
    extension1 = zip(extension_set_index, mark_extension_set)
    exsample_index, exsample_set_type,n,F = MarkExtension(sample_index, sample_set_type, extension1,extension_sample_set,attack_type, 0.05)
    print "扩展后标记目录：", exsample_index
    print "扩展后标记数量：", len(exsample_index)
    print "扩展后标记：", exsample_set_type
    #print "距离:", distance
    print "扩展标记数量:",n
    print "标记扩展错误率:",F


    # 预测攻击数组初始化
    attack0index = []
    attack1index = []
    attack2index = []
    attack3index = []
    # attack4index=[]

    attack0 = []
    attack1 = []
    attack2 = []
    attack3 = []
    # attack4=[]
    attack_pre = []

    # 预测结果
    pre_type = pre.tolist()
    for i in list(set(pre_type)):
        print ("预测攻击%s:" % i), pre_type.count(i)
    for i in range(len(pre_type)):
        if pre_type[i] == 0:
            attack0index.append(i)
        elif pre_type[i] == 1:
            attack1index.append(i)
        elif pre_type[i] == 2:
            attack2index.append(i)
        elif pre_type[i] == 3:
            attack3index.append(i)
    attack0 = attack_type_array[attack0index].tolist()
    attack1 = attack_type_array[attack1index].tolist()
    attack2 = attack_type_array[attack2index].tolist()
    attack3 = attack_type_array[attack3index].tolist()
    # attack4=attack_type_array[attack4index].tolist()
    attack_pre = [attack0, attack1, attack2, attack3]

    # 第一种预测结果：每簇的真实类型最多的
    j = 0
    for i in attack_pre:
        print ("预测攻击%d:" % j), Counter(i).most_common()
        print ("预测攻击%d:" % j), Counter(i).most_common()[0][0]
        j = j + 1

    # 第二种预测结果：每簇的标记最多的
    attack_pre_index = [attack0index, attack1index, attack2index, attack3index]
    attack_pre_type = []
    k = 0
    for attackindex in attack_pre_index:
        a = []
        b = []
        c = []
        for i in attackindex:
            if i in exsample_index:
                a.append(i)
                b.append(exsample_set_type[exsample_index.index(i)])
                if attack_type[i] == exsample_set_type[exsample_index.index(i)]:
                    c.append(exsample_set_type[exsample_index.index(i)])
                else:
                    continue
            else:
                continue
        print ("预测类型%d各标记数量:" % k), Counter(b).most_common()
        attack_pre_type.append(Counter(b).most_common()[0][0])
        k = k + 1
    print "按最多标记预测攻击类型:", attack_pre_type

    # 真实结果
    print "真实攻击类型:""\n", attack_type_array
    for i in list(set(attack_type)):
        print ("真实攻击%s:" % i), attack_type.count(i)


    '''with open(r"C:\Users\Administrator\Desktop\biaoji.txt", "w") as f9:
        for i in exsample_index:
            lines[i] = ",".join(lines[i])
            f9.write(lines[i])
            f9.write("\n")
    '''
    with open(r"C:\Users\Administrator\Desktop\zongshuju.txt", "r") as f9:        #直接使用lines会出错，我也不知道为什么，可能是extend函数的问题
        lines1 = f9.readlines()
        j = 0
        for line in lines1:
            lines1[j] = line.strip().split(",")
            j = j + 1
    with open(r"C:\Users\Administrator\Desktop\EM0.txt", "w") as f5:
        F0 = AddSymbol(lines1, exsample_index, exsample_set_type, attack0, attack0index, attack_pre_type[0],sample_index, f5)
    with open(r"C:\Users\Administrator\Desktop\EM1.txt", "w") as f6:
        F1 = AddSymbol(lines1, exsample_index, exsample_set_type, attack1, attack1index, attack_pre_type[1],sample_index, f6)
    with open(r"C:\Users\Administrator\Desktop\EM2.txt", "w") as f7:
        F2 = AddSymbol(lines1, exsample_index, exsample_set_type, attack2, attack2index, attack_pre_type[2],sample_index, f7)
    with open(r"C:\Users\Administrator\Desktop\EM3.txt", "w") as f8:
        F3 = AddSymbol(lines1, exsample_index, exsample_set_type, attack3, attack3index, attack_pre_type[3],sample_index, f8)
    # with open(r"C:\Users\Administrator\Desktop\EM4.txt", "w") as f9:
    # F4=AddSymbol(lines,exsample_index,exsample_set_type,attack4,attack4index,attack_pre_type[4],sample_index,f9)
    print "F0:", F0
    print "F1:", F1
    print "F2:", F2
    print "F3:", F3
    # print "F4:",F4

