# coding=utf-8

from numpy import *
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import SVC




feature_choose=(2,3,4,5,29,28,32,33,34,37,11,38,24,22,25)
str_feature_choose=(2,3)
#num_feature_choose=(4,5,13,25,29,36)
#num_feature_choose=(4,5,29,28,32,33,34,37,11,38,24,22,25)      #原始
num_feature_choose=(4,5,29,22,28,33,32,34,39,23,40,26,0,35,27,24,37,36)       #ig 20
#num_feature_choose=(4,5,29,22,28,33,32,34,39,23,40,26,0)                   #ig 15




'''
特征预处理

'''

#1.符号特征提取
def StrExtract(data,index):
    str=[]
    for i in data:
       str.append(i[index])
    return str

#2.数字特征提取
def NumExtract(data,index):
    num_feature=[]
    for i in data:
        num=[]
        for j in index:
            num.append(i[j])
        #print num
        num_feature.append(num)
    return num_feature

#3.符号特征添加至数字特征
def AddFeature(num_feature,str_feature):
    le = preprocessing.LabelEncoder()
    le.fit(str_feature)
    b = le.transform(str_feature)
    all_feature = c_[num_feature, b]
    return all_feature

#4.将特征转换为数组形式
def StrtoArray(data):
    a=[]
    for lines in data:
        a.append(map(float,lines))
    array_feature=array(a)
    return array_feature


#5.第二层准确率，数据提取
def LoadData(f):
    j = 0
    lines = f.readlines()
    for line in lines:
        lines[j] = line.strip().split(",")
        j = j + 1
    for line in lines:
        if line[-1]=="+1":
            type=line[-2]
        else:
            continue
    a=[]
    train_data=[]
    test_data=[]
    for line in lines:
        a.append(line[-1])
        if line[-1]=="-1" or line[-1]=="+1":
            train_data.append(line)
        else:
            test_data.append(line)
    F1=(a.count("+1")+a.count(type))/float(len(a))
    F2=(a.count(type)+len(train_data))/float(len(a))
    return lines,a,type,train_data,test_data,F1,F2


#5.2测试数据提取处理
def Testload(f):
    j = 0
    lines = f.readlines()
    for line in lines:
        lines[j] = line.strip().split(",")
        j = j + 1
    return lines

#6.数据预处理总函数

def loadDataSet(train_data):
    str1 = StrExtract(train_data, 2)
    str2 = StrExtract(train_data, 3)
    str3 = StrExtract(train_data,1)
    #print str1
    num_feature= NumExtract(train_data, num_feature_choose)
    #print num_feature
    all_feature= AddFeature(num_feature, str2)
    all_feature= AddFeature(all_feature,str1)
    all_feature= AddFeature(all_feature,str3)
    array_feature = StrtoArray(all_feature)
    #print array_feature
    return array_feature


def SVM(train_data,test_data,type,kernel,c):
    dataArr = loadDataSet(train_data)
    label = [i[-1] for i in train_data]
    labelArr = array(map(int, label))
    #print "labelArr:",labelArr

    min_max_scaler = preprocessing.MinMaxScaler()
    dataArr = min_max_scaler.fit_transform(dataArr)   #最大最小归一化
    #dataArr = preprocessing.scale(dataArr)  # 标准化特征
    #max_abs_scaler = preprocessing.MaxAbsScaler()
    #dataArr = max_abs_scaler.fit_transform(dataArr) #绝对值归一化
    #dataArr = preprocessing.normalize(dataArr, norm="l2")  # 正则化样本数据
    # normalized_feature=np.around(normalized_feature,8)

    svc=SVC(C=1,kernel=kernel,gamma=c,tol=0.0001,random_state=100)
    svc.fit(dataArr,labelArr)

    #导入测试数据
    dataArrTest = loadDataSet(test_data)
    labelTest = [i[-1] for i in test_data]
    j=0
    for i in labelTest:
        if i==type:
            labelTest[j]=+1
        else:
            labelTest[j]=-1
        j=j+1
    labelArrTest = map(int, labelTest)
    #print "labelArrTest:",labelArrTest

    min_max_scaler = preprocessing.MinMaxScaler()
    dataArrTest=min_max_scaler.fit_transform(dataArrTest)   #最大最小归一化
    #dataArrTest = preprocessing.scale(dataArrTest)  # 标准化特征
    #max_abs_scaler = preprocessing.MaxAbsScaler()
    #dataArrTest = max_abs_scaler.fit_transform(dataArrTest) #绝对值归一化
    #dataArrTest = preprocessing.normalize(dataArrTest, norm="l2")  # 正则化样本数据

    pre=svc.predict(dataArrTest)
    pre=pre.tolist()
    print "support_number:",svc.n_support_
    #print "support_index:",svc.support_
    #print "support:",svc.support_vectors_
    errorcount=len(labelArrTest)-(map(cmp,labelArrTest,pre).count(0))
    F=float(errorcount) / len(labelArrTest)
    print "the test error rate is: %f" % F
    false_positive=0
    false_negative=0
    true_positive=0
    true_negative=0
    true_pre=zip(labelArrTest,pre)
    for i,j in true_pre:
        if i==+1 and j==+1:
            true_positive +=1
        elif i==-1 and j==-1:
            true_negative +=1
        elif i==-1 and j==+1:
            false_positive +=1
        elif i==+1 and j==-1:
            false_negative +=1
    try:
        DR=(float(true_positive)/(true_positive+false_negative))
    except ZeroDivisionError:
        DR=0.0
    try:
        PR=(float(true_positive)/(true_positive+false_positive))
    except ZeroDivisionError:
        PR=0.0
    try:
        AR=(float(true_positive+true_negative)/(true_positive+false_positive+true_negative+false_negative))
    except ZeroDivisionError:
        AR=0.0
    print ("actual %s number=" % type),labelArrTest.count(+1)
    print ("number classified as %s =" % type),pre.count(+1)
    print "Recall(Re)=",DR
    print "Precision(Pr)=",PR
    print "Accuracy(Ar)=",AR
    print "FP=",false_positive,"FN=",false_negative,"TP=",true_positive,"TN=",true_negative


if __name__ == "__main__":
    # 加载数据
    with open(r"C:\Users\Administrator\Desktop\probe1.txt", "r") as f0:
        lines0,attack_type0,type0,train_data0, test_data0, T01, T02 = LoadData(f0)
        print "T01:",T01,"T02:",T02
    with open(r"C:\Users\Administrator\Desktop\probe2.txt", "r") as f1:
        lines1,attack_type1,type1,train_data1, test_data1, T11, T12 = LoadData(f1)
        print "T11:",T11, "T12:",T12
    with open(r"C:\Users\Administrator\Desktop\probe3.txt", "r") as f2:
        lines2,attack_type2,type2,train_data2, test_data2, T21, T22 = LoadData(f2)
        print "T21:",T21, "T22:",T22
    with open(r"C:\Users\Administrator\Desktop\probe4.txt", "r") as f3:
        lines3,attack_type3,type3, train_data3, test_data3, T31, T32 = LoadData(f3)
        print "T31:",T31, "T32:",T32
    #with open(r"C:\Users\Administrator\Desktop\EM4.txt", "r") as f4:
        #lines4,attack_type4,type4, train_data4, test_data4, T41, T42 = LoadData(f4)
        #print "T41:",T41, "T42:",T42
    with open(r"C:\Users\Administrator\Desktop\alltest.arff", "r") as f5:
        testdata=Testload(f5)
        print "Testdata:",testdata[0]
    SVM(train_data2,testdata,type2,"rbf",1)