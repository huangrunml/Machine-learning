# coding=utf-8

from numpy import *
import matplotlib.pyplot as plt
from sklearn import preprocessing





feature_choose=(2,3,4,5,13,25,29,36)
str_feature_choose=(2,3)
num_feature_choose=(4,5,13,25,29,36)
unnormalized_feature=(4,5,22,32)



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

#6.数据预处理总函数




#用于在区间范围内选择一个整数
#输入函数i代表第一个alpha的下标，m是所有alpha的数目
def selectJrand(i,m):
    j=i             #we want to select any J not equal to i
    while (j==i):
        j = int(random.uniform(0,m))
    return j

#用于在数值太大的情况下进行调整
#用于调整大于H小于L的alpha值
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   #linear kernel
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else: raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K


#建立一个全局的数据结构用来保存所有重要的数值
#将数据转移到一个结构来实现，省去手工输入麻烦
class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler,kTup):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler      #最大循环次数
        #m表示数据的行数
        self.m = shape(dataMatIn)[0]
        #初始化为m行1列的全零矩阵
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        #eCache的第一列表示是否有效的标志位，第二列是实际E值
        self.eCache = mat(zeros((self.m,2))) #first column is valid flag
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

#该函数用来计算E值并返回
#k可以表示i或者j的数值
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X*oS.X[k,:].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])    #数据点代入alpha和b后的值与真实标签的差值表示Ek
    return Ek


#该函数用于选择第二个alpha值或者说是内循环的alpha值，保证每次循环采用最大步长
def selectJ(i, oS, Ei):  # this is the second choice -heurstic, and calcs Ej
    maxK = -1;
    maxDeltaE = 0;
    Ej = 0
    oS.eCache[i] = [1, Ei]  # 将eCache的第一列数据也就是标志位设置为有效，接下来选择合适的alpha使得步长也就是Ei-Ej最大
    #构建出一个非零表
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]     #将eCache的标志位到处存入到validECacheList中
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:  #通过eCache的有效标志位作为判断依据进行循环找到满足步长最大化的alpha
            if k == i: continue     #如果k等于i，跳出循环，避免浪费时间
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k;
                maxDeltaE = deltaE;
                Ej = Ek
        return maxK, Ej
    else:   #这种情况是在第一次循环过程中eCache没有有效值，就采用遍历数据集的方式
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

#将计算误差值缓存到alpha值进行优化之后会用到
def updateEk(oS, k):  # after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

#Platt SMO算法的内循环部分，确定第二个alpha的值
def innerL(i, oS):
    Ei = calcEk(oS, i)      #计算第i组数据的误差
    #如果计算得到的误差在允许范围之内时
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        #selectJ函数用来选择第二个alpha值来保证步长得到最大
        j,Ej = selectJ(i, oS, Ei)
        #将alphai与alphaj的旧值存储
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        # 调整H与L的数值，便于用来控制alpha的范围
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print "L==H"; return 0     #如果L=H时，则返回0
        # eta代表的是alphas[j]的最优修改量，如果eta大于等于0则需要退出for循环的当前迭代过程
        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        if eta >= 0: print "eta>=0"; return 0
        # 计算alphaj的数值，给alphas[j]赋新值，同时控制alphas[j]的范围在H与L之间
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        # 将计算误差进行缓存，便于后面用到
        updateEk(oS, j) #added this for the Ecache
        # 然后检查alphas[j]是否有轻微改变，如改变很小，就进行下一次循环
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print "j not moving enough"; return 0
        # 将alpha[i]与alpha[j]同时进行改变，改变的数值保持一致，但是改变的方向相反
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        #将Ek存入缓存
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        # 在完成alpha[i]与alpha[j]的优化之后，给这两个alpha设置一个常数项b
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

#选择第一个Alpha值的外循环
#五个输入参数对应表示数据集，类别标签，常数C，容错率，退出前最大循环次数
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('rbf',10)):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler,kTup)
    iter = 0        #该变量用于存储是在没有任何alpha改变的情况下遍历数据集的次数
    entireSet = True
    # 该参数用于记录alpha是否已进行优化，每次循环先将其设为0
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):       #开始的for循环在数据集上遍历任意可能的alpha
                alphaPairsChanged += innerL(i,oS)       #用innerL来选择第二个alpha
                print "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]    #遍历非边界值
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)       #遍历非边界值来确定第二个alpha
                print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True
        print "iteration number: %d" % iter
    return oS.b,oS.alphas

def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

def testRbf(train_data,train_label,test_data,test_label,k1):
    dataArr = train_data
    labelArr = train_label
    #print "labelArr:",labelArr
    # min_max_scaler = preprocessing.MinMaxScaler()
    # normalized_feature = min_max_scaler.fit_transform(dataArr)   #最大最小归一化
    normalized_feature = preprocessing.scale(dataArr)  # 标准化特征
    # max_abs_scaler = preprocessing.MaxAbsScaler()
    # normalized_feature = max_abs_scaler.fit_transform(normalized_feature) #绝对值归一化
    dataArr = preprocessing.normalize(normalized_feature, norm="l2")  # 正则化样本数据
    # normalized_feature=np.around(normalized_feature,8)
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, k1)#C=200 important
    print "b:",b
    print "alphas:",alphas.T
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] #求出支持向量
    labelSV = labelMat[svInd];#支持向量的标记
    print "there are %d Support Vectors" % shape(sVs)[0]
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],k1)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print "the training error rate is: %f" % (float(errorCount)/m)
    dataArrTest = test_data
    labelArrTest = test_label
    #print "labelArrTest:",labelArrTest
    # min_max_scaler = preprocessing.MinMaxScaler()
    # normalized_feature = min_max_scaler.fit_transform(normalized_feature)   #最大最小归一化
    normalized_feature = preprocessing.scale(dataArrTest)  # 标准化特征
    # max_abs_scaler = preprocessing.MaxAbsScaler()
    # normalized_feature = max_abs_scaler.fit_transform(normalized_feature) #绝对值归一化
    dataArrTest = preprocessing.normalize(normalized_feature, norm="l2")  # 正则化样本数据
    # normalized_feature=np.around(normalized_feature,8)
    errorCount = 0
    datMat=mat(dataArrTest); labelMat = mat(labelArrTest).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],k1)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArrTest[i]): errorCount += 1
    print "the test error rate is: %f" % (float(errorCount)/m)



if __name__ == "__main__":
    # 加载数据
    with open(r"C:\Users\Administrator\Desktop\train.txt", "r") as f0:
        num_feature = genfromtxt(f0, dtype=float, delimiter=",", usecols=(range(33)))
        print "训练数据:",num_feature
        train_data=num_feature
        f0.seek(0)
        train_label = genfromtxt(f0,dtype=float,delimiter=",",usecols=-1)
        print train_label
        j=0
        for i in train_label:
          if i==0:
            train_label[j]=+1
          else:
            train_label[j]=-1
          j=j+1
        print "train_label:",train_label
    with open(r"C:\Users\Administrator\Desktop\test.txt", "r") as f1:
        num_feature1 = genfromtxt(f1, dtype=float, delimiter=",", usecols=(range(33)))
        print "测试数据:",num_feature1
        test_data=num_feature1
        f1.seek(0)
        test_label = genfromtxt(f1,dtype=float,delimiter=",",usecols=-1)
        print test_label
        j=0
        for i in test_label:
          if i==0:
            test_label[j]=+1
          else:
            test_label[j]=-1
          j=j+1
        print "test_label:",test_label
    testRbf(train_data,train_label,test_data,test_label,k1=('rbf',1))#支持向量机
