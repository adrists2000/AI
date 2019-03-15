import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""--------------------- Using Scikit-learn ML algorithm -------------------"""

from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

"""---------------------- Coding ML algorithm by myself --------------------"""

class Adaline(object):
    def __init__(self,n_iter,eta,random_state):
        self.n_iter=n_iter
        self.eta=eta
        self.random_state=random_state
               
    def fit(self,train_set,train_value):   # using gradient descent       
        t_size=train_set.shape[1]
        np.random.seed(self.random_state)
        self.weight_ = np.random.normal(loc=0,scale=0.015,size=t_size+1)
        self.cost_list = []
        
        for i in range(self.n_iter):
#            print("i=",i,"weight=",self.weight_)
            net_input = 1*self.weight_[0] + np.dot(train_set,self.weight_[1:] )            
            activation_fun = net_input
            error_array = train_value - activation_fun
            self.weight_[0]  += self.eta * error_array.sum()
            self.weight_[1:] += self.eta * ( np.dot(train_set.T,error_array) )
            cost_fun = 0.5 * ( (error_array**2).sum() )
            self.cost_list.append(cost_fun)
#        print("(end) activation_fun=",activation_fun)
#        print("cost_list=",self.cost_list)
        return self
        
    def predict(self,predict_set):
        activation_fun= 1*self.weight_[0] + np.dot(predict_set,self.weight_[1:])
        return np.where(activation_fun>=0.5, 1, 0)
            
#----------------------------------------------------------------------------------

class Logistic_Regression(object):
    def __init__(self,n_iter,eta,random_state):
        self.n_iter=n_iter
        self.eta=eta
        self.random_state=random_state
               
    def fit(self,train_set,train_value):      # using gradient descent    
        t_size=train_set.shape[1]
        np.random.seed(self.random_state)
        self.weight_ = np.random.normal(loc=0,scale=0.01,size=t_size+1)
        self.cost_list = []
        
        for i in range(self.n_iter):
#            print("i=",i,"weight=",self.weight_)            
            activation_fun = self.activation_func(train_set)
            error_array = train_value - activation_fun
            self.weight_[0]  += self.eta * error_array.sum()
            self.weight_[1:] += self.eta * ( np.dot(train_set.T,error_array) )
            cost_fun = self.cost_func(train_value,activation_fun)
            self.cost_list.append(cost_fun)    
#        print("(end) activation_fun=",activation_fun)            
#        print("cost_list=",self.cost_list)
        return self

    def activation_func(self,input_set):
        net_input = 1*self.weight_[0] + np.dot(input_set,self.weight_[1:] )            
        activation_fun = 1.0 / (1.0 + np.exp(-np.clip(net_input,-240,240)))
        return activation_fun        

    def cost_func(self,train_value,activation_fun):
        cost_fun = ( -train_value*np.log(activation_fun) - (1-train_value)*np.log(1-activation_fun) ).sum()
        return cost_fun
        
    def predict(self,predict_set):
        activation_fun= self.activation_func(predict_set)
        return np.where(activation_fun>=0.5, 1, 0)

#--------------------------------------------------------------------------------

def accuracy(predict_v,test_v,name):
    sum=0.0
    for i in range(test_v.size):
        if predict_v[i] == test_v[i]:
            sum+=1.0
    print("\n"+name)
    print("正確率 =",(100*sum)/test_v.size,"%")

def scattering_figure(x_set,y_set,obj,resolution=0.02):
    color_list=['blue','red','green','yellow','cyan']
    marker_list=['s','x','o','v','^']
    cmap=ListedColormap(color_list[:len(np.unique(y_set))])
    if len(x_set[1,:])==2:
        min_x1,max_x1 = x_set[:,0].min()-1 , x_set[:,0].max()+1
        min_x2,max_x2 = x_set[:,1].min()-1 , x_set[:,1].max()+1
        xgrid,ygrid =np.meshgrid(np.arange(min_x1,max_x1,resolution),np.arange(min_x2,max_x2,resolution))    
        value=obj.predict( np.array([xgrid.ravel(),ygrid.ravel()]).T )
        value=value.reshape(xgrid.shape)
        plt.contourf(xgrid,ygrid,value,alpha=0.4,cmap=cmap)      #cmap , ListedColormap  alpha透明度
        plt.axis( [xgrid.min(),xgrid.max(),ygrid.min(),ygrid.max()] )
        plt.xlim(xgrid.min(),xgrid.max())
        plt.ylim(ygrid.min(),ygrid.max())

    for i,y_value in enumerate(np.unique(y_set)):
        plt.scatter(x=x_set[y_set==y_value,0] , y=x_set[y_set==y_value,1],alpha=0.8,
                    c=color_list[i],marker=marker_list[i],label=y_value,edgecolor='black')
    
    print("取特徵1,2作圖")
    plt.legend(loc='upper left')
    plt.xlabel('feature1')
    plt.ylabel('feature2') 
    plt.show()
    
#--------------------------------------------------------------------------------

def Main_Classification(aa="0",t_num="0",f_num="0",select="c",algorithm=[]):

    print("\n分類演算法程式:")
    while not aa in [str(i) for i in range(1,4)]:        
        print("\n請選擇分類測試資料:\n1.iris(花的種類) \n2.wine(酒的種類) \n3.cancer(是否罹癌)")
        aa=input("請輸入(1~3): ")
        if not aa in [str(i) for i in range(1,4)]:
            print("輸入錯誤")
    if aa=="1" or aa=="2":
        while not t_num in [str(i) for i in range(2,4)]:
            t_num=input("請輸入分類的數目(2~3): ")
            if not t_num in [str(i) for i in range(2,4)]:
                print("輸入錯誤")
    elif aa=="3": t_num="2"

    print("\n若特徵數取2時,可顯示分類切割圖")
    if aa=="1":
        while not f_num in [str(i) for i in range(1,5)]:
            f_num=input("請輸入特徵數目(1~4):")
            if not f_num in [str(i) for i in range(1,5)]:
                print("輸入錯誤")
        t_num2 = lambda t_num: 100 if t_num=="2" else 150
        iris=datasets.load_iris()
        total_set=iris.data[:t_num2(t_num),:int(f_num)]
        total_value=iris.target[:t_num2(t_num)]

    elif aa=="2":
        while not f_num in [str(i) for i in range(1,15)]:
            f_num=input("請輸入特徵數目(1~14):")
            if not f_num in [str(i) for i in range(1,15)]:
                print("輸入錯誤")
        t_num2 = lambda t_num: 130 if t_num=="2" else 178  
        wine=datasets.load_wine()
        total_set=wine.data[:t_num2(t_num),:int(f_num)]
        total_value=wine.target[:t_num2(t_num)]

    elif aa=="3":
        while not f_num in [str(i) for i in range(1,31)]:
            f_num=input("請輸入特徵數目(1~30):") 
            if not f_num in [str(i) for i in range(1,31)]:
                print("輸入錯誤")
        print(f_num)
        cancer=datasets.load_breast_cancer()
        total_set=cancer.data[:425,:int(f_num)]
        total_value=cancer.target[:425]

    train_set,test_set,train_value,test_value =  \
      train_test_split(total_set,total_value,test_size=0.3, random_state=2, stratify=total_value)  # 參數調整處!!
#    train_set=total_set
#    train_value=total_value

    while not select in ["y","n"] :
        select=input("是否使用特徵縮放(y/n): ")
    if select=="y":
        sc=StandardScaler()
        sc.fit(train_set)
        train_set=sc.transform(train_set)
        test_set=sc.transform(test_set)        

#    print("train_set=\n",total_set)
#    print("train_value=\n",total_value)
#    print("train_set=\n",train_set)
#    print("train_value=\n",train_value)
    print("test_set=\n",test_set)
    print("test_value=\n",test_value)
    #print("number of train label:", np.bincount(train_value))
    #print("number of predict label:", np.bincount(test_value))

    if algorithm ==[]:
        while True:
            print("\n請選擇分類演算法: ")
            print("1. Adaline(自寫)(適用二元分類) ")
            print("2. 邏輯迴歸(自寫)(適用二元分類) ")
            print("3. Perceptron ")
            print("4. 邏輯迴歸 ")
            print("5. 支援向量機 ")
            print("6. 決策樹 ")
            print("7. KNN ")
            print("8. 全選 ")
            print("目前已選擇:",algorithm)
            al=input("請選擇[1~8], 或輸入[0]開始:")
            if al in [str(i) for i in range(1,8)]: algorithm.append(int(al))
            elif al =="8": algorithm=[1,2,3,4,5,6,7]  
            elif al=="0": break
            else: print("\n輸入錯誤!")
               
    for i in algorithm:
    #-----------------------------Using my ML code----------------------------
        if i==1:
            ADA=Adaline(n_iter=1000,eta=0.0001,random_state=3)            # 參數調整處!!
            ADA.fit(train_set,train_value)
            predict_value=ADA.predict(test_set)
            obj=ADA
            accuracy(predict_value,test_value,"Adaline(自寫)-分類:")
        elif i==2:
            LR=Logistic_Regression(n_iter=500,eta=0.001,random_state=3)   # 參數調整處!!
            LR.fit(train_set,train_value)
            predict_value=LR.predict(test_set)
            obj=LR
            accuracy(predict_value,test_value,"邏輯迴歸(自寫)-分類:")
    #--------------------Comparing with scikit-learn  ML algorithm-------------
        elif i==3:
            perceptron=Perceptron(n_iter=100,eta0=0.01,random_state=3)    # 參數調整處!!
            perceptron.fit(train_set,train_value)
            predict_value=perceptron.predict(test_set)
            obj=perceptron
            accuracy(predict_value,test_value,"Perceptron-分類:")
        elif i==4:
            lr=LogisticRegression(C=100.0,random_state=3)                 # 參數調整處!!
            lr.fit(train_set,train_value)
            predict_value=lr.predict(test_set)
            obj=lr
            accuracy(predict_value,test_value,"邏輯迴歸-分類:")
        elif i==5:
            svm=SVC(kernel='linear',C=1.1,random_state=3)                 # 參數調整處!!
            svm.fit(train_set,train_value)
            predict_value=svm.predict(test_set)
            obj=svm
            accuracy(predict_value,test_value,"支援向量機-分類:")
        elif i==6:
            dtree=DecisionTreeClassifier(criterion='gini',max_depth=6,random_state=3)   # 參數調整處!!
            dtree.fit(train_set,train_value)
            predict_value=dtree.predict(test_set)
            obj=dtree
            accuracy(predict_value,test_value,"決策樹-分類:")
        elif i==7:
            knn=KNeighborsClassifier(n_neighbors=7,p=2,metric='minkowski') # 參數調整處!!
            knn.fit(train_set,train_value)
            predict_value=knn.predict(test_set)
            obj=knn
            accuracy(predict_value,test_value,"KNN-分類:")
            
        scattering_figure(test_set,predict_value,obj)

#------------------------------------------------------------------------------
        
if __name__=="__main__":

    Main_Classification()
#    Main_Classification(aa="3",t_num="2",f_num="14",select="y",algorithm=[1,2,3,4,5,6,7])
