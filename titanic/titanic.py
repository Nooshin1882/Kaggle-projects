# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from pandas import get_dummies
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer,KNNImputer
# from sklearn.metrics import r2_score,mean_absolute_error,accuracy_score,f1_score,confusion_matrix,classification_report
# from sklearn.preprocessing import StandardScaler,PolynomialFeatures,MinMaxScaler
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC


# def make_one_hot(data,field):
#     one_hot=pd.get_dummies(data[field],prefix=field)
#     one_hotsum = one_hot.apply(np.sum, axis=1)
#     for i in range(data.shape[0]):
#         if one_hotsum.iloc[i]==0:
#             for j in range(one_hot.shape[1]):
#                 one_hot.iat[i,j]=np.nan
#     data=data.drop(field,axis=1)
#     return data.join(one_hot)


# # این کد برای این برنامه ست ولی اصلیتش همینه//باید جای وان هات را پیدا کرد و این برنامه را نوشت
#                     برنامه تبدیل بیشترین تخمین به عدد 1 و بقیه صفر     
# def func1(data1,a,b):
#     for index,row in data1.iterrows():            
#         m=int(data1.iloc[index,a:b].idxmax())
#         for j in range(a,b):
#             if j==m:
#                 data1.iat[index,j]=1
#             else:
#                 data1.iat[index,j]=0
#     return data1


# df=pd.read_csv("C:/Users/Nooshin/Desktop/sql/kaggle/titanic_train.csv")
# # df.info()

# df=df.drop("cabin",axis=1)
# df=df.drop("ticket",axis=1)

# df=df.replace({"female":1,"male":0})

# df=make_one_hot(df,"embarked")

# df=df.drop(df[(df.sibsp==5)&(df.sibsp==8)].index)
# df=df.drop(df[(df.parch==5)&(df.parch==3)&(df.parch==4)&(df.parch==6)].index)

# bin_ranges = [0,10,20,30,40,50,60,70,80,90]
# bin_names = [1,2,3,4,5,6,7,8,9]

# df["age_range"]=pd.cut(df['age'],bins=bin_ranges,labels=bin_names)

# df=df.drop(["name","age"],axis=1)
# # df['name'] = df['name'].map(lambda x: x.split(',')[1].split('.')[0].strip())      #Not good
# # df=make_one_hot(df,"name")

# # df=df.drop(df[df.age>70].index)      #Not good

# # df["sib+pa"]=df.parch+df.sibsp       #Not good
# # df=df.drop("parch",axis=1)
# # df=df.drop("sibsp",axis=1)



# target=df.survived
# data=df.drop("survived",axis=1)

# KI=KNNImputer()
# KI.fit(data)
# data=KI.transform(data)

# x_train,x_test,y_train,y_test=train_test_split(data,target,train_size=0.8,random_state=4)

# ###########################################################################    DecisionTree
# LR=DecisionTreeClassifier(criterion="entropy")           #entropy is better
# LR.fit(x_train,y_train)
# y_p=LR.predict(x_test)
# print(accuracy_score(y_p,y_test))
# print(f1_score(y_p,y_test))
###########################################################################     KNN

# SS=StandardScaler()
# SS.fit(x_train)
# x_train=SS.transform(x_train)
# x_test=SS.transform(x_test)

# a=[]
# f=[]
# for i in range(2,14):
#     KNN=KNeighborsClassifier(n_neighbors=i)              #6 is the best value for neighbors
#     KNN.fit(x_train,y_train)
#     y_p=KNN.predict(x_test)
#     a.append(accuracy_score(y_p,y_test))
#     f.append(f1_score(y_p,y_test))

# plt.plot(range(2,14),a)
# plt.show()
# plt.plot(range(2,14),f)
# plt.show()

# KNN=KNeighborsClassifier(n_neighbors=12)
# KNN.fit(x_train,y_train)
# y_p=KNN.predict(x_test)
# print(accuracy_score(y_p,y_test))
# print(f1_score(y_p,y_test))
# print(classification_report(y_p,y_test))
###########################################################################     LogisticRegression
# SS=StandardScaler()
# SS.fit(x_train)
# x_train=SS.transform(x_train)
# x_test=SS.transform(x_test)

# Cl=[]
# for i in np.arange(0.01,1,0.01):
#     LR=LogisticRegression(C=i,solver="newton-cg")
#     LR.fit(x_train,y_train)
#     y_p=LR.predict(x_test)
#     Cl.append(accuracy_score(y_p,y_test))
#     # print(f1_score(y_p,y_test))

# plt.plot(np.arange(0.01,1,0.01),Cl)
# plt.show()

# LR=LogisticRegression(C=0.2,solver="newton-cg")
# LR.fit(x_train,y_train)
# y_p=LR.predict(x_test)
# print(accuracy_score(y_p,y_test))
# print(f1_score(y_p,y_test))
# print(confusion_matrix(y_p,y_test))
# print(classification_report(y_p,y_test))
###########################################################################     SVM   (The Best)   86%
# SS=StandardScaler()
# SS.fit(x_train)
# x_train=SS.transform(x_train)
# x_test=SS.transform(x_test)

# SVM=SVC(kernel="rbf")
# SVM.fit(x_train,y_train)
# y_p=SVM.predict(x_test)
# print(accuracy_score(y_p,y_test))
# print(f1_score(y_p,y_test))
# print(confusion_matrix(y_p,y_test))
# print(classification_report(y_p,y_test))
#######################################################################################################       RFECV
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from pandas import get_dummies
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer,KNNImputer
# from sklearn.metrics import r2_score,mean_absolute_error,accuracy_score,f1_score,confusion_matrix,classification_report
# from sklearn.preprocessing import StandardScaler,PolynomialFeatures,MinMaxScaler
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_selection import RFECV
# from sklearn.svm import SVC


# def make_one_hot(data,field):
#     one_hot=pd.get_dummies(data[field],prefix=field)
#     one_hotsum = one_hot.apply(np.sum, axis=1)
#     for i in range(data.shape[0]):
#         if one_hotsum.iloc[i]==0:
#             for j in range(one_hot.shape[1]):
#                 one_hot.iat[i,j]=np.nan
#     data=data.drop(field,axis=1)
#     return data.join(one_hot)


# df=pd.read_csv("C:/Users/Nooshin/Desktop/sql/kaggle/titanic_train.csv")
# # df.info()

# df=df.drop("cabin",axis=1)
# df=df.drop("ticket",axis=1)

# df=df.replace({"female":1,"male":0})

# df=make_one_hot(df,"embarked")

# df=df.drop(df[(df.sibsp==5)&(df.sibsp==8)].index)
# df=df.drop(df[(df.parch==5)&(df.parch==3)&(df.parch==4)&(df.parch==6)].index)

# bin_ranges = [0,10,20,30,40,50,60,70,80,90]
# bin_names = [1,2,3,4,5,6,7,8,9]

# df["age_range"]=pd.cut(df['age'],bins=bin_ranges,labels=bin_names)

# df=df.drop(["name","age"],axis=1)
# # df['name'] = df['name'].map(lambda x: x.split(',')[1].split('.')[0].strip())      #Not good
# # df=make_one_hot(df,"name")

# # df=df.drop(df[df.age>70].index)      #Not good

# # df["sib+pa"]=df.parch+df.sibsp       #Not good
# # df=df.drop("parch",axis=1)
# # df=df.drop("sibsp",axis=1)



# target=df.survived
# data=df.drop("survived",axis=1)
# data1=data

# KI=KNNImputer()
# KI.fit(data)
# data=KI.transform(data)

# x_train,x_test,y_train,y_test=train_test_split(data,target,train_size=0.8,random_state=4)

###########################################################################    DecisionTree
# LR=DecisionTreeClassifier(criterion="gini")           #entropy is better
# rfecv = RFECV(estimator=LR, step=1, cv=10, scoring='accuracy')
# rfecv.fit(data, target)
# print('Optimal number of features: {}'.format(rfecv.n_features_))
# print(rfecv.support_)
# data1.info()

# print(rfecv.estimator_.feature_importances_)

# plt.figure(figsize=(11, 5))
# plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold')
# plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
# plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)
# plt.show()
#############################################################################