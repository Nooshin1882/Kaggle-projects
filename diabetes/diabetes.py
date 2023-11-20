import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,KNNImputer
from sklearn.preprocessing import StandardScaler,MinMaxScaler,PowerTransformer
from sklearn.metrics import f1_score,accuracy_score,silhouette_score,classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from math import nan
import scipy.stats as spstats

df=pd.read_csv("C:/Users/Nooshin/Desktop/data science/kaggle/diabetes.csv")
df.info()

df=df.drop(df[df.Glucose==0].index)
df=df.drop(df[df.Pregnancies>13].index)
df=df.drop(df[(df.BloodPressure>20)&(df.BloodPressure<38)&(df.BloodPressure>106)].index)
df=df.drop(df[df.SkinThickness>70].index)
df=df.drop(df[df.Insulin>310].index)
df=df.drop(df[df.BMI>50].index)
df=df.drop(df[df.DiabetesPedigreeFunction>1.2].index)
df=df.drop(df[df.Age>66].index)


df.BloodPressure.replace({0:nan},inplace=True)
df.SkinThickness.replace({0:nan},inplace=True)
df.BMI.replace({0:nan},inplace=True)

######################drop high correlations
# df=df.drop("Insulin",axis=1)
# df=df.drop("Pregnancies",axis=1)
#######################################transformation
# Glucose=np.array(df.Glucose)
# l,opt_lambda=spstats.boxcox(Glucose)
# df["Glucose"]=spstats.boxcox(df.Glucose,lmbda=opt_lambda)

# BloodPressure=np.array(df.BloodPressure)
# l,opt_lambda=spstats.boxcox(BloodPressure)
# df["BloodPressure"]=spstats.boxcox(df.BloodPressure,lmbda=opt_lambda)

# SkinThickness=np.array(df.SkinThickness)
# l,opt_lambda=spstats.boxcox(SkinThickness)
# df["SkinThickness"]=spstats.boxcox(df.SkinThickness,lmbda=opt_lambda)

# BMI=np.array(df.BMI)
# l,opt_lambda=spstats.boxcox(BMI)
# df["BMI"]=spstats.boxcox(df.BMI,lmbda=opt_lambda)

# DiabetesPedigreeFunction=np.array(df.DiabetesPedigreeFunction)
# l,opt_lambda=spstats.boxcox(DiabetesPedigreeFunction)
# df["DiabetesPedigreeFunction"]=spstats.boxcox(df.DiabetesPedigreeFunction,lmbda=opt_lambda)

# Age=np.array(df.Age)
# l,opt_lambda=spstats.boxcox(Age)
# df["Age"]=spstats.boxcox(df.Age,lmbda=opt_lambda)


#################################################

target=df.Outcome
data=df.drop("Outcome",axis=1)

PT=PowerTransformer(method="yeo-johnson")               #better than the above method
PT.fit(data)
data=PT.transform(data)

KI=KNNImputer()
KI.fit(data)
data=KI.transform(data)

x_train,x_test,y_train,y_test=train_test_split(data,target,train_size=0.8,shuffle=False)

SS=StandardScaler()
SS.fit(x_train)
x_train=SS.transform(x_train)
x_test=SS.transform(x_test)

# ###########################################################################    DecisionTree 62%
LR=DecisionTreeClassifier(criterion="entropy")           #entropy is better
LR.fit(x_train,y_train)
y_p=LR.predict(x_test)
print(accuracy_score(y_p,y_test))
print(f1_score(y_p,y_test))
###########################################################################     KNN  61%

# a=[]
# f=[]
# for i in range(2,16):
#     KNN=KNeighborsClassifier(n_neighbors=i)              #6 is the best value for neighbors
#     KNN.fit(x_train,y_train)
#     y_p=KNN.predict(x_test)
#     a.append(accuracy_score(y_p,y_test))
#     f.append(f1_score(y_p,y_test))

# plt.plot(range(2,16),a)
# plt.show()
# plt.plot(range(2,16),f)
# plt.show()

# KNN=KNeighborsClassifier(n_neighbors=12)
# KNN.fit(x_train,y_train)
# y_p=KNN.predict(x_test)
# print(accuracy_score(y_p,y_test))
# print(f1_score(y_p,y_test))
# print(classification_report(y_p,y_test))
###########################################################################     LogisticRegression  68%
# SS=StandardScaler()
# SS.fit(x_train)
# x_train=SS.transform(x_train)
# x_test=SS.transform(x_test)

# Cl=[]
# Clf=[]
# for i in np.arange(0.01,1,0.01):
#     LR=LogisticRegression(C=i,solver="liblinear")
#     LR.fit(x_train,y_train)
#     y_p=LR.predict(x_test)
#     Cl.append(accuracy_score(y_p,y_test))
#     Clf.append(f1_score(y_p,y_test))

# plt.plot(np.arange(0.01,1,0.01),Cl)
# plt.show()

# plt.plot(np.arange(0.01,1,0.01),Clf)
# plt.show()

# LR=LogisticRegression(C=0.5,solver="liblinear")
# LR.fit(x_train,y_train)
# y_p=LR.predict(x_test)
# print(accuracy_score(y_p,y_test))
# print(f1_score(y_p,y_test))
# print(confusion_matrix(y_p,y_test))
# print(classification_report(y_p,y_test))
###########################################################################     SVM   (The Best)   70%
# SS=StandardScaler()
# SS.fit(x_train)
# x_train=SS.transform(x_train)
# x_test=SS.transform(x_test)

# k=["linear","poly","rbf","sigmoid"]
# a=[]
# f=[]
# for i in k:
#     SVM=SVC(kernel=i)
#     SVM.fit(x_train,y_train)
#     y_p=SVM.predict(x_test)
#     a.append(accuracy_score(y_p,y_test))
#     f.append(f1_score(y_p,y_test))
#     # print(confusion_matrix(y_p,y_test))
#     # print(classification_report(y_p,y_test))

# plt.plot(np.arange(4),a)
# plt.show()

# plt.plot(np.arange(4),f)
# plt.show()

# SVM=SVC(kernel="rbf")
# SVM.fit(x_train,y_train)
# y_p=SVM.predict(x_test)
# print(accuracy_score(y_p,y_test))
# print(f1_score(y_p,y_test))