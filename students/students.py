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


# df=pd.read_csv("C:/Users/Nooshin/Desktop/Data Science/kaggle/students.csv")


# df=df.drop("NationalITy",axis=1)
# #
# df=df.drop(df[(df.PlaceofBirth=="Morocco")&(df.PlaceofBirth=="venzuela")].index)

# df=make_one_hot(df,"PlaceofBirth")
# df=make_one_hot(df,"SectionID")
# df=make_one_hot(df,"Topic")
# #
# df.gender.replace({"M":1,"F":0},inplace=True)
# df.StageID.replace({"HighSchool":3,"MiddleSchool":2,"lowerlevel":1},inplace=True)
# df.Semester.replace({"F":1,"S":0},inplace=True)
# df.Relation.replace({"Father":1,"Mum":0},inplace=True)
# df.ParentAnsweringSurvey.replace({"No":0,"Yes":1},inplace=True)
# # df=df.drop("ParentAnsweringSurvey",axis=1)       #high correlation with ParentschoolSatisfaction    #not good
# df.ParentschoolSatisfaction.replace({"Good":0,"Bad":1},inplace=True)
# df.StudentAbsenceDays.replace({"Above-7":1,"Under-7":0},inplace=True)
# df.Class.replace({"H":3,"M":2,"L":1},inplace=True)
# # df.GradeID.replace({"G-02":2,"G-08":8,"G-07":7,"G-04":4,"G-06":6,"G-11":11,"G-12":12,"G-09":9,"G-10":10,"G-05":5},inplace=True) #high correlated with stageid
# df=df.drop("GradeID",axis=1)

# df=df.drop("VisITedResources",axis=1)  #high correlation with AnnouncementsView
# df=df.drop("raisedhands",axis=1)       #high correlation with AnnouncementsView
# #

# # df_p=df[["raisedhands","VisITedResources","AnnouncementsView","Discussion"]]
# df_p=df[["AnnouncementsView","Discussion"]]


# from sklearn.preprocessing import PowerTransformer
# PT=PowerTransformer(method="yeo-johnson")
# PT.fit(df_p)
# df_p=PT.transform(df_p)
# df_p=pd.DataFrame(df_p)

# ############################ what happend to df
# # plt.subplot(2, 1, 1)
# # plt.hist(df.raisedhands)
# # plt.subplot(2, 1, 2)
# # plt.hist(df_p.iloc[:,0])
# # plt.show()


# # plt.subplot(2, 1, 1)
# # plt.hist(df.VisITedResources)
# # plt.subplot(2, 1, 2)
# # plt.hist(df_p.iloc[:,1])
# # plt.show()


# # plt.subplot(2, 1, 1)
# # plt.hist(df.AnnouncementsView)
# # plt.subplot(2, 1, 2)
# # plt.hist(df_p.iloc[:,2])
# # plt.show()


# # plt.subplot(2, 1, 1)
# # plt.hist(df.Discussion)
# # plt.subplot(2, 1, 2)
# # plt.hist(df_p.iloc[:,3])
# # plt.show()
# #############################

# # df=df.drop(["raisedhands","VisITedResources","AnnouncementsView","Discussion"],axis=1)
# df=df.drop(["AnnouncementsView","Discussion"],axis=1)
# df=pd.concat([df,df_p],axis=1)

# target=df.Class
# data=df.drop("Class",axis=1)

# x_train,x_test,y_train,y_test=train_test_split(data,target,train_size=0.8,random_state=4)


# ###############################################################Decision Tree               #64%   >    62%  >  71% elimination of high correlanetd coulumns
# # LR=DecisionTreeClassifier(criterion="gini")               
# # LR.fit(x_train,y_train)
# # y_p=LR.predict(x_test)
# # print("Decision Tree")
# # print(accuracy_score(y_p,y_test))
# # print(f1_score(y_p,y_test,average="weighted"))


# ############################################################################KNN                   #63%   >  70%
# # a=[]
# # f=[]
# # for i in range(2,14):
# #     KNN=KNeighborsClassifier(n_neighbors=i)              #6 is the best value for neighbors
# #     KNN.fit(x_train,y_train)
# #     y_p=KNN.predict(x_test)
# #     a.append(accuracy_score(y_p,y_test))
# #     f.append(f1_score(y_p,y_test,average="weighted"))

# # plt.plot(range(2,14),a)
# # plt.show()
# # plt.plot(range(2,14),f)
# # plt.show()

# # KNN=KNeighborsClassifier(n_neighbors=7)
# # KNN.fit(x_train,y_train)
# # y_p=KNN.predict(x_test)
# # print(accuracy_score(y_p,y_test))
# # print(f1_score(y_p,y_test,average="macro"))
# # print(classification_report(y_p,y_test))


# ######################################################## LogisticRegression    #74%   >   71%
# # SS=StandardScaler()
# # SS.fit(x_train)
# # x_train=SS.transform(x_train)
# # x_test=SS.transform(x_test)

# # Cl=[]
# # for i in np.arange(0.01,1,0.01):
# #     LR=LogisticRegression(C=i,solver="newton-cg")
# #     LR.fit(x_train,y_train)
# #     y_p=LR.predict(x_test)
# #     Cl.append(accuracy_score(y_p,y_test))
# #     # print(f1_score(y_p,y_test))

# # plt.plot(np.arange(0.01,1,0.01),Cl)
# # plt.show()

# # LR=LogisticRegression(C=0.3,solver="newton-cg")
# # LR.fit(x_train,y_train)
# # y_p=LR.predict(x_test)
# # print(accuracy_score(y_p,y_test))
# # print(f1_score(y_p,y_test,average="weighted"))
# # print(confusion_matrix(y_p,y_test))
# # print(classification_report(y_p,y_test))
# ########################################################## SVM         #62%     >     71%    >73% elimination of high correlanetd coulumns
# # SS=StandardScaler()
# # SS.fit(x_train)
# # x_train=SS.transform(x_train)
# # x_test=SS.transform(x_test)

# SVM=SVC(kernel="rbf")
# SVM.fit(x_train,y_train)
# y_p=SVM.predict(x_test)
# print(accuracy_score(y_p,y_test))
# print(f1_score(y_p,y_test,average="weighted"))
# print(confusion_matrix(y_p,y_test))
# print(classification_report(y_p,y_test))
#########################################################################################################################################################


#RFECV

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import get_dummies
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,KNNImputer
from sklearn.metrics import r2_score,mean_absolute_error,accuracy_score,f1_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler,PolynomialFeatures,MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV



df=pd.read_csv("C:/Users/Nooshin/Desktop/Data Science/kaggle/students.csv")

target=df.Class
data=df.drop("Class",axis=1)

SVM=SVC(kernel="rbf")
rfecv=RFECV(estimator=SVM,step=1,cv=13,scoring="f1_score")
rfecv.fit(data,target)

