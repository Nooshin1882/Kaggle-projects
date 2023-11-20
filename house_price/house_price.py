

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import get_dummies
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer,KNNImputer
from sklearn.metrics import r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler,PolynomialFeatures,MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor


def make_one_hot(data,field):
    one_hot=pd.get_dummies(data[field],prefix=field)
    one_hotsum = one_hot.apply(np.sum, axis=1)
    for i in range(data.shape[0]):
        if one_hotsum.iloc[i]==0:
            for j in range(one_hot.shape[1]):
                one_hot.iat[i,j]=np.nan
    data=data.drop(field,axis=1)
    return data.join(one_hot)


df=pd.read_csv("C:/Users/Nooshin/Desktop/sql/kaggle/house_price_train.csv")
# df.info()

df=df.drop("Id",axis=1)
df=df.drop("Alley",axis=1)
df=df.drop("PoolQC",axis=1)
df=df.drop("Fence",axis=1)
df=df.drop("MiscFeature",axis=1)
df=df.drop("FireplaceQu",axis=1)
df=df.drop(df[df.LotConfig=="FR3"].index)
df=df.drop(df[df.Neighborhood=="Blueste"].index)
df=df.drop(df[(df.Condition1=="RRNn")&(df.Condition1=="RRNe")].index)
df=df.drop("Condition2",axis=1)
df=df.drop(df[(df.RoofStyle=="Mansard")&(df.RoofStyle=="Shed")].index)
df=df.drop(df[(df.Exterior1st=="BrkComm")&(df.Exterior1st=="Stone")&(df.Exterior1st=="AsphShn")&(df.Exterior1st=="ImStucc")&(df.Exterior1st=="CBlock")].index)
df=df.drop(df[(df.Exterior2nd=="Stone")&(df.Exterior2nd=="AsphShn")&(df.Exterior2nd=="CBlock")&(df.Exterior2nd=="Other")].index)
df=df.drop(df[(df.ExterCond=="Ex")&(df.ExterCond=="Po")].index)
df=df.drop(df[(df.Foundation=="Stone")&(df.Foundation=="Wood")].index)
df=df.drop(df[df.BsmtCond=="Po"].index)
df=df.drop(df[(df.Heating=="Grav")&(df.Heating=="Wall")&(df.Heating=="OthW")&(df.Heating=="Floor")].index)
df=df.drop(df[df.HeatingQC=="Po"].index)
df=df.drop(df[(df.Electrical=="FuseP")&(df.Electrical=="Mix")].index)
df=df.drop(df[(df.Functional=="Maj2")&(df.Functional=="Sev")].index)
df=df.drop(df[(df.GarageType=="2Types")].index)
df=df.drop(df[(df.SaleType=="ConLI")&(df.SaleType=="ConLw")&(df.SaleType=="CWD")&(df.SaleType=="Oth")&(df.SaleType=="Con")].index)
df=df.drop(df[(df.SaleCondition=="AdjLand")].index)
df=df.drop(df[(df.RoofMatl=="WdShngl")&(df.RoofMatl=="WdShake")&(df.RoofMatl=="Metal")&(df.RoofMatl=="Membran")&(df.RoofMatl=="Roll")&(df.RoofMatl=="ClyTile")].index)


df=df.drop(df[(df.OverallCond==2)&(df.OverallCond==1)].index)
df=df.drop(df[(df.OverallQual==2)&(df.OverallQual==1)].index)
df=df.drop(df[(df.MSSubClass==40)].index)
df=df.drop("LowQualFinSF",axis=1)
df=df.drop(df[(df.BsmtFullBath==3)].index)
df=df.drop(df[(df.BsmtHalfBath==2)].index)
df=df.drop(df[(df.BedroomAbvGr==6)&(df.BedroomAbvGr==0)&(df.BedroomAbvGr==8)].index)
df=df.drop(df[(df.KitchenAbvGr==3)&(df.KitchenAbvGr==0)].index)
df=df.drop(df[(df.Fireplaces==3)].index)
df=df.drop("EnclosedPorch",axis=1)
df=df.drop("3SsnPorch",axis=1)
df=df.drop("ScreenPorch",axis=1)
df=df.drop("PoolArea",axis=1)
df=df.drop("MiscVal",axis=1)
df=df.drop("BsmtFinSF2",axis=1)
df=df.drop("2ndFlrSF",axis=1)
df=df.drop(df[df.LotFrontage>200].index)
df=df.drop(df[df.LotArea>100000].index)
df=df.drop(df[df.MasVnrArea>1150].index)
df=df.drop(df[df.BsmtFinSF1>2000].index)
df=df.drop(df[df.TotalBsmtSF>3000].index)
df=df.drop(df[df.GrLivArea>4000].index)
df=df.drop(df[df.WoodDeckSF>800].index)
df=df.drop(df[df.OpenPorchSF>450].index)


###### suggestions of drops due to the twins       NOT GOOD
df=df.drop(df[df.GarageArea>1200].index)
# df=df.drop("GarageArea",axis=1)
# df=df.drop("1stFlrSF",axis=1)
# df=df.drop("FullBath",axis=1)
df=df.drop(df[(df.TotRmsAbvGrd==2)&(df.TotRmsAbvGrd==14)].index)
# df=df.drop("TotRmsAbvGrd",axis=1)
#######


#######  suggestions of drops due to nan          NOT GOOD
# df=df.drop("GarageType",axis=1)
df=make_one_hot(df,"GarageType")
# df=df.drop("GarageYrBlt",axis=1)
# df=df.drop("GarageFinish",axis=1)
df=make_one_hot(df,"GarageFinish")
# df=df.drop("GarageCars",axis=1)
df=df.drop(df[(df.GarageCars==4)].index)
# df=df.drop("GarageQual",axis=1)
df=df.drop(df[(df.GarageQual=="Ex")&(df.GarageQual=="Po")].index)
df=make_one_hot(df,"GarageQual")
# df=df.drop("GarageCond",axis=1)
df=df.drop(df[(df.GarageCond=="Ex")&(df.GarageCond=="Po")].index)
df=make_one_hot(df,"GarageCond")
# df=df.drop("1stFlrSF",axis=1)
# df=df.drop("1stFlrSF",axis=1)
#######

df=make_one_hot(df,"MSZoning")
df=make_one_hot(df,"LandContour")
df=make_one_hot(df,"LotShape")
df=make_one_hot(df,"LotConfig")
df=make_one_hot(df,"LandSlope")
df=make_one_hot(df,"Neighborhood")
df=make_one_hot(df,"Condition1")
# df=make_one_hot(df,"Condition2")
df=make_one_hot(df,"BldgType")
df=make_one_hot(df,"HouseStyle")
df=make_one_hot(df,"RoofStyle")
df=make_one_hot(df,"Exterior1st")
df=make_one_hot(df,"Exterior2nd")
df=make_one_hot(df,"MasVnrType")
df=make_one_hot(df,"ExterQual")
df=make_one_hot(df,"ExterCond")
df=make_one_hot(df,"Foundation")
df=make_one_hot(df,"BsmtQual")
df=make_one_hot(df,"BsmtCond")
df=make_one_hot(df,"BsmtExposure")
df=make_one_hot(df,"BsmtFinType1")
df=make_one_hot(df,"BsmtFinType2")
df=make_one_hot(df,"Heating")
df=make_one_hot(df,"HeatingQC")
df=make_one_hot(df,"Electrical")
df=make_one_hot(df,"KitchenQual")
df=make_one_hot(df,"Functional")
df=make_one_hot(df,"PavedDrive")
df=make_one_hot(df,"SaleType")
df=make_one_hot(df,"SaleCondition")
df=make_one_hot(df,"RoofMatl")


df.Street.replace({"Pave":1,"Grvl":0},inplace=True)
df.Utilities.replace({"AllPub":1,"NoSeWa":0},inplace=True)
df.CentralAir.replace({"Y":1,"N":0},inplace=True)

# df.info()
# df=df.dropna()

# df.dropna(how='any', inplace=True)


target=df.SalePrice
data=df.drop("SalePrice",axis=1)

KI=KNNImputer(n_neighbors=250, weights="uniform")
KI.fit(data)
data=KI.transform(data)

# r2_list=[]
# for i in range(1,100):
#     x_train,x_test,y_train,y_test=train_test_split(data,target,train_size=0.8)
#     LR=LinearRegression()
#     LR.fit(x_train,y_train)
#     y_p=LR.predict(x_test)
#     r2_list.append(r2_score(y_p,y_test))
# print(sum(r2_list)/len(r2_list))

x_train,x_test,y_train,y_test=train_test_split(data,target,train_size=0.8,random_state=4)


##################################################             LinearRegression
LR=LinearRegression()
LR.fit(x_train,y_train)
y_p=LR.predict(x_test)
print(r2_score(y_p,y_test))

##################################################             MLPRegressor      NOT GOOD
# SS=MinMaxScaler()
# SS.fit(x_train)
# x_train=SS.transform(x_train)
# x_test=SS.transform(x_test)

# MLP=MLPRegressor()
# MLP.fit(x_train,y_train)
# y_p=MLP.predict(x_test)
# print(r2_score(y_p,y_test))
#####################################
# r2_list=[]
# for i in range(2,500):
#     KI=KNNImputer(n_neighbors=i, weights="uniform")
#     KI.fit(data)
#     data=KI.transform(data)
#     x_train,x_test,y_train,y_test=train_test_split(data,target,train_size=0.8,random_state=4)
#     LR=LinearRegression()
#     LR.fit(x_train,y_train)
#     y_p=LR.predict(x_test)
#     r2_list.append(r2_score(y_p,y_test))

# plt.plot(range(2,500),r2_list)
# plt.show()

##########################################################################################################################
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from pandas import get_dummies
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer,KNNImputer
# from sklearn.metrics import r2_score,mean_absolute_error
# from sklearn.preprocessing import StandardScaler,PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from math import nan
# import csv

# def make_one_hot(data,field):
#     one_hot=pd.get_dummies(data[field],prefix=field)
#     data=data.drop(field,axis=1)
#     return data.join(one_hot)


# df=pd.read_csv("C:/Users/Nooshin/Desktop/sql/kaggle/house_price.csv")

# # df.info()

# ########################## train #########################
# df=df.drop("Id",axis=1)
# df=df.drop("Alley",axis=1)
# df=df.drop("PoolQC",axis=1)
# df=df.drop("Fence",axis=1)
# df=df.drop("MiscFeature",axis=1)
# df=df.drop("FireplaceQu",axis=1)
# df=df.drop(df[df.LotConfig=="FR3"].index)
# df=df.drop(df[df.Neighborhood=="Blueste"].index)
# df=df.drop(df[(df.Condition1=="RRNn")&(df.Condition1=="RRNe")].index)
# df=df.drop("Condition2",axis=1)
# df=df.drop(df[(df.RoofStyle=="Mansard")&(df.RoofStyle=="Shed")].index)
# df=df.drop(df[(df.Exterior1st=="BrkComm")&(df.Exterior1st=="Stone")&(df.Exterior1st=="AsphShn")&(df.Exterior1st=="ImStucc")&(df.Exterior1st=="CBlock")].index)
# df=df.drop(df[(df.Exterior2nd=="Stone")&(df.Exterior2nd=="AsphShn")&(df.Exterior2nd=="CBlock")&(df.Exterior2nd=="Other")].index)
# df=df.drop(df[(df.ExterCond=="Ex")&(df.ExterCond=="Po")].index)
# df=df.drop(df[(df.Foundation=="Stone")&(df.Foundation=="Wood")].index)
# df=df.drop(df[df.BsmtCond=="Po"].index)
# df=df.drop(df[(df.Heating=="Grav")&(df.Heating=="Wall")&(df.Heating=="OthW")&(df.Heating=="Floor")].index)
# df=df.drop(df[df.HeatingQC=="Po"].index)
# df=df.drop(df[(df.Electrical=="FuseP")&(df.Electrical=="Mix")].index)
# df=df.drop(df[(df.Functional=="Maj2")&(df.Functional=="Sev")].index)
# df=df.drop(df[(df.GarageType=="2Types")].index)
# df=df.drop(df[(df.SaleType=="ConLI")&(df.SaleType=="ConLw")&(df.SaleType=="CWD")&(df.SaleType=="Oth")&(df.SaleType=="Con")].index)
# df=df.drop(df[(df.SaleCondition=="AdjLand")].index)
# df=df.drop(df[(df.RoofMatl=="WdShngl")&(df.RoofMatl=="WdShake")&(df.RoofMatl=="Metal")&(df.RoofMatl=="Membran")&(df.RoofMatl=="Roll")&(df.RoofMatl=="ClyTile")].index)
# df=df.drop(df[(df.OverallCond==2)&(df.OverallCond==1)].index)
# df=df.drop(df[(df.OverallQual==2)&(df.OverallQual==1)].index)
# df=df.drop(df[(df.MSSubClass==40)].index)
# df=df.drop("LowQualFinSF",axis=1)
# df=df.drop(df[(df.BsmtFullBath==3)].index)
# df=df.drop(df[(df.BsmtHalfBath==2)].index)
# df=df.drop(df[(df.BedroomAbvGr==6)&(df.BedroomAbvGr==0)&(df.BedroomAbvGr==8)].index)
# df=df.drop(df[(df.KitchenAbvGr==3)&(df.KitchenAbvGr==0)].index)
# df=df.drop(df[(df.Fireplaces==3)].index)
# df=df.drop("EnclosedPorch",axis=1)
# df=df.drop("3SsnPorch",axis=1)
# df=df.drop("ScreenPorch",axis=1)
# df=df.drop("PoolArea",axis=1)
# df=df.drop("MiscVal",axis=1)
# df=df.drop("BsmtFinSF2",axis=1)
# df=df.drop("2ndFlrSF",axis=1)
# df=df.drop(df[df.LotFrontage>200].index)
# df=df.drop(df[df.LotArea>100000].index)
# df=df.drop(df[df.MasVnrArea>1150].index)
# df=df.drop(df[df.BsmtFinSF1>2000].index)
# df=df.drop(df[df.TotalBsmtSF>3000].index)
# df=df.drop(df[df.GrLivArea>4000].index)
# df=df.drop(df[df.WoodDeckSF>800].index)
# df=df.drop(df[df.OpenPorchSF>450].index)
# df=df.drop(df[df.GarageArea>1200].index)
# df=df.drop(df[(df.TotRmsAbvGrd==2)&(df.TotRmsAbvGrd==14)].index)
# df=make_one_hot(df,"GarageType")
# df=make_one_hot(df,"GarageFinish")
# df=df.drop(df[(df.GarageCars==4)].index)
# df=df.drop(df[(df.GarageQual=="Ex")&(df.GarageQual=="Po")].index)
# df=make_one_hot(df,"GarageQual")
# df=df.drop(df[(df.GarageCond=="Ex")&(df.GarageCond=="Po")].index)
# df=make_one_hot(df,"GarageCond")
# df=make_one_hot(df,"MSZoning")
# df=make_one_hot(df,"LandContour")
# df=make_one_hot(df,"LotShape")
# df=make_one_hot(df,"LotConfig")
# df=make_one_hot(df,"LandSlope")
# df=make_one_hot(df,"Neighborhood")
# df=make_one_hot(df,"Condition1")
# df=make_one_hot(df,"BldgType")
# df=make_one_hot(df,"HouseStyle")
# df=make_one_hot(df,"RoofStyle")
# df=make_one_hot(df,"Exterior1st")
# df=make_one_hot(df,"Exterior2nd")
# df=make_one_hot(df,"MasVnrType")
# df=make_one_hot(df,"ExterQual")
# df=make_one_hot(df,"ExterCond")
# df=make_one_hot(df,"Foundation")
# df=make_one_hot(df,"BsmtQual")
# df=make_one_hot(df,"BsmtCond")
# df=make_one_hot(df,"BsmtExposure")
# df=make_one_hot(df,"BsmtFinType1")
# df=make_one_hot(df,"BsmtFinType2")
# df=make_one_hot(df,"Heating")
# df=make_one_hot(df,"HeatingQC")
# df=make_one_hot(df,"Electrical")
# df=make_one_hot(df,"KitchenQual")
# df=make_one_hot(df,"Functional")
# df=make_one_hot(df,"PavedDrive")
# df=make_one_hot(df,"SaleType")
# df=make_one_hot(df,"SaleCondition")
# df=make_one_hot(df,"RoofMatl")
# df.Street.replace({"Pave":1,"Grvl":0},inplace=True)
# df.Utilities.replace({"AllPub":1,"NoSeWa":0},inplace=True)
# df.CentralAir.replace({"Y":1,"N":0},inplace=True)
# ####################################

# target=df.SalePrice
# data=df.drop("SalePrice",axis=1)

# KI=KNNImputer()
# KI.fit(data)
# data=KI.transform(data)

# SS=StandardScaler()
# SS.fit(data)
# data=KI.transform(data)

# x_train=data[0:1459]
# y_train=target[0:1459]
# x_test=data[1460:2919]

# LR=LinearRegression()
# LR.fit(x_train,y_train)
# y_p=LR.predict(x_test)
# print(y_p)
# print(y_p.shape)

# # with open('C:/Users/Nooshin/Desktop/sql/kaggle/house_price_prediction.csv', 'w', newline='') as file:
# #      writer = csv.writer(file)
     
# #      writer.writerow(["SalePrice"])
# #      writer.writerow(y_p)