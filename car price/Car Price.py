import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,r2_score
import pandas_profiling
from pandas_profiling import ProfileReport
from pandas_profiling.utils.cache import cache_file


def make_one_hot(data,field):
    one_hot=pd.get_dummies(data[field],prefix=field)
    one_hotsum = one_hot.apply(np.sum, axis=1)
    for i in range(data.shape[0]):
        if one_hotsum.iloc[i]==0:
            for j in range(one_hot.shape[1]):
                one_hot.iat[i,j]=np.nan
    data=data.drop(field,axis=1)
    return data.join(one_hot)


df=pd.read_csv("CarPrice.csv")
# print('Pandas profiling version:',pandas_profiling.__version__)
profile = ProfileReport(df, title="CarPrice Dataset", html={'style': {'full_width': True}}, sort=None)
profile.to_notebook_iframe()
profile.to_widgets()


df=df.drop(["CarName","wheelbase","car_ID","enginelocation"],axis=1)
df=df.drop(["carlength","carwidth","carheight"],axis=1)


df=make_one_hot(df,"carbody")
df=make_one_hot(df,"drivewheel")
df=make_one_hot(df,"enginetype")
df=make_one_hot(df,"fuelsystem")

df.fueltype.replace({"gas":0,"diesel":1},inplace=True)
df.aspiration.replace({"std":0,"turbo":1},inplace=True)
df.doornumber.replace({"two":0,"four":1},inplace=True)
df.cylindernumber.replace({"four":4,"six":6,"five":5,"twelve":12,"three":3,"eight":8,"two":2},inplace=True)
df.symboling.replace({3:1,2:2,1:3,0:4,-1:5,-2:6},inplace=True)


y=df["price"]
x=df.drop("price",axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)


SS=StandardScaler()
SS.fit(x_train)
x_train=SS.transform(x_train)
x_test=SS.transform(x_test)

LR = LinearRegression(fit_intercept=True, normalize=False)
LR.fit(x_train,y_train)
y_p=LR.predict(x_test)
print(r2_score(y_test, y_p))


#######################################################################   DEEP LEARNING
import tensorflow
from tensorflow import keras


model=keras.models.Sequential()
model.add(keras.layers.Dense(100,activation="relu"))
model.add(keras.layers.Dense(50,activation="relu"))
model.add(keras.layers.Dense(1,activation="relu"))

model.compile(metrics="mean_absolute_error",
              loss="mean_absolute_error",
              optimizer="sgd")

history=model.fit(x_train,y_train,epochs=100,validation_split=0.15,callbacks=keras.callbacks.EarlyStopping(patience=5))

model.evaluate(x_test,y_test)