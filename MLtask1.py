import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sm
data=pd.read_csv("movie_train_data.csv")

weights=np.array([1.0,1.0])
bias=-10
w1=sm.Symbol("w1")
w2=sm.Symbol("w2")
b=sm.Symbol("b")

def sigmoid(x):
    return 1/(1+sm.exp(-x))
for i in range(100):
    print(i)
    total=0
    for i in range(100):
        model_rawoutput=w1*data.loc[i,"action_level"]+w2*data.loc[i,"romance_level"]+b
        after_sigmoid=sigmoid(model_rawoutput)
        actual_output=data.loc[i,"like_movie"]
        loss=actual_output*sm.log(after_sigmoid)+(1-actual_output)*sm.log(1-after_sigmoid)
        total+=loss

    avj_loss=total/100
    g_w1=sm.diff(avj_loss,w1)
    g_w2=sm.diff(avj_loss,w2)
    g_b=sm.diff(avj_loss,b)
    ch_x=g_w1.subs({w1:weights[0],w2:weights[1],b:bias})
    ch_y=g_w2.subs({w1:weights[0],w2:weights[1],b:bias})
    ch_b=g_b.subs({w1:weights[0],w2:weights[1],b:bias})
    weights[0]=weights[0]+0.1*ch_x
    weights[1]=weights[1]+0.1*ch_y
    bias=bias+0.001*ch_b

print(f'weights:{weights},biases:{bias}')

