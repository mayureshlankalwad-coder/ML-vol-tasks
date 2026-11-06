import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sm
data=pd.read_csv("movie_test_data.csv")


w1,w2,b=0.75154176,1.2816427,-10.0010114754160
def sigmoid(x):
    return 1/(1+np.exp(-x))

def Model(action_level,romance_level):
    model_rawoutput=w1*action_level+w2*romance_level+b
    return sigmoid(model_rawoutput)>=0.5

#testing
True_positive=0#User will like movie and model predicted that only
False_positive=0#User won't like movie and model predicted the opposite\
True_negative=0#User won't like movi and model predicted that only
False_negative=0#User will like movie but model predicted the opposite
for i in range(30):
    models_output=int(Model(data.loc[i,"action_level"],data.loc[i,"romance_level"]))
    actual_output=int(data.loc[i,"like_movie"])
    if models_output and actual_output:
        True_positive+=1
    elif models_output and not actual_output:
        False_positive+=1
    elif not models_output and not actual_output:
        True_negative+=1
    else:
        False_negative+=1


print({"True_positive":True_positive,"False_positive":False_positive,"True_negative":True_negative,"False_negative":False_negative})
    