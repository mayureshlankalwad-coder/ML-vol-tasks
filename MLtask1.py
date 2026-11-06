import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("movie_train_data.csv")
print(max(data["romance_level"]))
