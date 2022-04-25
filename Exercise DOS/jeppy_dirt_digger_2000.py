import numpy as np
import pandas as pd

data = pd.read_csv('C:/Users/Jesper/OneDrive/Dokument/GitHub/BigD/Exercise DOS/data.csv',index_col=0,low_memory=False)
 
print(data)

data = data.to_numpy()
print(data)
#print(np.var(data[:,0]))