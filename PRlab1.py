import numpy as np
import time
import sys


#1.6 Generate input of random inputs of size 10 x 3

a = np.random.random((10,3))
b = 100*a+1
b=b.astype(int)

print "MEAN=",np.mean(b,axis=0)
print "STD=",np.var(b,axis=0)**.5




#1.7 Scatter Plots
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

# plt.scatter(b[:,0],b[:,1],label='scatter',color='blue',marker="x",s=100)
#
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.title('Scatter')
# plt.show()

#1.8 label data

import pandas as pd

df = pd.DataFrame({"var1":b[:,0],"var2":b[:,1]})
print "MEAN=", np.mean(df,axis=0)," STD=",np.std(df,axis=0)


#Loading matlab data
import scipy.io
from PIL import Image
mat = scipy.io.loadmat('cigars.mat')


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

ax.scatter(b[:,0],b[:,1],b[:,2], c='r',marker='o')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title("3d Scatter")

plt.show()






