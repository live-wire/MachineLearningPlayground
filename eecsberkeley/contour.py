import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

fig = plt.figure()
ax=fig.add_subplot(111)
u = np.linspace(-1,1,100)
a= [0,100,200,300,400]
x,y = np.meshgrid(a,a)
print x,y


#z=<YOUR LOG FUNCTION>
# for i,item in enumerate(x):
#
# ax.contourf(x,y,z)
# plt.show()