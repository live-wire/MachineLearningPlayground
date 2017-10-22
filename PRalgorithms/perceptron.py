import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from time import sleep

N = 50
x1 = np.random.random((N,2))
x1 = x1
c = x1[:,0]+x1[:,1]>1
x1 = np.hstack([x1,np.reshape(c,(N,1))])
type1 = np.array(list(filter(lambda x: x[2] == 0, x1)))
type2 = np.array(list(filter(lambda x: x[2] > 0, x1)))
# print type1,"\n\n",type2

plt.scatter(type1[:,0],type1[:,1],marker="o",c="r",label="class-0")
plt.scatter(type2[:,0],type2[:,1],marker="x",c="b",label="class-1")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Check it out")
plt.axis([0,1,0,1])

# print "FEATURE MATRIX",x1[:,0:2]

w = np.array([[0, 0, 0]])
Y = np.array([])
i=0
x11 = np.hstack((np.ones((N,1)),x1[:,0:2]))
y11=x1[:,2]
print x11,y11



t=0
alpha = 0.1
def graph(w0 = [-.5, 1, 0],bold=False):
    dontplot = 0
    if w0[1]!=0 and w0[2]!=0:
        y = np.arange(0, 1, 0.1)
        x=(-w0[0] - w0[2] * y)/w0[1]
    else:
        if w0[1]==0 and w0[2]!=0:
            x=np.arange(0, 1, 0.1)
            y=(-w0[0])/w0[2]
        elif w0[1]!=0 and w0[2]==0:
            y = np.arange(0, 1, 0.1)
            x = (-w0[0])/w0[1]
        else:
            dontplot =1

    if dontplot!=1:
        if bold==True:
            plt.plot(x,y,linewidth=7.0)
        else:
            plt.plot(x,y)
        plt.pause(0.02)





def net_input(x,w):
    return np.dot(x,w[1:]) + w[0]

def predict(x,w):
    return np.where(net_input(x,w)>=0.0,1,-1)



while(len(Y)!=0 or i==0):
    Y = []
    i=1
    s=np.array([0,0,0],dtype='float64')
    for key,val in enumerate(x11):
        if y11[key] == 0:
            delta = -1
        else:
            delta = 1
        if delta*w[t].transpose().dot(val) >= 0 :
            Y.append(np.hstack((val, delta)))

    print "Y---",len(Y),"  ",Y
    graph(w[t])
    if len(Y)==0:
        graph(w[-1],True)
        print 'solution',w[-1]
        break

    for item in Y:
        s+=item[3]*item[0:3]
    s=alpha*s
    print "s:",s," alpha:",alpha
    t=t+1
    z = w[t - 1] - s
    zx = []
    zx.append(z)
    w = np.concatenate((w,np.array(zx)))
    print "W",t,"    ",w

    if t==100:
        break

plt.show()