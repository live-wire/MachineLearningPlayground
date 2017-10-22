import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

from decimal import *
getcontext().prec = 15
# N = 20
# x1 = np.random.random((N,2))
# x1 = x1
# c = 5*(x1[:,0] - 0.5)**3 -x1[:,0]**2 + 0.75 - x1[:,1]>0
# x1 = np.hstack([x1,np.reshape(c,(N,1))])
# type1 = np.array(list(filter(lambda x: x[2] == 0, x1)))
# type2 = np.array(list(filter(lambda x: x[2] > 0, x1)))



######    LINEAR   --------------------

def graph(w0 = [-.5, 1, 0],bold=False):
    dontplot = 0
    if w0[1]!=0 and w0[2]!=0:
        y = np.arange(0, 1, 0.1)
        x=(float(w0[2]) - float(w0[1]) * y)/float(w0[0])
        print x,y
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
#
# def svm_linear(X,Y):
#     w = np.zeros(len(X[0]))
#     alpha = 1 #learning rate
#     iterations= 10000
#     misclassifications = []
#
#     #Gradient Descent belowww
#     for iteration in range(1,iterations):
#         misclassification = 0
#         print iteration,w
#         for i,x in enumerate(X):
#
#             if(Y[i]*np.dot(X[i],w)) < 1:
#                 #misclassification
#                 w = w + alpha*((X[i] * Y[i]) + (-2 *(1/iteration)*w))
#                 misclassification = 1
#             else:
#                 #correct classification
#                 w = w + alpha * (-2 *(1/iteration)*w)
#         misclassifications.append(misclassification)
#     plt.plot(misclassifications, '|')
#     plt.ylim(0.5, 1.5)
#     plt.axes().set_yticklabels([])
#     plt.xlabel('Epoch')
#     plt.ylabel('Misclassified')
#     plt.show()
#
#
def checker(a):
    if a==0:
        return -1
    else:
        return 1

# N = 20
# x1 = np.random.random((N,2))
# x1 = x1
#
# c = -.75*x1[:,0] - x1[:,1] + 1 > 0
#
# x1 = np.hstack([x1,np.reshape(c,(N,1))])
# type1 = np.array(list(filter(lambda x: x[2] == 0, x1)))
# type2 = np.array(list(filter(lambda x: x[2] > 0, x1)))
#
# y = np.array(list(map(lambda x:checker(x[2]), x1)))
# x1 = np.hstack([x1[:,0:2],np.ones((N,1))*-1])
# svm_linear(x1,y)
#
# plt.scatter(type1[:,0],type1[:,1],marker="o",c="r",label="class-0")
# plt.scatter(type2[:,0],type2[:,1],marker="x",c="b",label="class-1")
#
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.title("Check it out")
# # plt.axis([0,1,0,1])
#
# plt.show()




#Step 1 - Define our data

#Input data - Of the form [X value, Y value, Bias term]
# X = np.array([
#     [-2,4,-1],
#     [4,1,-1],
#     [1, 6, -1],
#     [2, 4, -1],
#     [6, 2, -1],
# ])
#
# #Associated output labels - First 2 examples are labeled '-1' and last 3 are labeled '+1'
# y = np.array([-1,-1,1,1,1])


N = 20
x1 = np.random.random((N,2))
x1 = x1

c = x1[:,0] + x1[:,1] > .9
# c = x1[:,0] > .6

x1 = np.hstack([x1,np.reshape(c,(N,1))])
type1 = np.array(list(filter(lambda x: x[2] == 0, x1)))
type2 = np.array(list(filter(lambda x: x[2] > 0, x1)))

y = np.array(list(map(lambda x:checker(x[2]), x1)))
x1 = np.hstack([x1[:,0:2],np.ones((N,1))*-1])
X=x1

print X,y
#lets plot these examples on a 2D graph!
#for each example

def svm_sgd_plot(X, Y):
    # Initialize our SVMs weight vector with zeros (3 values)
    w = np.zeros(len(X[0]),dtype='float64')
    # The learning rate
    eta = 1
    # how many iterations to train for
    epochs = 10000
    # store misclassifications so we can plot how they change over time
    errors = []

    # training part, gradient descent part
    for epoch in range(1, epochs):
        if epoch%1000==0:
            print epoch
        error = 0
        for i, x in enumerate(X):
            # misclassification
            if (Y[i] * np.dot(X[i], w)) < 1:
                # misclassified update for ours weights
                w = w + float(eta) * ((X[i] * Y[i]) + (float(-2) * (float(1) / float(epoch)) * w))
                error = 1
            else:
                # correct classification, update our weights
                w = w + float(eta) * (float(-2) * (float(1)/ float(epoch)) * w)
        errors.append(error)

    # lets plot the rate of classification errors during training for our SVM
    # plt.plot(errors, '|')
    # plt.xlabel('Epoch')
    # plt.ylabel('Misclassified')
    # plt.show()

    return w

w = svm_sgd_plot(X,y)
print w

for d, sample in enumerate(X):
    # Plot the negative samples
    if y[d] < 0:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    # Plot the positive samples
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

# Add our test samples
# plt.scatter(2,2, s=120, marker='_', linewidths=2, color='yellow')
# plt.scatter(4,3, s=120, marker='+', linewidths=2, color='blue')

# Print the hyperplane calculated by svm_sgd()
# x2=[w[0],w[1],-w[1],w[0]]
# x3=[w[0],w[1],w[1],-w[0]]
#
# x2x3 =np.array([x2,x3])
# X,Y,U,V = zip(*x2x3)
# ax = plt.gca()
# ax.quiver(X,Y,U,V,scale=1, color='blue')
graph(w)

plt.show()