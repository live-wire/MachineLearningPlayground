import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math

# mu = 0
# variance = 0.5
# sigma = math.sqrt(variance)
# x = np.linspace(mu-3*variance,mu+3*variance, 100)
# plt.plot(x,mlab.normpdf(x, mu, sigma))
# mu = 1
# x = np.linspace(mu-3*variance,mu+3*variance, 100)
# plt.plot(x,mlab.normpdf(x, mu, sigma))
#
# plt.show()





mean = [0, 0]
cov = [[9, 9], [9, 9]]

x, y = np.random.multivariate_normal(mean, cov, 5000).T
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()
