import numpy as np
import matplotlib.pyplot as plt
mu = [15, 5]
sigma = [[20, 0], [0, 10]]
samples = np.random.multivariate_normal(mu, sigma, size=100)

sum1=0
sum2=0
for sample in samples:
    sum1+=sample[0]
    sum2+=sample[1]
mu = []
mu.append(sum1/len(samples))
mu.append(sum2/len(samples))

muvar=np.array(mu)
print "MLE mu = ",muvar

sumsigma = 0
for sample in samples:
    sample = np.array(sample)
    temp = sample-muvar
    print temp,temp.dot(np.transpose(temp))
    #temp.dot(np.transpose(temp))
variance = sumsigma/len(samples)
print variance


plt.scatter(samples[:, 0], samples[:, 1])
plt.show()