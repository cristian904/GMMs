# GMM implementation
# good resource http://www.rmki.kfki.hu/~banmi/elte/bishop_em.pdf

import numpy as np
from scipy import stats
import seaborn as sns
from random import shuffle, uniform
sns.set_style("white")

#Generate some data from 2 different distributions
x1 = np.linspace(start=-10, stop=10, num=1000)
x2 = np.linspace(start=5, stop=10, num=800)
y1 = stats.norm.pdf(x1, loc=3, scale=1.5)
y2 = stats.norm.pdf(x2, loc=0, scale=3)

#Put data in dataframe for better handling
x = list(x1)
x.extend(list(x2))
shuffle(x)

K = 2 #number of assumed distributions within the dataset
epsilon = 0.001 #tolerance change for log-likelihood
max_iter = 100

#gaussian pdf function
def G(datum, mu, sigma):
    y = (1 / (np.sqrt((2 * np.pi) * sigma * sigma)) * np.exp(datum-mu)*(datum-mu)/(2*sigma*sigma))
    return y

#compute log-likelihood
def L(X, N, mu, sigma, pi):
    L = 0
    for i in range(N):
        Gk = 0
        for k in range(K):
            Gk += pi[k] * G(X[i], mu[k], sigma[k])
        L += Gk
    print(L)
    return np.log(L)


def estimate_gmm(X, K, epsilon, max_iter):
    N = len(X)
    # assign random mean and variance to each distribution
    mu, sigma = [uniform(0, 10) for _ in range(K)], [uniform(0, 10) for _ in range(K)]
    # assign random probability to each distribution
    pi = [uniform(0, 10) for _ in range(K)]
    mu = [2, 0]
    sigma = [1, 1]
    current_loglike = np.inf
    for _ in range(max_iter):
        previous_loglike = current_loglike
        #E step
        mixture_affiliation_all_k = {}
        for i in range(N):
            parts = [pi[k] * G(X[i], mu[k], sigma[k]) for k in range(K)]
            total = sum(parts)
            for k in range(K):
                mixture_affiliation_all_k[(i, k)] = parts[k] / total

        #M step
        mixture_affiliation_for_k = [sum(mixture_affiliation_all_k[(i, k)] for i in range(N)) for k in range(K)]
        for k in range(K):
            pi[k] = mixture_affiliation_for_k[k] / N
            mu[k] = sum([mixture_affiliation_all_k[(i, k)] * X[i] for i in range(N)]) / mixture_affiliation_for_k[k]
            sigma[k] = sum([mixture_affiliation_all_k[(i, k)] * (X[i] - mu[k]) ** 2 for i in range(N)]) / mixture_affiliation_for_k[k]

        current_loglike = L(X, N, mu, sigma, pi)
        if abs(previous_loglike - current_loglike) < epsilon:
            print("break")
            break
    return mu, sigma, pi

print(estimate_gmm(x, K, epsilon, max_iter))