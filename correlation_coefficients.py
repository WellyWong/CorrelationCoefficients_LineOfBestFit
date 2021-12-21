from math import sqrt


def correlation_coeff(points):
    x = [elt[0] for elt in points]
    y = [elt[1] for elt in points]
    n = len(points)
    mean_x = sum(x)/n
    total = 0
    for elt in x:
        total += (elt - mean_x)**2
    total = total / (n-1)
    std_dev_x = sqrt(total)

    mean_y = sum(y)/n
    total = 0
    for elt in y:
        total += (elt - mean_y)**2
    total = total / (n - 1)
    std_dev_y = sqrt(total)

    r = 0
    for i in range(n):
        r += ((x[i] - mean_x)/std_dev_x) * ((y[i] - mean_y)/std_dev_y)
    r = r/(n-1)

    return r

points = [[1, 1], [2, 2], [2, 3], [3, 6]]
corr_coeff = correlation_coeff(points)
print(corr_coeff)

#https://machinelearningmastery.com/how-to-use-correlation-to-understand-the-relationship-between-variables/
"""
We will generate 1,000 samples of two two variables with a strong positive correlation. The first variable will be 
random numbers drawn from a Gaussian distribution with a mean of 100 and a standard deviation of 20. The second variable
will be values from the first variable with Gaussian noise added with a mean of a 50 and a standard deviation of 10.

We will use the randn() function to generate random Gaussian values with a mean of 0 and a standard deviation of 1, 
then multiply the results by our own standard deviation and add the mean to shift the values into the preferred range.

The pseudorandom number generator is seeded to ensure that we get the same sample of numbers each time the code is run.
"""
from numpy import mean, std, cov
from numpy.random import randn, seed
from matplotlib import pyplot

#seed random number generator
seed(1)
data1 = 20 * randn(1000) + 100
data2 = data1 + (10 * randn(1000) + 50)

print('data1: mean = %.3f  stdv = %.3f' % (mean(data1), std(data1)))
print('data2: mean = %.3f  stdv = %.3f' % (mean(data2), std(data2)))

pyplot.scatter(data1, data2)
pyplot.show()

def covariance(x, y):
    n = len(x)
    mean_x = sum(x)/n
    mean_y = sum(y)/n
    cov = 0
    for i in range(n):
        cov += (x[i] - mean_x) * (y[i] - mean_y)
    cov = cov / (n-1)
    return cov

covar = covariance(data1, data2)
print(covar)

covar2 = cov(data1, data2)
print(covar2)

#calculate Pearson's correlation
from scipy.stats import pearsonr, spearmanr

corr, _ = pearsonr(data1, data2)
print('Pearsons correlation: %.3f' % corr)

#calculate Spaearman's correlation
corr_sp, _ = spearmanr(data1, data2)
print('Spearman"s correlation: %.3f' % corr_sp)


"""
y = a + b * x
where:
b = ( sum(xi * yi) - n * x_mean * y_mean ) / sum((xi - x_mean)^2)
a = y_mean - b * x_mean
"""
X = [0, 5, 10, 15, 20]
Y = [0, 7, 10, 13, 20]

def best_fit(X, Y):
    n = len(X)
    x_mean = sum(X)/n
    y_mean = sum(Y)/n
    numerator = sum([xi * yi for xi, yi in zip(X, Y)]) - n * x_mean * y_mean
    denominator = sum([(xi - x_mean)**2 for xi in X])
    b = numerator / denominator
    a = y_mean - b * x_mean
    print('Best fit line: y = {:.2f} + {:.2f}x'.format(a, b))
    return a, b

a, b = best_fit(X, Y)
pyplot.scatter(X, Y)
y_fit = [a + b * xi for xi in X]
pyplot.plot(X, y_fit)
pyplot.show()

