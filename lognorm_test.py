from scipy.stats import lognorm
import matplotlib.pyplot as plt
import numpy as np

plt.figure()
ax = plt.subplot(111)
s = 1
mean, var, skew, kurt = lognorm.stats(s, moments='mvsk')
print mean, var, skew, kurt
x = np.linspace(lognorm.ppf(0.01, s), 
        lognorm.ppf(0.99, s), 100)
ax.plot(x, lognorm.pdf(x, s),
        'r-', lw=5, alpha=0.6, label='lognorm pdf')
#print x
plt.show()
#rv = lognorm(s)
#ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
