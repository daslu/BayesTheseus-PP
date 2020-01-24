# import numpy as np
# import scipy.stats as stats
# from matplotlib import pyplot as plt
#
# x = np.linspace (0, 100, 200)
# y1 = stats.gamma.pdf(x, a=2, scale=2) #a = alpha, scale = 1/beta
# plt.plot(x, y1, "y-", label=(r'$\alpha=1, \beta=1$'))
#
#
# plt.ylim([0,0.08])
# plt.xlim([0,150])
# plt.show()


import numpy as np
from scipy.stats import gamma
from matplotlib import pyplot as plt

#----------------------------------------------------------------------
# This function adjusts matplotlib settings for a uniform feel in the textbook.
# Note that with usetex=True, fonts are rendered with LaTeX.  This may
# result in an error if LaTeX is not installed on your system.  In that case,
# you can set usetex to False.
#from astroML.plotting import setup_text_plots
#setup_text_plots(fontsize=8, usetex=True)

#------------------------------------------------------------
# plot the distributions
k_values = [2, 3, 5,1]
theta_values = [1, 1, 0.5,1] #Remember theta = 1/beta;
linestyles = ['-', '--', ':', '-.']
x = np.linspace(1E-6, 10, 1000)

#------------------------------------------------------------
# plot the distributions
fig, ax = plt.subplots(figsize=(5, 3.75))

for k, t, ls in zip(k_values, theta_values, linestyles):
    dist = gamma(k, 0, t)
    plt.plot(x, dist.pdf(x), ls=ls, c='black',
             label=r'$k=%.1f,\ \theta=%.1f$' % (k, t))

plt.xlim(0, 10)
plt.ylim(0, 0.45)

plt.xlabel('$x$')
plt.ylabel(r'$p(x|k,\theta)$')
plt.title('Gamma Distribution')

plt.legend(loc=0)
plt.show()