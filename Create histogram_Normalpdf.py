# Create histogram
import numpy as np
s = np.random.uniform(-1,0,1000)
import matplotlib.pyplot as plt
count, bins, ignored = plt.hist(s,50, density=True)
plt.plot(bins, np.ones_like(bins), linewidth=1, color='r')
plt.show()

# Create Normalpdf
import matplotlib.pyplot as pl
import numpy as np
import matplotlib.mlab as MLA
 
mu, sigma = 10, 10
x = mu + sigma*np.random.randn(5000)

 

n, bins, patches = plt.hist(x, 200, normed=1, facecolor='blue')

y = MLA.normpdf( bins, mu, sigma)

l = plt.plot(bins, y, 'g--', linewidth=2)
 
plt.xlabel('samples')
plt.ylabel('p')
plt.title(r'$Normal\ pdf\ m=10,\ \sigma=10$')
plt.axis([-30, 50, 0, 0.042])
plt.grid(True)
plt.show()
