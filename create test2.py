{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Create histogram\n",
    "import numpy as np\n",
    "s = np.random.uniform(-1,0,1000)\n",
    "import matplotlib.pyplot as plt\n",
    "count, bins, ignored = plt.hist(s,50, density=True)\n",
    "plt.plot(bins, np.ones_like(bins), linewidth=1, color='r')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Create Normalpdf\n",
    "import matplotlib.pyplot as pl\n",
    "import numpy as np\n",
    "import matplotlib.mlab as MLA\n",
    " \n",
    "mu, sigma = 10, 10\n",
    "x = mu + sigma*np.random.randn(5000)\n",
    "\n",
    " \n",
    "\n",
    "n, bins, patches = plt.hist(x, 200, normed=1, facecolor='blue')\n",
    "\n",
    "y = MLA.normpdf( bins, mu, sigma)\n",
    "\n",
    "l = plt.plot(bins, y, 'g--', linewidth=2)\n",
    " \n",
    "plt.xlabel('samples')\n",
    "plt.ylabel('p')\n",
    "plt.title(r'$Normal\\ pdf\\ m=10,\\ \\sigma=10$')\n",
    "plt.axis([-30, 50, 0, 0.042])\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
