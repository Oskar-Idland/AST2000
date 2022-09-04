import numpy as np
from scipy.integrate import quad
from f_P import P2


# Challenge A_1_1 --------------------
"""
See P2() in the file f_P.py
"""


# Challenge A_1_2 --------------------
"""
P(a ≤ x ≤ b) stands for the total probability that x will be in the interval [a, b]
"""


# Challenge A_1_3 --------------------
mu = 20
sigma = 2
a1 = mu - sigma
b1 = mu + sigma
a2 = mu - (2*sigma)
b2 = mu + (2*sigma)
a3 = mu - (3*sigma)
b3 = mu + (3*sigma)

p1 = P2(a1, b1, mu, sigma)  # Probability for -1*sigma ≤ x-mu ≤ 1*sigma
p2 = P2(a2, b2, mu, sigma)  # Probability for -2*sigma ≤ x-mu ≤ 2*sigma
p3 = P2(a3, b3, mu, sigma)  # Probability for -3*sigma ≤ x-mu ≤ 3*sigma

"""
p1 = 0.682689492137086
p2 = 0.9544997361036417
p3 = 0.9973002039367398
"""


# Challenge A_1_4 --------------------
"""
FWHM is an acronym for Full Width at Full Maximum
This means the width of the "bump" between the points mu-x1 and mu+x1 where the curve is at half ist maximum value 
(which for the gaussian distribution is at mu).
We will have to find x1 so that 

f(mu + x1) = f(mu - x1) = f(mu)/2
1/(sigma*sqrt(2*pi)) * exp((-(mu+x1-mu)**2)/(2*sigma**2) = 1/2 * 1/(sigma*sqrt(2*pi)) * exp((-(mu-mu)**2)/(2*sigma**2)
exp((-(x1-mu)**2)/(2*sigma**2) = 1/2 * exp((-(mu-mu)**2)/(2*sigma**2)
exp(-(x1**2)/(2*sigma**2) = 1/2 * exp(0)
2 = exp((x1**2)/(2*sigma**2)
ln(2) = (x1**2)/(2*sigma**2)
2*ln(2)*sigma**2 = x1**2
x1 = sqrt(2*ln(2)*sigma**2)

Since this is the distance to one side, and the bump is symmetric, the distance from the left point where 
f(x) = f(mu)/2 to the right point where f(x) = f(mu)/2 is equal to 2*sqrt(2*ln(2)*sigma**2)
This is FWHM
"""
