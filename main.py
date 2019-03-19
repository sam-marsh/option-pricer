import itertools
import numpy as np
from scipy.special import binom


def price(n, m, t, mu, sigma, s0, f):
    a = 2 * np.sqrt(t / m) * np.linalg.cholesky(sigma)
    b = mu * t - 2 * a * np.ones((n, 1))  # TODO why is this 2 *

    exp = 0
    t = 0
    for y in itertools.product(range(m + 1), repeat=n):
        x = a * np.reshape(np.array(y), (n, 1)) + b
        w = np.exp(x)
        st = np.multiply(w, np.reshape(s0, (n, 1)))
        payoff = f(st)

        pr = 1
        for i in range(n):
            pr *= binom(m, y[i])
        pr *= (1/2)**(m*n)

        t += pr

        exp = exp + payoff * pr

    return exp


if __name__ == '__main__':
    print(price(
        3, 4, 1/4,
        np.transpose(np.array([[0, -0.03, 0.035]])),
        np.matrix([[0.04, 0.072, 0.012], [0.072, 0.16, 0.032], [0.012, 0.032, 0.01]]),
        [5, 3, 2],
        lambda s: max(10 - np.sum(s), 0)
    ))
