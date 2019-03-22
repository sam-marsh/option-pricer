import itertools
import numpy as np
from scipy.special import binom


def price(n, m, t, mu, sigma, s0, f, r, q):
    a = 2 * np.sqrt(t / m) * np.linalg.cholesky(sigma)
    b = np.zeros((n, 1))

    for i in range(n):
        b[i] = (r - q[i]) * t - m * np.sum(np.log((np.exp(a[i, :]) + 1) / 2))

    exp = 0
    for y in itertools.product(range(m + 1), repeat=n):
        x = a * np.reshape(np.array(y), (n, 1)) + b
        w = np.exp(x)
        st = np.multiply(w, np.reshape(s0, (n, 1)))
        payoff = f(st)

        pr = 1
        for i in range(n):
            pr *= binom(m, y[i])
        pr *= (1/2)**(m*n)

        exp = exp + payoff * pr

    return exp * np.exp(-r * t)


if __name__ == '__main__':
    # print(price(
    #     3, 4, 0.25,
    #     np.transpose(np.array([[0, -0.03, 0.035]])),
    #     np.matrix(
    #         [[0.2*0.2, 0.2*0.4*0.9, 0.2*0.1*0.6],
    #          [0.2*0.4*0.9, 0.4*0.4, 0.4*0.1*0.8],
    #          [0.2*0.1*0.6, 0.4*0.1*0.8, 0.1*0.1]]),
    #     [5, 3, 2],
    #     lambda s: max(10 - sum(s), 0),
    #     0.06,
    #     [0.04, 0.01, 0.02]
    # ))
    print(price(
        2, 60, 5,
        np.transpose(np.array([[0.04, 0.04]])),
        np.matrix([[0.2*0.2, 0.2*0.2*0.7], [0.2*0.2*0.7, 0.2*0.2]]),
        [380, 400],
        lambda s: max(s[0] - 380, 0) + max(s[1] - 400, 0),
        0.1,
        [0, 0]
    ))
