# Уравнение в частных производных
from ChM.diff_2_equations import tridiagonal_matrix_algorithm
import numpy as np


def heat_conduct_equation_explicit_scheme(a_coef, T, m, l, n, uxt0, ux0t, uxnt):
    t, x = [], []
    tau = T / m
    h = l / n

    for i in range(m + 1):
        t.append(i * tau)
    for j in range(n + 1):
        x.append(j * h)

    u = np.zeros((len(t), len(x)))
    for i in range(n + 1):
        u[0][i] = uxt0(x[i])
    for j in range(m + 1):
        u[j][0] = ux0t(t[j])
        u[j][-1] = uxnt(t[j])

    for i in range(m):
        for j in range(1, n):
            u[i + 1][j] = u[i][j] + a_coef * ((u[i][j + 1] - 2 * u[i][j] + u[i][j - 1]) * tau) / (h ** 2)

    return t, x, u


def heat_conduct_equation_implicit_scheme(a_coef, T, m, l, n, uxt0, ux0t, uxnt):
    t, x = [], []
    tau = T / m
    h = l / n

    for i in range(m + 1):
        t.append(i * tau)
    for j in range(n + 1):
        x.append(j * h)

    u = np.zeros((len(t), len(x)))
    for i in range(n + 1):
        u[0][i] = uxt0(x[i])
    for j in range(m + 1):
        u[j][0] = ux0t(t[j])
        u[j][len(x) - 1] = uxnt(t[j])

    alpha = a_coef * tau / (h ** 2)
    a, b, c = [0, ], [-(1 + 2 * alpha), ], []

    for i in range(n - 1):
        a.append(alpha)
        b.append(-(1 + 2 * alpha))
        c.append(alpha)
    c.append(0)
    for i in range(m):
        d = np.zeros(n - 1)
        d[0] = -(u[i][0] + alpha * ux0t(t[i + 1]))
        for j in range(1, n - 2):
            d[j] = -u[i][j]
        d[n - 2] = -(u[i][n - 2] + alpha * uxnt(t[i + 1]))
        u[i + 1, 1:u.shape[1] - 1] = tridiagonal_matrix_algorithm(a, b, c, d)

    return t, x, u


def heat_conduct_equation_implicit_scheme_common(a_coef, T, m, l, n, uxt0, ux0t, uxnt, alpha=0, beta=0, gamma=0, delta=0):
    t, x = [], []
    tau = T / m
    h = l / n

    for i in range(m + 1):
        t.append(i * tau)
    for j in range(n + 1):
        x.append(j * h)
    u = np.zeros((len(t), len(x)))
    for i in range(n + 1):
        u[0][i] = uxt0(x[i])

    sigma = a_coef * tau / (h ** 2)
    a, b, c = [0, ], [beta - alpha / h, ], [alpha / h, ]

    for i in range(n - 1):
        a.append(sigma)
        b.append(-(1 + 2 * sigma))
        c.append(sigma)
    a.append(-gamma / h)
    b.append(delta + gamma / h)
    c.append(0)

    for i in range(m):
        d = np.zeros(n + 1)
        d[0] = ux0t(t[i + 1]) / (beta - alpha / h)
        for j in range(1, n - 1):
            d[j] = -u[i][j]
        d[n - 1] = uxnt(t[i + 1]) / (delta + gamma / h)
        u[i + 1] = tridiagonal_matrix_algorithm(a, b, c, d)

    return t, x, u


def heat_conduct_equation_semi_explicit_scheme(a_coef, T, m, l, n, uxt0, ux0t, uxnt, teta):
    t, x = [], []
    tau = T / m
    h = l / n

    for i in range(m + 1):
        t.append(i * tau)
    for j in range(n + 1):
        x.append(j * h)

    u = np.zeros((len(t), len(x)))
    for i in range(n + 1):
        u[0][i] = uxt0(x[i])
    for j in range(m + 1):
        u[j][0] = ux0t(t[j])
        u[j][len(x) - 1] = uxnt(t[j])

    alpha = a_coef * tau / (h ** 2)
    a, b, c = [0, ], [-(1 + 2 * alpha), ], []

    for i in range(n - 1):
        a.append(teta * alpha)
        b.append(-(1 + 2 * alpha * teta))
        c.append(alpha * teta)
    c.append(0)
    for i in range(m):
        d = np.zeros(n - 1)
        d[0] = -(u[i][1] + teta * alpha * u[i + 1][0] + (1 - teta) * alpha * (u[i][2] - 2 * u[i][1] + u[i][0]))
        for j in range(1, n - 2):
            d[j] = -u[i][j + 1] - (1 - teta) * alpha * (u[i][j + 2] - 2 * u[i][j + 1] + u[i][j])
        d[n - 2] = -(u[i][n - 1] + alpha * u[i + 1][n] + (1 - teta) * alpha * (u[i][n] - 2 * u[i][n - 1] + u[i][n - 2]))
        u[i + 1, 1:u.shape[1] - 1] = tridiagonal_matrix_algorithm(a, b, c, d)

    return t, x, u


def wave_equation_implicit_scheme_common(a_coef, T, m, l, n, uxt0, duxt0dt, ux0t, uxnt, alpha=0, beta=0, gamma=0, delta=0):
    t, x = [], []
    tau = T / m
    h = l / n

    for i in range(m + 1):
        t.append(i * tau)
    for j in range(n + 1):
        x.append(j * h)
    u = np.zeros((len(t), len(x)))
    for i in range(n + 1):
        u[0][i] = uxt0(x[i])
        u[1][i] = uxt0(x[i]) + duxt0dt(x[i]) * tau

    sigma = a_coef * (tau ** 2) / (h ** 2)
    a, b, c = [0, ], [beta - alpha / h, ], [alpha / h, ]

    for i in range(n - 1):
        a.append(sigma)
        b.append(-(1 + 2 * sigma))
        c.append(sigma)
    a.append(-gamma / h)
    b.append(delta + gamma / h)
    c.append(0)

    for i in range(1, m):
        d = np.zeros(n + 1)
        d[0] = ux0t(t[i + 1]) / (beta - alpha / h)
        for j in range(1, n - 1):
            d[j] = u[i - 1][j] - 2 * u[i][j]
        d[n - 1] = uxnt(t[i + 1]) / (delta + gamma / h)
        u[i + 1] = tridiagonal_matrix_algorithm(a, b, c, d)

    return t, x, u
