import numpy as np
from math import cos, pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def tridiagonal_matrix_algorithm(a, b, c, d):
    answers = []
    alpha = []
    beta = []
    alpha.append(-c[0] / b[0])
    beta.append(d[0] / b[0])
    for i in range(len(d) - 1):
        alpha.append(-c[i + 1] / (a[i + 1] * alpha[i] + b[i + 1]))
        beta.append((-a[i + 1] * beta[i] + d[i + 1]) / (a[i + 1] * alpha[i] + b[i + 1]))
    answers.append(beta[len(beta) - 1])
    for i in range(len(alpha) - 1):
        answers.append(answers[i] * alpha[len(alpha) - 2 - i] + beta[len(beta) - 2 - i])
    answers.reverse()
    return answers


def wave_equation_implicit_scheme(a_coef, T, m, l, n, uxt0, duxt0dt, ux0t, uxnt):
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
    for j in range(m + 1):
        u[j][0] = ux0t(t[j])
        u[j][len(x) - 1] = uxnt(t[j])

    alpha = a_coef * (tau ** 2) / (h ** 2)
    a, b, c = [0, ], [-(1 + 2 * alpha), ], []

    for i in range(n - 2):
        a.append(alpha)
        b.append(-(1 + 2 * alpha))
        c.append(alpha)
    c.append(0)
    for i in range(1, m):
        d = np.zeros(n - 1)
        d[0] = -2 * u[i][1] + u[i - 1][1] - alpha * u[i + 1][0]
        for j in range(1, n - 2):
            d[j] = u[i - 1][j + 1] - 2 * u[i][j + 1]
        d[n - 2] = -2 * u[i][n - 1] + u[i][n - 1] - alpha * uxnt(t[i + 1])
        u[i + 1, 1:u.shape[1] - 1] = tridiagonal_matrix_algorithm(a, b, c, d)

    return t, x, u


# d^2u/dt^2 = a * d^2u/dx^2
# 0 <= x <= l, t > 0, n, m
# u(x, 0) = f(x)
# du(x, 0)/dt = F(x)
# u(0, t) = fi(t), u(l, t) = psi(t)

# №31 - 15 = 16
def uxt0(x):
    return (1 - x ** 2) * cos(pi * x)


def duxt0dt(x):
    return 2 * x + 0.6


def ux0t(t):
    return 1 + 0.4 * t


def uxnt(t):
    return 0


t, x, u = wave_equation_implicit_scheme(a_coef=1, T=4, m=400, l=1, n=100, uxt0=uxt0, duxt0dt=duxt0dt, ux0t=ux0t,
                                        uxnt=uxnt)
print(u)
xgrid, tgrid = np.meshgrid(x, t)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Axes3D.plot_surface(ax, X=xgrid, Y=tgrid, Z=u)
plt.show()
