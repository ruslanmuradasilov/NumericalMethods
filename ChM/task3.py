import numpy as np
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
    a = np.ones(n) * alpha
    b = np.ones(n) * (-1 - 2 * alpha)
    c = np.ones(n) * alpha
    a[1] = 0
    c[n - 1] = 0

    for i in range(m):
        d = np.zeros(n)
        d[1] = -(u[i][1] + alpha * ux0t(t[i + 1]))
        for j in range(2, n - 1):
            d[j] = -u[i][j]
        d[n - 1] = -(u[i][n - 1] + alpha * uxnt(t[i + 1]))
        u[i + 1, 1:u.shape[1] - 1] = tridiagonal_matrix_algorithm(a[1:], b[1:], c[1:], d[1:])

    return t, x, u


# du/dt = a * d^2u/dx^2
# 0 <= x <= l, 0 <= t <= T, n, m
# u(x, 0) = f(x)
# u(0, t) = fi(t), u(l, t) = psi(t)

# №31 - 15 = 16
def uxt0(x):
    return x * (0.3 + 2 * x)


def ux0t(t):
    return 0


def uxnt(t):
    return 6 * t + 0.9


t, x, u = heat_conduct_equation_implicit_scheme(a_coef=1, T=3, m=300, l=0.6, n=30, uxt0=uxt0, ux0t=ux0t, uxnt=uxnt)
print(u)
xgrid, tgrid = np.meshgrid(x, t)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
Axes3D.plot_surface(ax, X=xgrid, Y=tgrid, Z=u)
plt.show()
