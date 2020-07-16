from ChM import integration, multidimensional_integration
from ChM import diff_equations, systems_of_de, diff_2_equations, partial_diff_equations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from math import sin, cos, sqrt, exp, log, e, fabs, tan, pi, trunc


def function(x):
    return x * exp(x * x)


def multifunction(x, y):
    # return (x * x - 2 * y) / (2 * x)
    # return y - 2 * x / y
    # return x + y
    # return (y - 4 * x ** 3) / x
    # return y * tan(x) + sin(x)
    return -50 * (y + cos(x))


def multifunction1(t, y):
    return y[0] - 2 * y[1]


def multifunction2(t, y):
    return y[0] - y[1]


def accuracy(answer, real):
    sum = 0
    for i in range(len(answer)):
        sum += (answer[i] - real[i]) ** 2
    return sum


                                               # Integration

# print(multidimensional_integration.method_simple(multifunction, 200, 100, -1, 1, 0, 1))


                                              # Diff_equations

# x1, y1 = diff_equations.method_of_runge_kutta_third_order(multifunction, 100, 0, 1.5, 0, 1, 1 / 2)
# x2, y2 = diff_equations.method_of_runge_kutta_third_order(multifunction, 100, 0, 1.5, 0, 1 / 2, 1)
# x3, y3 = diff_equations.method_of_runge_kutta_third_order(multifunction, 100, 0, 1.5, 0, 1, 2)
# x4, y4 = diff_equations.method_of_runge_kutta_third_order(multifunction, 100, 0, 1.5, 0, 2 / 3, 1 / 3)
#
# real = []
# for i in range(101):
#     # real.append(sqrt(1 / (4 - 17 / (4 * x1[i] * x1[i]))))
#     # real.append(sqrt(2 * x1[i] + 1))
#     # real.append(3 * exp(-2 * x[i]) / 4 + x[i] * x[i] / 2 - x[i] / 2 + 1 / 4)
#     # real.append(exp(x[i]) - x[i] - 1)
#     # real.append(1 / (3 * x1[i] / 2 - x1[i] * x1[i] / 2))
#     real.append((1 - cos(2 * x1[i])) / (4*cos(x1[i])))
# print(y1[len(y1) - 1])
# print(y2[len(y2) - 1])
# print(y3[len(y3) - 1])
# print(y4[len(y4) - 1])
# print(f'real: {real[len(real) - 1]}')
# print(accuracy(y1, real), accuracy(y2, real), accuracy(y3, real), accuracy(y4, real))
#
# plt.plot(x1, y1, x2, y2, x3, y3, x4, y4)
# plt.plot(x1, real)
# plt.show()


# n0 = 20
# x1, y1, n = diff_equations.method_of_runge_kutta_fourth_order(multifunction, n0, 0, 10, 0, alpha2=1 / 2, alpha3=0,
#                                                         c3=1 / 6, epsilon=0.000001)
# print(f'n = {n}')
# real = []
# for i in range(n + 1):
#     real.append(((-50) * (50 * cos(x1[i]) + sin(x1[i])) + 2500 * exp((-50) * x1[i])) / 2501)
#     # real.append(sqrt(2 * x1[i] + 1))
#     # real.append((1 - cos(2 * x1[i])) / (4 * cos(x1[i])))
# print(y1[len(y1) - 1])
# print(f'real: {real[len(real) - 1]}')
# print(accuracy(y1, real))
#
# plt.plot(x1, y1)
# plt.plot(x1, real)
# plt.show()


                                       # Systems of diff_equations

# n0 = 20
# t1, y11, y21, n1 = systems_of_de.method_of_euler(multifunction1, multifunction2, n0, 0, 5, 1, 2, epsilon=0.001)
# print(f'n1 = {n1}')
# t2, y12, y22, n2 = systems_of_de.method_of_runge_kutta_second_order(multifunction1, multifunction2, n0, 0, 5, 1, 2,
#                                                               epsilon=0.001)
# print(f'n2 = {n2}')
# t3, y13, y23, n3 = systems_of_de.method_of_runge_kutta_third_order(multifunction1, multifunction2, n0, 0, 5, 1, 2,
#                                                              epsilon=0.001)
# print(f'n3 = {n3}')
# t4, y14, y24, n4 = systems_of_de.method_of_runge_kutta_fourth_order(multifunction1, multifunction2, n0, 0, 5, 1, 2,
#                                                               epsilon=0.001)
# print(f'n4 = {n4}')
# real1, real2 = [], []
# for i in range(n4 + 1):
#     real1.append(cos(t4[i]) - 3 * sin(t4[i]))
#     real2.append(2 * cos(t4[i]) - sin(t4[i]))
# print(y14[len(y14) - 1])
# print(y24[len(y24) - 1])
# print(f'real: {real1[len(real1) - 1]}')
# print(f'real: {real2[len(real2) - 1]}')
# print(accuracy(y14, real1), accuracy(y24, real2))
#
# plt.plot(t4, y14)
# # plt.plot(x1, y2)
# plt.plot(t4, real1)
# # plt.plot(x1, real2)
# plt.show()

# y0 = np.zeros((2, 3))
# y0[0][0] = 1
# y0[1][0] = 2

# multifunctions = np.array([multifunction1, multifunction2])
# # t1, y1, n1 = systems_of_de.method_of_adams_third_order(multifunctions, n0, 0, 5, y0, epsilon=0.0000001)
#
# n0 = 10
# y0 = np.zeros((2, 4))
# y0[0][0] = 1
# y0[1][0] = 2
#
# # t2, y2, n2 = systems_of_de.method_of_adams_fourth_order(multifunctions, n0, 0, 5, y0, epsilon=0.0000001)
# t2, y2, n2 = systems_of_de.method_of_adams_fourth_order_with_runge_kutta_fourth_order(multifunctions, n0, 0, 5, y0, epsilon=0.0000001)
#
# # print(f'n = {n1}')
# print(f'n = {n2}')
# real_y1, real_y2 = [], []
# for i in range(len(t2)):
#     real_y1.append(cos(t2[i]) - 3 * sin(t2[i]))
#     real_y2.append(2 * cos(t2[i]) - sin(t2[i]))
# plt.plot(t2, y2[0])
# plt.plot(t2, real_y1)
# plt.show()


                                       # Diff_equations_2_order

#     y'' + p(x)y' + q(x)y = f(x)
#     alpha11*y(x0) + alpha12*y'(x0) = beta1
#     alpha21*y(xn) + alpha22*y'(xn) = beta2


def p(x):
    return -0.5 * x**2
    # return tan(x)


def q(x):
    return 2
    # return cos(x) ** 2


def f(x):
    return x ** 2
    # return 0


# func_arr = [p, q, f]
#
# n, x, y = diff_2_equations.finite_difference_method(func_arr, n=20, x0=0, xn=1, alpha11=1, alpha12=0, beta1=1,
#                                                     alpha21=1, alpha22=0, beta2=10, eps=0.001)
# real = []
# c = (10 - cos(sin(1))) / sin(sin(1))
# for i in range(len(x)):
#     real.append(cos(sin(x[i])) + c * sin(sin(x[i])))
# plt.plot(x, y)
# plt.plot(x, real)
# plt.show()
#
# x, y = diff_2_equations.finite_difference_method(func_arr, n=3, x0=2, xn=2.3, alpha11=1, alpha12=2, beta1=1,
#                                                     alpha21=1, alpha22=0, beta2=2.15)
# print(y)
# plt.plot(x, y)
# plt.show()

# x0 = 0.6
# xn = 1.9
# h = 0.01
# n = trunc((xn - x0) / h)
# x, y = diff_2_equations.shoot_method(p=p, q=q, f=f, alpha11=1, alpha12=0.7, betta1=2, alpha21=1, alpha22=0, betta2=0.8, x0=x0, xn=xn,
#                     n=n, eps=1e-7)
#
# for i in range(len(x)):
#     print(f'y({x[i]}) = {y[i]}')


#                                         Heat-conduct equation
#
# du/dt = a * d^2u/dx^2
# 0 <= x <= l, 0 <= t <= T, n, m
# u(x, 0) = f(x)
# u(0, t) = fi(t), u(l, t) = psi(t)

def uxt0(x):
    return x * (0.3 + 2 * x)
    # return cos(2*x)


def ux0t(t):
    return 0
    # return 1 - 6*t


def uxnt(t):
    return 6 * t + 0.9
    # return 0.3624


# # t, x, u = partial_diff_equations.heat_conduct_equation_explicit_scheme(a_coef=4, T=10, m=100, l=2, n=20, uxt0 = uxt0, ux0t=ux0t, uxnt=uxnt)
# t, x, u = partial_diff_equations.heat_conduct_equation_implicit_scheme(a_coef=1, T=3, m=300, l=0.6, n=30, uxt0=uxt0, ux0t=ux0t, uxnt=uxnt)
# print(u)
# t, x, u = partial_diff_equations.heat_conduct_equation_implicit_scheme_common(a_coef=1, T=10, m=100, l=pi, n=100, uxt0=uxt0, ux0t=ux0t, uxnt=uxnt, beta=1, delta=1)
# # t, x, u = partial_diff_equations.heat_conduct_equation_implicit_scheme_common(a_coef=1, T=3, m=300, l=0.6, n=30, uxt0=uxt0, ux0t=ux0t, uxnt=uxnt, beta=1, delta=1)
# # t, x, u = partial_diff_equations.heat_conduct_equation_semi_explicit_scheme(a_coef=4, T=10, m=100, l=2, n=20, uxt0=uxt0, ux0t=ux0t, uxnt=uxnt, teta=0.75)
#
# xgrid, tgrid = np.meshgrid(x, t)
#
# ureal = np.zeros((len(t), len(x)))
# for i in range(ureal.shape[0]):
#     for j in range(ureal.shape[1]):
#         ureal[i][j] = exp(-4 * t[i]) * cos(2 * x[j])
#
# fig1 = plt.figure()
# ax = fig1.add_subplot(111, projection='3d')
# Axes3D.plot_surface(ax, X=xgrid, Y=tgrid, Z=u)
# fig2 = plt.figure()
# ax = fig2.add_subplot(111, projection='3d')
# Axes3D.plot_surface(ax, X=xgrid, Y=tgrid, Z=ureal)
# plt.show()


#                                                 Wave equation
#
# d^2u/dt^2 = a * d^2u/dx^2
# 0 <= x <= l, t > 0, n, m
# u(x, 0) = f(x)
# du(x, 0)/dt = F(x)
# u(0, t) = fi(t), u(l, t) = psi(t)

# def uxt0(x):
#     # return x * (pi - x)
#     # return sin(3 * x)
#     # return x * (x + 1)
#     return (1 - x**2) * cos(pi * x)
#
#
# def duxt0dt(x):
#     # return 0
#     # return cos(x)
#     return 2 * x + 0.6
#
#
# def ux0t(t):
#     return 1 + 0.4 * t
#
#
# def uxnt(t):
#     # return 2 * (t + 1)
#     return 0

# t, x, u = partial_diff_equations.wave_equation_implicit_scheme(a_coef=1, T=4, m=400, l=1, n=100, uxt0=uxt0, duxt0dt=duxt0dt, ux0t=ux0t, uxnt=uxnt)
# print(u)
# t, x, u = partial_diff_equations.wave_equation_implicit_scheme_common(a_coef=1, T=4, m=400, l=1, n=100, uxt0=uxt0, duxt0dt=duxt0dt, ux0t=ux0t, uxnt=uxnt, beta=1, delta=1)
# # t, x, u = partial_diff_equations.wave_equation_implicit_scheme_common(a_coef=1, T=0.1, m=400, l=pi, n=1000, uxt0=uxt0, duxt0dt=duxt0dt, ux0t=ux0t, uxnt=uxnt, beta=1, delta=1)
# # t, x, u = partial_diff_equations.wave_equation_implicit_scheme_common(a_coef=1, T=5*pi/18, m=5, l=pi, n=9, uxt0=uxt0, duxt0dt=duxt0dt, ux0t=ux0t, uxnt=uxnt, beta=1, delta=1)
#
# xgrid, tgrid = np.meshgrid(x, t)
# ureal = np.zeros((len(t), len(x)))
# for i in range(ureal.shape[0]):
#     for j in range(ureal.shape[1]):
#         ureal[i][j] = (exp(3 * t[i]) / 2 + exp(-3 * t[i]) / 2) * sin(3 * x[j])
#
# fig1 = plt.figure()
# ax = fig1.add_subplot(111, projection='3d')
# Axes3D.plot_surface(ax, X=xgrid, Y=tgrid, Z=u)
# fig2 = plt.figure()
# ax = fig2.add_subplot(111, projection='3d')
# Axes3D.plot_surface(ax, X=xgrid, Y=tgrid, Z=ureal)
# plt.show()