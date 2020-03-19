from ChM import integration, multidimensional_integration, diff_equations, systems_of_de
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, exp, log, e, fabs, tan


def function(x):
    return x * exp(x * x)


def multifunction(x, y):
    # return (x * x - 2 * y) / (2 * x)
    # return y - 2 * x / y
    # return x + y
    # return (y - 4 * x ** 3) / x
    # return y * tan(x) + sin(x)
    return -50 * (y + cos(x))


def multifunction1(t, x, y):
    return x - 2 * y


def multifunction2(t, x, y):
    return x - y


def accuracy(answer, real):
    sum = 0
    for i in range(len(answer)):
        sum += (answer[i] - real[i]) ** 2
    return sum


# Integration

# print(multidimensional_integration.method_simple(multifunction, 200, 100, -1, 1, 0, 1))

# Diff_equations

# x1, y1 = diff_equations.method_of_euler_third_order(multifunction, 100, 0, 1.5, 0, 1, 1 / 2)
# x2, y2 = diff_equations.method_of_euler_third_order(multifunction, 100, 0, 1.5, 0, 1 / 2, 1)
# x3, y3 = diff_equations.method_of_euler_third_order(multifunction, 100, 0, 1.5, 0, 1, 2)
# x4, y4 = diff_equations.method_of_euler_third_order(multifunction, 100, 0, 1.5, 0, 2 / 3, 1 / 3)
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
# x1, y1, n = diff_equations.method_of_euler_fourth_order(multifunction, n0, 0, 10, 0, alpha2=1 / 2, alpha3=0,
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

n0 = 20
t1, y11, y21, n1 = systems_of_de.method_of_euler(multifunction1, multifunction2, n0, 0, 5, 1, 2, epsilon=0.001)
print(f'n1 = {n1}')
t2, y12, y22, n2 = systems_of_de.method_of_euler_second_order(multifunction1, multifunction2, n0, 0, 5, 1, 2, epsilon=0.001)
print(f'n2 = {n2}')
t3, y13, y23, n3 = systems_of_de.method_of_euler_third_order(multifunction1, multifunction2, n0, 0, 5, 1, 2, epsilon=0.001)
print(f'n3 = {n3}')
t4, y14, y24, n4 = systems_of_de.method_of_euler_fourth_order(multifunction1, multifunction2, n0, 0, 5, 1, 2, epsilon=0.001)
print(f'n4 = {n4}')
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
