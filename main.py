from ChM import integration, multidimensional_integration, diff_equations
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
x1, y1, n = diff_equations.method_of_adams_fourth_order(multifunction, n0, 0, 10, 0, epsilon=0.000001)
print(f'n = {n}')
real = []
for i in range(n + 1):
    real.append((-50 * (50 * cos(x1[i]) + sin(x1[i])) + 2500 * exp(-50 * x1[i])) / 2501)
    # real.append(sqrt(2 * x1[i] + 1))
    # real.append((1 - cos(2 * x1[i])) / (4 * cos(x1[i])))
print(y1[len(y1) - 1])
print(f'real: {real[len(real) - 1]}')
print(accuracy(y1, real))

plt.plot(x1, y1)
plt.plot(x1, real)
plt.show()
