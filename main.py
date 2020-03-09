from ChM import integration, multidimensional_integration, diff_equations
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, exp, log, e, fabs, tan


def function(x):
    return x * exp(x * x)


def multifunction(x, y):
    # return x * x - 2 * y
    # return y - 2 * x / y
    # return x + y
    return (y - 4 * x ** 3) / x


def accuracy(answer, real):
    sum = 0
    for i in range(len(answer)):
        sum += (answer[i] - real[i]) ** 2
    return sum


# Integration
# print(multidimensional_integration.method_simple(multifunction, 200, 100, -1, 1, 0, 1))

# Diff_equations
x1, y1 = diff_equations.method_of_euler_second_order(multifunction, 100, 1, 2, 2, 5)
x2, y2 = diff_equations.method_of_euler_second_order(multifunction, 100, 1, 2, 2, 1)
x3, y3 = diff_equations.method_of_euler_second_order(multifunction, 100, 1, 2, 2, 3 / 4)
x4, y4 = diff_equations.method_of_euler_second_order(multifunction, 100, 1, 2, 2, 1.25)
real = []
for i in range(101):
    real.append(sqrt(1 / (4 - 17 / (4 * x1[i] * x1[i]))))
    # real.append(sqrt(2 * x1[i] + 1))
    # real.append(3 * exp(-2 * x[i]) / 4 + x[i] * x[i] / 2 - x[i] / 2 + 1 / 4)
    # real.append(exp(x[i]) - x[i] - 1)
print(y1[len(y1) - 1])
print(y2[len(y2) - 1])
print(y3[len(y3) - 1])
print(y4[len(y4) - 1])
print(f'real: {real[len(real) - 1]}')
print(accuracy(y1, real), accuracy(y2, real), accuracy(y3, real), accuracy(y4, real))
plt.plot(x1, y1, x2, y2, x3, y3, x4, y4)
plt.plot(x1, real)
plt.show()
