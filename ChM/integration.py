import numpy as np
import matplotlib.pyplot as plt
from math import sin, sqrt, exp, log, e


def method_of_trapezoid(function, n, a, b, epsilon=-1):
    if epsilon == -1:
        h = (b - a) / n
        sum = 0
        x = a
        for i in range(n + 1):
            if i == 0 or i == n:
                sum += 0.5 * function(x)
            else:
                sum += function(x)
            x += h
        return sum * h
    else:
        i = 1
        while True:
            result1 = method_of_trapezoid(function, i * n, a, b)
            result2 = method_of_trapezoid(function, i * 2 * n, a, b)
            if (abs(result2 - result1) / 3) < epsilon:
                return result2
            i *= 2


def method_of_rectangles(function, n, a, b, epsilon=-1):
    if epsilon == -1:
        h = (b - a) / n
        sum = 0
        x = a + h / 2
        for i in range(n):
            sum += function(x)
            x += h
        return sum * h
    else:
        i = 1
        while True:
            result1 = method_of_rectangles(function, i * n, a, b)
            result2 = method_of_rectangles(function, i * 2 * n, a, b)
            if (abs(result2 - result1) / 7) < epsilon:
                return result2
            i *= 2


def method_of_simpson(function, n, a, b, epsilon=-1):
    if epsilon == -1:
        h = (b - a) / n
        sum = 0
        x = a
        for i in range(n + 1):
            if i == 0 or i == n:
                sum += function(x)
            elif i % 2 == 1:
                sum += 4 * function(x)
            else:
                sum += 2 * function(x)
            x += h
        return sum * h / 3
    else:
        i = 1
        while True:
            result1 = method_of_simpson(function, i * n, a, b)
            result2 = method_of_simpson(function, i * 2 * n, a, b)
            if (abs(result2 - result1) / 31) < epsilon:
                return result2
            i *= 2


def method_of_simpson2(function, n, a, b, epsilon=-1):
    if epsilon == -1:
        h = (b - a) / n
        sum = 0
        x = []
        x.append(a)
        for i in range(1, n + 1):
            x.append(x[i - 1] + h)
        for i in range(2, n - 1, 4):
            sum += 7 * function(x[i - 2]) + 32 * function(x[i - 1]) + 12 * function(x[i]) + 32 * function(
                x[i + 1]) + 7 * function(x[i + 2])
        return sum * 2 * h / 45
    else:
        i = 1
        while True:
            result1 = method_of_simpson2(function, i * n, a, b)
            result2 = method_of_simpson2(function, i * 2 * n, a, b)
            if (abs(result2 - result1) / 31) < epsilon:
                return result2
            i *= 2


def method_of_simpson3(function, n, a, b, epsilon=-1):
    if epsilon == -1:
        h = (b - a) / n
        sum = 0
        x = []
        x.append(a)
        for i in range(1, n + 1):
            x.append(x[i - 1] + h)
        for i in range(3, n - 2, 6):
            sum += function(x[i - 3]) + 5 * function(x[i - 2]) + function(x[i - 1]) + 6 * function(x[i]) + function(
                x[i + 1]) + 5 * function(x[i + 2]) + function(x[i + 3])
        return sum * 3 * h / 10
    else:
        i = 1
        while True:
            result1 = method_of_simpson3(function, i * n, a, b)
            result2 = method_of_simpson3(function, i * 2 * n, a, b)
            if (abs(result2 - result1) / 31) < epsilon:
                return result2
            i *= 2


def method_of_simpson4(function, n, a, b, epsilon=-1):
    if epsilon == -1:
        h = (b - a) / n
        sum = 0
        x = []
        x.append(a)
        for i in range(1, n + 1):
            x.append(x[i - 1] + h)
        for i in range(3, n - 2, 6):
            sum += 41 * function(x[i - 3]) + 216 * function(x[i - 2]) + 27 * function(x[i - 1]) + 272 * function(
                x[i]) + 27 * function(x[i + 1]) + 216 * function(x[i + 2]) + 41 * function(x[i + 3])
        return sum * h / 140
    else:
        i = 1
        while True:
            result1 = method_of_simpson4(function, i * n, a, b)
            result2 = method_of_simpson4(function, i * 2 * n, a, b)
            if (abs(result2 - result1) / 31) < epsilon:
                return result2
            i *= 2


def method_of_gauss(function, n, a, b, epsilon=-1):
    if epsilon == -1:
        h = (b - a) / n
        sum = 0
        x = a + h / 2 - h / (2 * sqrt(3))
        y = a + h / 2 + h / (2 * sqrt(3))
        for i in range(n):
            sum += function(x)
            sum += function(y)
            x += h
            y += h
        return sum * h / 2
    else:
        i = 1
        while True:
            result1 = method_of_gauss(function, i * n, a, b)
            result2 = method_of_gauss(function, i * 2 * n, a, b)
            if (abs(result2 - result1) / 7) < epsilon:
                return result2
            i *= 2
