import numpy as np


def method_of_euler(multifunction1, multifunction2, n, a, b, y10, y20, epsilon=None):
    if epsilon == None:
        h = (b - a) / n
        t, y1, y2 = [], [], []
        t.append(a)
        y1.append(y10)
        y2.append(y20)
        for i in range(n):
            t.append(t[i] + h)
            y1.append(y1[i] + multifunction1(t[i], y1[i], y2[i]) * h)
            y2.append(y2[i] + multifunction2(t[i], y1[i], y2[i]) * h)
        return t, y1, y2
    else:
        i = 1
        while True:
            sum1, sum2 = 0, 0
            x1, result11, result12 = method_of_euler(multifunction1, multifunction2, i * n, a, b, y10, y20)
            x2, result21, result22 = method_of_euler(multifunction1, multifunction2, i * 2 * n, a, b, y10, y20)
            for j in range(len(x1) - 1):
                sum1 += (result11[j] - result21[2 * j]) ** 2
                sum2 += (result12[j] - result22[2 * j]) ** 2
            if sum1 < epsilon and sum2 < epsilon:
                return x2, result21, result22, i * 2 * n
            i *= 2


def method_of_euler_second_order(multifunction1, multifunction2, n, a, b, y10, y20, epsilon=None):
    if epsilon == None:
        h = (b - a) / n
        t, y1, y2, k11, k21, k12, k22 = [], [], [], [], [], [], []
        t.append(a)
        y1.append(y10)
        y2.append(y20)
        for i in range(n):
            t.append(t[i] + h)
            k11.append(multifunction1(t[i], y1[i], y2[i]) * h)
            k21.append(multifunction2(t[i], y1[i], y2[i]) * h)
            k12.append(multifunction1(t[i] + h / 2, y1[i] + k11[i] / 2, y2[i] + k21[i] / 2) * h)
            k22.append(multifunction2(t[i] + h / 2, y1[i] + k11[i] / 2, y2[i] + k21[i] / 2) * h)
            y1.append(y1[i] + 1 / 2 * k11[i] + 1 / 2 * k12[i])
            y2.append(y2[i] + 1 / 2 * k21[i] + 1 / 2 * k22[i])
        return t, y1, y2
    else:
        i = 1
        while True:
            sum1, sum2 = 0, 0
            x1, result11, result12 = method_of_euler_second_order(multifunction1, multifunction2, i * n, a, b, y10, y20)
            x2, result21, result22 = method_of_euler_second_order(multifunction1, multifunction2, i * 2 * n, a, b, y10,
                                                                  y20)
            for j in range(len(x1) - 1):
                sum1 += (result11[j] - result21[2 * j]) ** 2
                sum2 += (result12[j] - result22[2 * j]) ** 2
            if sum1 < epsilon and sum2 < epsilon:
                return x2, result21, result22, i * 2 * n
            i *= 2


def method_of_euler_third_order(multifunction1, multifunction2, n, a, b, y10, y20, epsilon=None):
    if epsilon == None:
        h = (b - a) / n
        t, y1, y2, k11, k21, k12, k22, k13, k23 = [], [], [], [], [], [], [], [], []
        t.append(a)
        y1.append(y10)
        y2.append(y20)
        for i in range(n):
            t.append(t[i] + h)
            k11.append(multifunction1(t[i], y1[i], y2[i]) * h)
            k21.append(multifunction2(t[i], y1[i], y2[i]) * h)
            k12.append(multifunction1(t[i] + h / 2, y1[i] + k11[i] / 2, y2[i] + k21[i] / 2) * h)
            k22.append(multifunction2(t[i] + h / 2, y1[i] + k11[i] / 2, y2[i] + k21[i] / 2) * h)
            k13.append(multifunction1(t[i] + h / 2, y1[i] + k12[i] / 2, y2[i] + k22[i] / 2) * h)
            k23.append(multifunction2(t[i] + h / 2, y1[i] + k12[i] / 2, y2[i] + k22[i] / 2) * h)
            y1.append(y1[i] + 1 / 6 * k11[i] + 1 / 2 * k12[i] + 1 / 3 * k13[i])
            y2.append(y2[i] + 1 / 6 * k21[i] + 1 / 2 * k22[i] + 1 / 3 * k23[i])
        return t, y1, y2
    else:
        i = 1
        while True:
            sum1, sum2 = 0, 0
            x1, result11, result12 = method_of_euler_third_order(multifunction1, multifunction2, i * n, a, b, y10, y20)
            x2, result21, result22 = method_of_euler_third_order(multifunction1, multifunction2, i * 2 * n, a, b, y10,
                                                                 y20)
            for j in range(len(x1) - 1):
                sum1 += (result11[j] - result21[2 * j]) ** 2
                sum2 += (result12[j] - result22[2 * j]) ** 2
            if sum1 < epsilon and sum2 < epsilon:
                return x2, result21, result22, i * 2 * n
            i *= 2


def method_of_euler_fourth_order(multifunction1, multifunction2, n, a, b, y10, y20, epsilon=None):
    if epsilon == None:
        h = (b - a) / n
        t, y1, y2, k11, k21, k12, k22, k13, k23, k14, k24 = [], [], [], [], [], [], [], [], [], [], []
        t.append(a)
        y1.append(y10)
        y2.append(y20)
        for i in range(n):
            t.append(t[i] + h)
            k11.append(multifunction1(t[i], y1[i], y2[i]) * h)
            k21.append(multifunction2(t[i], y1[i], y2[i]) * h)
            k12.append(multifunction1(t[i] + h / 2, y1[i] + k11[i] / 2, y2[i] + k21[i] / 2) * h)
            k22.append(multifunction2(t[i] + h / 2, y1[i] + k11[i] / 2, y2[i] + k21[i] / 2) * h)
            k13.append(multifunction1(t[i] + h / 2, y1[i] + k12[i] / 2, y2[i] + k22[i] / 2) * h)
            k23.append(multifunction2(t[i] + h / 2, y1[i] + k12[i] / 2, y2[i] + k22[i] / 2) * h)
            k14.append(multifunction1(t[i] + h, y1[i] + k13[i] / 2, y2[i] + k23[i] / 2) * h)
            k24.append(multifunction2(t[i] + h, y1[i] + k13[i] / 2, y2[i] + k23[i] / 2) * h)
            y1.append(y1[i] + 1 / 6 * k11[i] + 1 / 3 * k12[i] + 1 / 3 * k13[i] + 1 / 6 * k14[i])
            y2.append(y2[i] + 1 / 6 * k21[i] + 1 / 3 * k22[i] + 1 / 3 * k23[i] + 1 / 6 * k24[i])
        return t, y1, y2
    else:
        i = 1
        while True:
            sum1, sum2 = 0, 0
            x1, result11, result12 = method_of_euler_fourth_order(multifunction1, multifunction2, i * n, a, b, y10, y20)
            x2, result21, result22 = method_of_euler_fourth_order(multifunction1, multifunction2, i * 2 * n, a, b, y10,
                                                                  y20)
            for j in range(len(x1) - 1):
                sum1 += (result11[j] - result21[2 * j]) ** 2
                sum2 += (result12[j] - result22[2 * j]) ** 2
            if sum1 < epsilon and sum2 < epsilon:
                return x2, result21, result22, i * 2 * n
            i *= 2


def method_of_adams_third_order(multifunctions, n, a, b, y0, epsilon=None):
    if epsilon == None:
        h = (b - a) / n
        func_answers = np.zeros((multifunctions.shape[0], n + 1))
        t = [a, ]
        for i in range(n):
            t.append(t[i] + h)
        for i in range(y0.shape[0]):
            for j in range(y0.shape[1]):
                if j != 0:
                    y0[i][j] = y0[i][j - 1] + multifunctions[i](t[i], y0[:, j - 1]) * h
                func_answers[i][j] = y0[i][j]

        for j in range(3, n + 1):
            for i in range(func_answers.shape[0]):
                func_answers[i][j] = func_answers[i][j - 1] + h * (
                        23 * multifunctions[i](t[j - 1], func_answers[:, j - 1])
                        - 16 * multifunctions[i](t[j - 2], func_answers[:, j - 2])
                        + 5 * multifunctions[i](t[j - 3], func_answers[:, j - 3])) / 12
        return t, func_answers
    else:
        i = 1
        while True:
            sum = np.zeros(len(multifunctions))
            t1, result1 = method_of_adams_third_order(multifunctions, i * n, a, b, y0)
            t2, result2 = method_of_adams_third_order(multifunctions, i * 2 * n, a, b, y0)
            for j in range(len(multifunctions)):
                for k in range(len(t1) - 1):
                    sum[j] += (result1[j][k] - result2[j][2 * k]) ** 2
            sum /= n
            if all(sum < epsilon):
                return t2, result2, i * 2 * n
            i *= 2


def method_of_adams_fourth_order(multifunctions, n, a, b, y0, epsilon=None):
    if epsilon == None:
        h = (b - a) / n
        func_answers = np.zeros((len(multifunctions), n + 1))
        t = [a, ]
        for i in range(n):
            t.append(t[i] + h)
        for i in range(y0.shape[0]):
            for j in range(y0.shape[1]):
                if j != 0:
                    y0[i][j] = y0[i][j - 1] + multifunctions[i](t[i], y0[:, j - 1]) * h
                func_answers[i][j] = y0[i][j]

        for j in range(4, n + 1):
            for i in range(func_answers.shape[0]):
                func_answers[i][j] = func_answers[i][j - 1] + h * (
                        55 * multifunctions[i](t[j - 1], func_answers[:, j - 1]) -
                        59 * multifunctions[i](t[j - 2], func_answers[:, j - 2]) +
                        37 * multifunctions[i](t[j - 3], func_answers[:, j - 3]) -
                        9 * multifunctions[i](t[j - 4], func_answers[:, j - 4])) / 24
        return t, func_answers
    else:
        i = 1
        while True:
            sum = np.zeros(len(multifunctions))
            t1, result1 = method_of_adams_fourth_order(multifunctions, i * n, a, b, y0)
            t2, result2 = method_of_adams_fourth_order(multifunctions, i * 2 * n, a, b, y0)
            for j in range(len(multifunctions)):
                for k in range(len(t1) - 1):
                    sum[j] += (result1[j][k] - result2[j][2 * k]) ** 2
            sum /= n
            if all(sum < epsilon):
                return t2, result2, i * 2 * n
            i *= 2


def method_of_adams_fourth_order_with_euler_fourth_order(multifunctions, n, a, b, y0, epsilon=None):
    if epsilon == None:
        h = (b - a) / n
        t, k11, k21, k12, k22, k13, k23, k14, k24 = [], [], [], [], [], [], [], [], []
        t.append(a)
        for i in range(n):
            t.append(t[i] + h)

        func_answers = np.zeros((len(multifunctions), n + 1))
        func_answers[0][0] = y0[0][0]
        func_answers[1][0] = y0[1][0]
        for i in range(y0.shape[1] - 1):
            k11.append(multifunctions[0](t[i], [y0[0][i], y0[1][i]]) * h)
            k21.append(multifunctions[1](t[i], [y0[0][i], y0[1][i]]) * h)
            k12.append(multifunctions[0](t[i] + h / 2, [y0[0][i] + k11[i] / 2, y0[1][i] + k21[i] / 2]) * h)
            k22.append(multifunctions[1](t[i] + h / 2, [y0[0][i] + k11[i] / 2, y0[1][i] + k21[i] / 2]) * h)
            k13.append(multifunctions[0](t[i] + h / 2, [y0[0][i] + k12[i] / 2, y0[1][i] + k22[i] / 2]) * h)
            k23.append(multifunctions[1](t[i] + h / 2, [y0[0][i] + k12[i] / 2, y0[1][i] + k22[i] / 2]) * h)
            k14.append(multifunctions[0](t[i] + h, [y0[0][i] + k13[i] / 2, y0[1][i] + k23[i] / 2]) * h)
            k24.append(multifunctions[1](t[i] + h, [y0[0][i] + k13[i] / 2, y0[1][i] + k23[i] / 2]) * h)
            y0[0][i + 1] = y0[0][i] + 1 / 6 * k11[i] + 1 / 3 * k12[i] + 1 / 3 * k13[i] + 1 / 6 * k14[i]
            y0[1][i + 1] = y0[1][i] + 1 / 6 * k21[i] + 1 / 3 * k22[i] + 1 / 3 * k23[i] + 1 / 6 * k24[i]
            func_answers[0][i + 1] = y0[0][i + 1]
            func_answers[1][i + 1] = y0[1][i + 1]

        print(y0)
        for j in range(4, n + 1):
            for i in range(func_answers.shape[0]):
                func_answers[i][j] = func_answers[i][j - 1] + h * (
                        55 * multifunctions[i](t[j - 1], func_answers[:, j - 1]) -
                        59 * multifunctions[i](t[j - 2], func_answers[:, j - 2]) +
                        37 * multifunctions[i](t[j - 3], func_answers[:, j - 3]) -
                        9 * multifunctions[i](t[j - 4], func_answers[:, j - 4])) / 24
        return t, func_answers
    else:
        i = 1
        while True:
            sum = np.zeros(len(multifunctions))
            t1, result1 = method_of_adams_fourth_order(multifunctions, i * n, a, b, y0)
            t2, result2 = method_of_adams_fourth_order(multifunctions, i * 2 * n, a, b, y0)
            for j in range(len(multifunctions)):
                for k in range(len(t1) - 1):
                    sum[j] += (result1[j][k] - result2[j][2 * k]) ** 2
            sum /= n
            if all(sum < epsilon):
                return t2, result2, i * 2 * n
            i *= 2
