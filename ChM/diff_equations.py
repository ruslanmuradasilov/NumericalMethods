def method_of_euler(multifunction, n, a, b, y0):
    h = (b - a) / n
    x, y = [], []
    x.append(a)
    y.append(y0)
    for i in range(n):
        x.append(x[i] + h)
        y.append(y[i] + multifunction(x[i], y[i]) * h)
    return x, y


def method_of_euler2(multifunction, n, a, b, y0):
    h = (b - a) / n
    x, y, k = [], [], []
    x.append(a)
    y.append(y0)
    for i in range(n):
        k.append(multifunction(x[i], y[i]) * h)
        x.append(x[i] + h)
        y.append(y[i] + multifunction(x[i] + h / 2, y[i] + k[i] / 2) * h)
    return x, y


def method_of_euler_second_order(multifunction, n, a, b, y0, C2):
    h = (b - a) / n
    x, y, k1, k2 = [], [], [], []
    C1 = 1 - C2
    alpha = 1 / (2 * C2)
    x.append(a)
    y.append(y0)
    for i in range(n):
        k1.append(multifunction(x[i], y[i]) * h)
        k2.append(multifunction(x[i] + alpha * h, y[i] + alpha * k1[i]) * h)
        x.append(x[i] + h)
        y.append(y[i] + C1 * k1[i] + C2 * k2[i])
    return x, y


def method_of_euler_third_order(multifunction, n, a, b, y0, alpha3, alpha2):
    h = (b - a) / n
    x, y, k1, k2, k3 = [], [], [], [], []
    C2 = (alpha3 / 2 - 1 / 3) / (alpha2 * (alpha3 - alpha2))
    C3 = (1 / 2 - C2 * alpha2) / alpha3
    C1 = 1 - C2 - C3
    beta21 = alpha2
    beta32 = 1 / (6 * alpha2 * C3)
    beta31 = alpha3 - beta32
    x.append(a)
    y.append(y0)
    for i in range(n):
        k1.append(multifunction(x[i], y[i]) * h)
        k2.append(multifunction(x[i] + alpha2 * h, y[i] + beta21 * k1[i]) * h)
        k3.append(multifunction(x[i] + alpha3 * h, y[i] + beta31 * k1[i] + beta32 * k2[i]) * h)
        x.append(x[i] + h)
        y.append(y[i] + C1 * k1[i] + C2 * k2[i] + C3 * k3[i])
    return x, y


def method_of_euler_fourth_order(multifunction, n, a, b, y0, alpha2, alpha3, c3=None, c4=None, epsilon=None):
    if epsilon == None:
        h = (b - a) / n
        alpha4 = 1
        delta = alpha2 * alpha3 * alpha4 * (alpha3 - alpha2) * (alpha4 - alpha3) * (alpha4 - alpha2)

        if delta == 0 and alpha3 == 0:
            c1 = 1 / 6 - c3
            c2 = 2 / 3
            c4 = 1 / 6
            beta32 = 1 / (12 * c3)
            beta42 = 3 / 2
            beta43 = 6 * c3
            beta41 = -1 / 2 - 6 * c3
            beta31 = -1 / (12 * c3)
            beta21 = 1 / 2
        elif delta == 0 and alpha2 == alpha3:
            c1 = 1 / 6
            c2 = 2 / 3 - c3
            c4 = 1 / 6
            beta32 = 1 / (6 * c3)
            beta42 = 1 - 3 * c3
            beta43 = 3 * c3
            beta41 = 0
            beta31 = 1 / 2 - 1 / (6 * c3)
            beta21 = 1 / 2
        elif delta == 0 and alpha2 == alpha4:
            c1 = 1 / 6
            c2 = 1 / 6 - c4
            c3 = 2 / 3
            beta32 = 1 / 8
            beta42 = -1 / (6 * c4)
            beta43 = 1 / (3 * c4)
            beta41 = 1 - 1 / (6 * c4)
            beta31 = 3 / 8
            beta21 = 1
        else:
            c2 = (2 * alpha3 - 1) / (12 * alpha2 * (alpha3 - alpha2) * (1 - alpha3))
            c3 = (1 - 2 * alpha2) / (12 * alpha3 * (1 - alpha2) * (1 - alpha3))
            c4 = (6 * alpha2 * alpha3 - 4 * alpha2 - 4 * alpha3 + 3) / (12 * (1 - alpha2) * (1 - alpha3))
            beta42 = -(4 * alpha3 ** 2 - alpha2 - 5 * alpha3 + 2) / (24 * c4 * alpha3 * (alpha3 - alpha2))
            beta43 = (1 - 2 * alpha2) / (24 * c4 * alpha3 * (alpha3 - alpha2))
            beta41 = 1 - beta42 - beta43
            beta21 = alpha2
            beta32 = 1 / (24 * c4 * beta43 * alpha2)
            beta31 = alpha3 - beta32
            c1 = 1 - c2 - c3 - c4

        x, y, k1, k2, k3, k4 = [], [], [], [], [], []

        x.append(a)
        y.append(y0)
        for i in range(n):
            k1.append(multifunction(x[i], y[i]) * h)
            k2.append(multifunction(x[i] + alpha2 * h, y[i] + beta21 * k1[i]) * h)
            k3.append(multifunction(x[i] + alpha3 * h, y[i] + beta31 * k1[i] + beta32 * k2[i]) * h)
            k4.append(multifunction(x[i] + alpha4 * h, y[i] + beta41 * k1[i] + beta42 * k2[i] + beta43 * k3[i]) * h)
            x.append(x[i] + h)
            y.append(y[i] + c1 * k1[i] + c2 * k2[i] + c3 * k3[i] + c4 * k4[i])

        return x, y
    else:
        i = 1
        while True:
            sum = 0
            x1, result1 = method_of_euler_fourth_order(multifunction, i * n, a, b, y0, alpha2, alpha3, c3,
                                                       c4)
            x2, result2 = method_of_euler_fourth_order(multifunction, i * 2 * n, a, b, y0, alpha2, alpha3,
                                                       c3, c4)
            for j in range(len(x1) - 1):
                sum += (result1[j] - result2[2 * j]) ** 2
            if sum < epsilon:
                return x2, result2, i * 2 * n
            i *= 2


def method_of_adams_third_order(multifunction, n, a, b, y0, epsilon=None):
    if epsilon == None:
        h = (b - a) / n
        x, y = [], []
        x.append(a)
        x.append(x[0] + h)
        x.append(x[1] + h)
        y.append(y0)
        y.append(y[0] + multifunction(x[0], y[0]) * h)
        y.append(y[1] + multifunction(x[1], y[1]) * h)
        for i in range(2, n):
            x.append(x[i] + h)
            y.append(y[i] + (h * (23 * multifunction(x[i], y[i]) - 16 * multifunction(x[i - 1], y[i - 1])
                                  + 5 * multifunction(x[i - 2], y[i - 2]))) / 12)
        return x, y
    else:
        i = 1
        while True:
            sum = 0
            x1, result1 = method_of_adams_third_order(multifunction, i * n, a, b, y0)
            x2, result2 = method_of_adams_third_order(multifunction, i * 2 * n, a, b, y0)
            for j in range(len(x1) - 1):
                sum += (result1[j] - result2[2 * j]) ** 2
            if sum < epsilon:
                return x2, result2, i * 2 * n
            i *= 2


def method_of_adams_fourth_order(multifunction, n, a, b, y0, epsilon=None):
    if epsilon == None:
        h = (b - a) / n
        x, y = [], []
        x.append(a)
        x.append(x[0] + h)
        x.append(x[1] + h)
        x.append(x[2] + h)
        y.append(y0)
        y.append(y[0] + multifunction(x[0], y[0]) * h)
        y.append(y[1] + multifunction(x[1], y[1]) * h)
        y.append(y[2] + multifunction(x[2], y[2]) * h)
        for i in range(3, n):
            x.append(x[i] + h)
            y.append(y[i] + (h * (
                    55 * multifunction(x[i], y[i]) - 59 * multifunction(x[i - 1], y[i - 1]) + 37 * multifunction(
                x[i - 2], y[i - 2]) - 9 * multifunction(x[i - 3], y[i - 3]))) / 24)
        return x, y
    else:
        i = 1
        while True:
            sum = 0
            x1, result1 = method_of_adams_fourth_order(multifunction, i * n, a, b, y0)
            x2, result2 = method_of_adams_fourth_order(multifunction, i * 2 * n, a, b, y0)
            for j in range(len(x1) - 1):
                sum += (result1[j] - result2[2 * j]) ** 2
            if sum < epsilon:
                return x2, result2, i * 2 * n
            i *= 2
