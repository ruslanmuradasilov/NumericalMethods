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
        k3.append(multifunction(x[i] + alpha3 * h, y[i] + beta31 * k1[i] + beta32*k2[i]) * h)
        x.append(x[i] + h)
        y.append(y[i] + C1 * k1[i] + C2 * k2[i] + C3 * k3[i])
    return x, y
