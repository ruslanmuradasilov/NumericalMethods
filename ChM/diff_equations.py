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

def method_of_euler_fourth_order(multifunction, n, a, b, y0, alpha2, alpha3, alpha4 = 1):
    h = (b - a) / n
    delta = alpha2 * alpha3 * alpha4 * (alpha3 - alpha2) * (alpha4 - alpha3) * (alpha4 - alpha2)

    if delta == 0:
        return None

    alpha4 = 1
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
