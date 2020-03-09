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
    alpha = 1/(2*C2)
    x.append(a)
    y.append(y0)
    for i in range(n):
        k1.append(multifunction(x[i], y[i]) * h)
        k2.append(multifunction(x[i] + alpha*h, y[i] + alpha*k1[i]) * h)
        x.append(x[i] + h)
        y.append(y[i] + C1*k1[i] + C2*k2[i])
    return x, y
