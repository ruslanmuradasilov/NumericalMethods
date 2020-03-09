def method_simple(multifunction, n1, n2, a, b, c, d):
    h1 = (b - a) / n1
    h2 = (d - c) / n2
    sum, x, y = 0, a + h1/2, c + h2/2
    for i in range(n1):
        y = c + h2/2
        for j in range(n2):
            if y >= (x * x):
                sum += multifunction(x, y)
            y += h2
        x += h1
    return sum * h1 * h2
