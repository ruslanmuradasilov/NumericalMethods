import numpy as np

def run_method(a, b, c, d):
    answers = []
    alpha = []
    beta = []
    alpha.append(-c[0]/b[0])
    beta.append(d[0]/b[0])
    for i in range(len(d) - 1):
        alpha.append(-c[i + 1]/(a[i + 1]*alpha[i] + b[i + 1]))
        beta.append((-a[i + 1]*beta[i] + d[i + 1])/(a[i + 1]*alpha[i] + b[i + 1]))
        # print(len(alpha))
        # print(len(beta))
    answers.append(beta[len(beta) - 1])
    for i in range(len(alpha) - 1):
        answers.append(answers[i]*alpha[len(alpha) - 2 - i] + beta[len(beta) - 2 - i])
    answers.reverse()
    return answers


def finite_difference_method(func_arr, n, x0, xn, alpha11, alpha12, beta1, alpha21, alpha22, beta2, eps = 0):
    if eps == 0:
        h = (xn - x0) / n
        a, b, c, d, x = [], [], [], [], []
        x.append(x0)
        a.append(0)
        b.append(alpha11 - alpha12/h)
        c.append(alpha12/h)
        d.append(beta1)
        for i in range(1, n):
            x.append(x[i - 1] + h)
            a.append(1 - func_arr[0](x[i])*h/2)
            b.append(-2 + func_arr[1](x[i])*h**2)
            c.append(1 + func_arr[0](x[i])*h/2)
            d.append(func_arr[2](x[i])*h**2)
        x.append(xn)
        a.append(-alpha22/h)
        b.append(alpha21 + alpha22/h)
        c.append(0)
        d.append(beta2)

        answers = run_method(a, b, c, d)

        return x, answers
    else:
        i = 1
        while True:
            sum = 0
            x1, answer1 = finite_difference_method(func_arr, i*n, x0, xn, alpha11, alpha12, beta1, alpha21, alpha22, beta2)
            x2, answer2 = finite_difference_method(func_arr, i*2*n, x0, xn, alpha11, alpha12, beta1, alpha21, alpha22, beta2)
            # print(i*2*n)
            for k in range(len(x1) - 1):
                sum += (answer1[k] - answer2[2 * k]) ** 2
            sum /= len(x1)
            # print (i)
            if sum < eps:
                return i * 2 * n, x2, answer2
            i = i * 2