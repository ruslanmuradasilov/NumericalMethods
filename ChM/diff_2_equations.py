import numpy as np
from math import fabs
from ChM.systems_of_de import method_of_adams_fourth_order_with_runge_kutta_fourth_order


def tridiagonal_matrix_algorithm(a, b, c, d):
    answers = []
    alpha = []
    beta = []
    alpha.append(-c[0] / b[0])
    beta.append(d[0] / b[0])
    for i in range(len(d) - 1):
        alpha.append(-c[i + 1] / (a[i + 1] * alpha[i] + b[i + 1]))
        beta.append((-a[i + 1] * beta[i] + d[i + 1]) / (a[i + 1] * alpha[i] + b[i + 1]))
        # print(len(alpha))
        # print(len(beta))
    answers.append(beta[len(beta) - 1])
    for i in range(len(alpha) - 1):
        answers.append(answers[i] * alpha[len(alpha) - 2 - i] + beta[len(beta) - 2 - i])
    answers.reverse()
    return answers


def finite_difference_method(func_arr, n, x0, xn, alpha11, alpha12, beta1, alpha21, alpha22, beta2, eps=0):
    if eps == 0:
        h = (xn - x0) / n
        a, b, c, d, x = [], [], [], [], []
        x.append(x0)
        a.append(0)
        b.append(alpha11 - alpha12 / h)
        c.append(alpha12 / h)
        d.append(beta1)
        for i in range(1, n):
            x.append(x[i - 1] + h)
            a.append(1 - func_arr[0](x[i]) * h / 2)
            b.append(-2 + func_arr[1](x[i]) * h ** 2)
            c.append(1 + func_arr[0](x[i]) * h / 2)
            d.append(func_arr[2](x[i]) * h ** 2)
        x.append(xn)
        a.append(-alpha22 / h)
        b.append(alpha21 + alpha22 / h)
        c.append(0)
        d.append(beta2)

        answers = tridiagonal_matrix_algorithm(a, b, c, d)

        return x, answers
    else:
        i = 1
        while True:
            sum = 0
            x1, answer1 = finite_difference_method(func_arr, i * n, x0, xn, alpha11, alpha12, beta1, alpha21, alpha22,
                                                   beta2)
            x2, answer2 = finite_difference_method(func_arr, i * 2 * n, x0, xn, alpha11, alpha12, beta1, alpha21,
                                                   alpha22, beta2)
            # print(i*2*n)
            for k in range(len(x1) - 1):
                sum += (answer1[k] - answer2[2 * k]) ** 2
            sum /= len(x1)
            # print (i)
            if sum < eps:
                return i * 2 * n, x2, answer2
            i = i * 2


def adams_method_solve(p, q, f, alpha11, alpha12, betta1, x0, xn, n, t):
    def z_func(x, funcs):
        return funcs[0]

    def temp_func(x, funcs):
        return f(x) - p(x) * funcs[0] - q(x) * funcs[1]

    functions = [temp_func, z_func]
    n0 = n
    y0 = np.zeros((2, 4))
    y0[0][0] = (betta1 - alpha12 * t) / alpha11
    y0[1][0] = t
    # func_ans[0] = y(x) func_ans[1]=z(x)
    return method_of_adams_fourth_order_with_runge_kutta_fourth_order(functions, n0, x0, xn, y0)


def find_interval(p, q, f, alpha11, alpha12, betta1, alpha21, alpha22, betta2, x0, xn, n, t_start, step):
    # Если sgn(G(t)) != sgn(G(-t)) Тогда интревал найден
    current_t = t_start

    def G(y, z):
        return alpha21 * y - alpha22 * z - betta2

    while (True):
        temp, answ_1 = adams_method_solve(p, q, f, alpha11, alpha12, betta1, x0, xn, n, current_t)
        temp, answ_2 = adams_method_solve(p, q, f, alpha11, alpha12, betta1, x0, xn, n, -current_t)
        G_1 = G(answ_1[0][len(answ_1[0]) - 1], answ_1[1][len(answ_1[1]) - 1])
        G_2 = G(answ_2[0][len(answ_2[0]) - 1], answ_2[1][len(answ_2[1]) - 1])
        if ((G_1 > 0 and G_2 < 0) or (G_2 > 0 and G_1 < 0)):
            break
        current_t = current_t + step
    return (current_t, -current_t)


def shoot_method(p, q, f, alpha11, alpha12, betta1, alpha21, alpha22, betta2, x0, xn, n, eps):
    # y'=z
    # z'=f(x)-p(x)z-q(x)y

    # alpha11*y(x0)+alpha12*t=betta1
    # z(x0)=t
    def G(y, z):
        return alpha21 * y - alpha22 * z - betta2

    interval = find_interval(p, q, f, alpha11, alpha12, betta1, alpha21, alpha22, betta2, x0, xn, n, 1, 1)
    while (True):
        current_t = (interval[0] + interval[1]) / 2
        x, answ = adams_method_solve(p, q, f, alpha11, alpha12, betta1, x0, xn, n, current_t)
        G_answ = G(answ[0][len(answ[0]) - 1], answ[1][len(answ[1]) - 1])
        if (fabs(G_answ) <= eps):
            break
        if (G_answ > 0):
            interval = (interval[0], current_t)
        if (G_answ < 0):
            interval = (current_t, interval[1])
    return x, answ[0]
