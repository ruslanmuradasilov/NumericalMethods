import numpy as np
from math import fabs, trunc


def method_of_adams_fourth_order_with_runge_kutta_fourth_order(multifunctions, n, a, b, y0, epsilon=None):
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
            t1, result1 = method_of_adams_fourth_order_with_runge_kutta_fourth_order(multifunctions, i * n, a, b, y0)
            t2, result2 = method_of_adams_fourth_order_with_runge_kutta_fourth_order(multifunctions, i * 2 * n, a, b,
                                                                                     y0)
            for j in range(len(multifunctions)):
                for k in range(len(t1) - 1):
                    sum[j] += (result1[j][k] - result2[j][2 * k]) ** 2
            sum /= n
            if all(sum < epsilon):
                return t2, result2, i * 2 * n
            i *= 2


def adams_method_solve(p, q, f, alpha11, alpha12, beta1, x0, xn, n, t):
    def z_func(x, funcs):
        return funcs[0]

    def temp_func(x, funcs):
        return f(x) - p(x) * funcs[0] - q(x) * funcs[1]

    functions = [z_func, temp_func]
    n0 = n
    y0 = np.zeros((2, 4))
    y0[0][0] = (beta1 - alpha12 * t) / alpha11
    y0[1][0] = t
    return method_of_adams_fourth_order_with_runge_kutta_fourth_order(functions, n0, x0, xn, y0)


def find_interval(p, q, f, alpha11, alpha12, beta1, alpha21, alpha22, beta2, x0, xn, n, t_start, step):
    temp = t_start

    def G(y, z):
        return alpha21 * y + alpha22 * z - beta2

    while True:
        x1, answ_1 = adams_method_solve(p, q, f, alpha11, alpha12, beta1, x0, xn, n, temp)
        x2, answ_2 = adams_method_solve(p, q, f, alpha11, alpha12, beta1, x0, xn, n, -temp)
        G_1 = G(answ_1[0][len(answ_1[0]) - 1], answ_1[1][len(answ_1[1]) - 1])
        G_2 = G(answ_2[0][len(answ_2[0]) - 1], answ_2[1][len(answ_2[1]) - 1])
        if (G_1 > 0 and G_2 < 0) or (G_2 > 0 and G_1 < 0):
            break
        temp = temp + step
    return (temp, -temp)


def shoot_method(p, q, f, alpha11, alpha12, beta1, alpha21, alpha22, beta2, x0, xn, n, eps):
    def G(y, z):
        return alpha21 * y + alpha22 * z - beta2

    interval = find_interval(p, q, f, alpha11, alpha12, beta1, alpha21, alpha22, beta2, x0, xn, n, 1, 1)
    while True:
        temp = (interval[0] + interval[1]) / 2
        x, answ = adams_method_solve(p, q, f, alpha11, alpha12, beta1, x0, xn, n, temp)
        G_answ = G(answ[0][len(answ[0]) - 1], answ[1][len(answ[1]) - 1])
        if fabs(G_answ) <= eps:
            break
        if G_answ > 0:
            interval = (interval[0], temp)
        if G_answ < 0:
            interval = (temp, interval[1])
    return x, answ[0]


#     y'' + p(x)y' + q(x)y = f(x)
#     alpha11*y(x0) + alpha12*y'(x0) = beta1
#     alpha21*y(xn) + alpha22*y'(xn) = beta2

# №15
def p(x):
    return -0.5 * x ** 2


def q(x):
    return 2


def f(x):
    return x ** 2


x0 = 0.6
xn = 1.9
h = 0.01
n = trunc((xn - x0) / h)
x, y = shoot_method(p=p, q=q, f=f, alpha11=1, alpha12=0.7, beta1=2, alpha21=1, alpha22=0, beta2=0.8, x0=x0, xn=xn,
                    n=n, eps=1e-7)

for i in range(len(x)):
    print(f'y({x[i]}) = {y[i]}')

# Output
# y(0.6) = 0.21803093552589425
# y(0.6100775193798449) = 0.22023740045860318
# y(0.6201550387596898) = 0.22246619473410711
# y(0.6302325581395347) = 0.22471754432452204
# y(0.6403100775193796) = 0.2269935793049817
# y(0.6503875968992245) = 0.22929269159566654
# y(0.6604651162790695) = 0.2316150686370162
# y(0.6705426356589144) = 0.23396097427594764
# y(0.6806201550387593) = 0.2363306407759975
# y(0.6906976744186042) = 0.2387243082173056
# y(0.7007751937984491) = 0.24114221986124562
# y(0.710852713178294) = 0.2435846212508156
# y(0.7209302325581389) = 0.24605176042531463
# y(0.7310077519379838) = 0.24854388794135335
# y(0.7410852713178288) = 0.25106125689257197
# y(0.7511627906976737) = 0.2536041229360281
# y(0.7612403100775186) = 0.2561727443182208
# y(0.7713178294573635) = 0.25876738190128873
# y(0.7813953488372084) = 0.261388299189504
# y(0.7914728682170533) = 0.2640357623560336
# y(0.8015503875968982) = 0.266710040269971
# y(0.8116279069767431) = 0.269411404523642
# y(0.821705426356588) = 0.27214012946018645
# y(0.831782945736433) = 0.2748964922014203
# y(0.8418604651162779) = 0.277680772675979
# y(0.8519379844961228) = 0.2804932536477464
# y(0.8620155038759677) = 0.2833342207445715
# y(0.8720930232558126) = 0.28620396248727586
# y(0.8821705426356575) = 0.2891027703189552
# y(0.8922480620155024) = 0.29203093863457746
# y(0.9023255813953474) = 0.2949887648108807
# y(0.9124031007751923) = 0.297976549236574
# y(0.9224806201550372) = 0.300994595342844
# y(0.9325581395348821) = 0.30404320963417053
# y(0.942635658914727) = 0.30712270171945444
# y(0.9527131782945719) = 0.3102333843434606
# y(0.9627906976744168) = 0.31337557341857936
# y(0.9728682170542617) = 0.31654958805690975
# y(0.9829457364341067) = 0.31975575060266775
# y(0.9930232558139516) = 0.3229943866649226
# y(1.0031007751937966) = 0.3262658251506646
# y(1.0131782945736416) = 0.32957039829820833
# y(1.0232558139534866) = 0.33290844171093364
# y(1.0333333333333317) = 0.3362802943913686
# y(1.0434108527131767) = 0.33968629877561773
# y(1.0534883720930217) = 0.3431268007681389
# y(1.0635658914728667) = 0.3466021497768724
# y(1.0736434108527118) = 0.3501126987487259
# y(1.0837209302325568) = 0.3536588042054186
# y(1.0937984496124018) = 0.3572408262796888
# y(1.1038759689922468) = 0.3608591287518679
# y(1.1139534883720918) = 0.3645140790868249
# y(1.1240310077519369) = 0.3682060484712853
# y(1.134108527131782) = 0.3719354118515275
# y(1.144186046511627) = 0.37570254797146163
# y(1.154263565891472) = 0.3795078394110935
# y(1.164341085271317) = 0.38335167262537856
# y(1.174418604651162) = 0.387234437983469
# y(1.184496124031007) = 0.39115652980835885
# y(1.194573643410852) = 0.39511834641693006
# y(1.204651162790697) = 0.3991202901604046
# y(1.214728682170542) = 0.40316276746520624
# y(1.224806201550387) = 0.40724618887423586
# y(1.2348837209302321) = 0.4113709690885653
# y(1.2449612403100772) = 0.4155375270095531
# y(1.2550387596899222) = 0.41974628578138706
# y(1.2651162790697672) = 0.4239976728340576
# y(1.2751937984496122) = 0.42829211992676625
# y(1.2852713178294572) = 0.43263006319177416
# y(1.2953488372093023) = 0.4370119431786943
# y(1.3054263565891473) = 0.4414382048992326
# y(1.3155038759689923) = 0.4459092978723819
# y(1.3255813953488373) = 0.4504256761700743
# y(1.3356589147286824) = 0.4549877984632949
# y(1.3457364341085274) = 0.45959612806866357
# y(1.3558139534883724) = 0.464251132995488
# y(1.3658914728682174) = 0.4689532859932934
# y(1.3759689922480625) = 0.4737030645998337
# y(1.3860465116279075) = 0.4785009511895889
# y(1.3961240310077525) = 0.4833474330227538
# y(1.4062015503875975) = 0.4882430022947227
# y(1.4162790697674426) = 0.49318815618607564
# y(1.4263565891472876) = 0.4981833969130703
# y(1.4364341085271326) = 0.503229231778646
# y(1.4465116279069776) = 0.508326173223944
# y(1.4565891472868226) = 0.5134747388803496
# y(1.4666666666666677) = 0.518675451622061
# y(1.4767441860465127) = 0.5239288396191915
# y(1.4868217054263577) = 0.5292354363914088
# y(1.4968992248062027) = 0.5345957808621177
# y(1.5069767441860478) = 0.5400104174131918
# y(1.5170542635658928) = 0.5454798959402593
# y(1.5271317829457378) = 0.551004771908549
# y(1.5372093023255828) = 0.556585606409302
# y(1.5472868217054279) = 0.562222966216754
# y(1.5573643410852729) = 0.5679174238456961
# y(1.567441860465118) = 0.5736695576096171
# y(1.577519379844963) = 0.5794799516794362
# y(1.587596899224808) = 0.585349196142829
# y(1.597674418604653) = 0.5912778870641557
# y(1.607751937984498) = 0.5972666265449957
# y(1.617829457364343) = 0.6033160227852952
# y(1.627906976744188) = 0.6094266901451344
# y(1.637984496124033) = 0.6155992492071204
# y(1.648062015503878) = 0.6218343268394115
# y(1.6581395348837231) = 0.6281325562593808
# y(1.6682170542635681) = 0.6344945770979232
# y(1.6782945736434132) = 0.6409210354644153
# y(1.6883720930232582) = 0.6474125840123319
# y(1.6984496124031032) = 0.6539698820055284
# y(1.7085271317829482) = 0.6605935953851931
# y(1.7186046511627933) = 0.6672843968374789
# y(1.7286821705426383) = 0.674042965861819
# y(1.7387596899224833) = 0.6808699888399354
# y(1.7488372093023283) = 0.6877661591055458
# y(1.7589147286821734) = 0.6947321770147766
# y(1.7689922480620184) = 0.7017687500172891
# y(1.7790697674418634) = 0.7088765927281264
# y(1.7891472868217084) = 0.7160564270002875
# y(1.7992248062015535) = 0.7233089819980368
# y(1.8093023255813985) = 0.7306349942709561
# y(1.8193798449612435) = 0.7380352078287464
# y(1.8294573643410885) = 0.7455103742167876
# y(1.8395348837209335) = 0.753061252592463
# y(1.8496124031007786) = 0.7606886098022583
# y(1.8596899224806236) = 0.768393220459639
# y(1.8697674418604686) = 0.7761758670237191
# y(1.8798449612403136) = 0.784037339878725
# y(1.8899224806201587) = 0.791978437414265
# y(1.9000000000000037) = 0.7999999661064112
