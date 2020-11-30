import numpy as np
import pandas as pd
import math
import random
from sympy import *
from scipy.stats import bernoulli
import matplotlib.pyplot as plt

function = input( "please select the function to minimize (f/g): ")
if function == 'f':
    df = pd.read_csv("asst1_f_data.csv")
    df = df.drop(df.columns[0], axis=1)
    # the const y1 randomly sampled from file
    sub1 = df.at[random.randint(0, 49), 'data1']
    # y2 sampled from bernoulli dist
    p = 0.6
    sub2 = bernoulli.rvs(p, size=1)
    data = [sub1, sub2]

    # #stopping criterion
    precision = 0.000001
    previous_step_size = 1
    max_iters = 2000
    iters = 0
    # #AdaGrad learning scale initialization
    eps = 1e-8
    v_x = 0
    v_y = 0




    #scaling rates (Robbins_Monro & AdaGrad
    rate_RB = [pow(iters + 100, -1), pow(iters + 100, -1)]
    rate_AG = [0.1 / math.sqrt(v_x + eps), 0.1 / math.sqrt(v_y + eps)]
    #rate vector
    rate = [rate_RB[0], rate_RB[1], rate_AG[0], rate_AG[1]]
    # two initial points
    point = np.random.randint(0, 1, size=4)
    for k in range(0, len(data), 1):
        y = data[k]
        if sub1 == 0:
            f = lambda x, y: -x + log(exp(x) + exp(y))
            dx = lambda x, y: -1 + exp(x) / (exp(x) + exp(y))
            dy = lambda x, y: exp(y) / (exp(x) + exp(y))
        else:
            f = lambda x, y: -y + log(exp(x) + exp(y))
            dx = lambda x, y: exp(x) / (exp(x) + exp(y))
            dy = lambda x, y: -1 + exp(y) / (exp(x) + exp(y))

        for j in range(0, len(point), 2):
            cur_x = point[j]
            cur_y = point[j + 1]

            iters = 0
            v_x = 0
            v_y = 0
            #rate_RB = [pow(iters + 100, -1), pow(iters + 100, -1)]
            #rate_AG = [0.1 / math.sqrt(v_x + eps), 0.1 / math.sqrt(v_y + eps)]
            while previous_step_size > precision and iters < max_iters:
                # keep cur_x value in prev_x
                prev_x = cur_x
                # keep cur_y value in prev_x
                prev_y = cur_y
                # Calculating Gx and Gy for AdaGrad
                v_x = v_x + dx(prev_x, prev_y) ** 2
                v_y = v_y + dy(prev_x, prev_y) ** 2
                rate_RB[0] = rate_RB[1] = pow(iters + 100, -1)
                # Calculating AdaGrad rate for x
                rate_AG[0] = 0.1 / math.sqrt(v_x + eps)
                # Calculating AdaGrad rate for y
                rate_AG[1] = 0.1 / math.sqrt(v_y + eps)
                # Leaning rate vector for both learning rates
                rate = [rate_RB[0], rate_RB[1], rate_AG[0], rate_AG[1]]

                for i in range(0, len(rate), 2):
                    r1 = rate[i]
                    r2 = rate[i + 1]

                    # Updating the initial point (x) based on Gradient descent wrt x and learning rate
                    cur_x = cur_x - r1 * np.sum(dx(prev_x, prev_y))
                    # Updating the initial point (y) based on Gradient descent wrt y and learning rate
                    cur_y = cur_y - r2 * np.sum(dy(prev_x, prev_y))
                    # Calculating the minimizing function value
                    # cost = pow((a - cur_x), 2) + (b * (pow((cur_y - pow(cur_x, 2)), 2)))
                    cost = np.sum(f(cur_x, cur_y))
                    # Change in x and y to decide on optimization termination
                    previous_step_size_x = abs(cur_x - prev_x)
                    previous_step_size_y = abs(cur_y - prev_y)
                    print("Iteration:", iters, "scaling factor:", r1, r2, "const value:", data[k],
                          "\nx , y and f:", cur_x, cur_y, cost)
                    iters = iters + 1
            print("The function is minimized at", cur_x, cur_y, cost)

else:
    # setting constants for g
    a1 = int(input("Enter a value for a: "))
    b1 = int(input("Enter a value for b: "))
    a2 = float(input("Enter second value for a: "))
    b2 = float(input("Enter second value for b: "))
    data = [a1, b1, a2, b2]
    # stopping criterion
    precision = 0.000001
    previous_step_size = 1
    max_iters = 2000
    iters = 0
    # AdaGrad learning scale initialization
    eps = 1e-8
    v_x = 0
    v_y = 0

    # derivitive wrt x and y
    dx = lambda x, y: -2 * a + 2 * x - (4 * b * x * (y - pow(x, 2)))
    dy = lambda x, y: 2 * b * (y - pow(x, 2))
    # scaling rates (Robbins_Monro & AdaGrad
    rate_RB = [pow(iters + 100, -1), pow(iters + 100, -1)]
    rate_AG = [0.1 / math.sqrt(v_x + eps), 0.1 / math.sqrt(v_y + eps)]
    # rate vector
    rate = [rate_RB[0], rate_RB[1], rate_AG[0], rate_AG[1]]
    # two initial points
    point = np.random.randint(0, 1, size=4)
    for k in range(0, len(data), 2):
        a = data[k]
        b = data[k + 1]
        for j in range(0, len(point), 2):
            cur_x = point[j]
            cur_y = point[j + 1]

            iters = 0
            v_x = 0
            v_y = 0
            c = np.zeros(max_iters)
            dx = lambda x, y: -2 * a + 2 * x - (4 * b * x * (y - pow(x, 2)))
            dy = lambda x, y: 2 * b * (y - pow(x, 2))
            rate_RB = [pow(iters + 100, -1), pow(iters + 100, -1)]
            rate_AG = [0.1 / math.sqrt(v_x + eps), 0.1 / math.sqrt(v_y + eps)]
            while previous_step_size > precision and iters < max_iters:
                # keep cur_x value in prev_x
                prev_x = cur_x
                # keep cur_y value in prev_x
                prev_y = cur_y
                # Calculating Gx and Gy for AdaGrad
                v_x = v_x + dx(prev_x, prev_y) ** 2
                v_y = v_y + dy(prev_x, prev_y) ** 2
                # Calculating Robbins_Monro rate (same for x and y)
                rate_RB[0] = rate_RB[1] = pow(iters + 100, -1)
                # Calculating AdaGrad rate for x
                rate_AG[0] = 0.1 / math.sqrt(v_x + eps)
                # Calculating AdaGrad rate for y
                rate_AG[1] = 0.1 / math.sqrt(v_y + eps)
                # Leaning rate vector for both learning rates
                rate = [rate_RB[0], rate_RB[1], rate_AG[0], rate_AG[1]]

                for i in range(0, len(rate), 2):
                    r1 = rate[i]
                    r2 = rate[i + 1]
                    # Updating the initial point (x) based on Gradient descent wrt x and learning rate
                    cur_x = cur_x - r1 * (dx(prev_x, prev_y))
                    # Updating the initial point (y) based on Gradient descent wrt y and learning rate
                    cur_y = cur_y - r2 * (dy(prev_x, prev_y))
                    # Calculating the minimizing function value
                    cost = np.sum(pow((a - cur_x), 2) + (b * (pow((cur_y - pow(cur_x, 2)), 2))))
                    c[iters] = cost

                    # Change in x and y to decide on optimization termination
                    previous_step_size_x = abs(cur_x - prev_x)
                    previous_step_size_y = abs(cur_y - prev_y)

                    print("Iteration: ", iters, "scaling factor: ", r1, r2, "a and b : ", a, b,
                          "\nx, y and g:", cur_x, cur_y, cost)
                    iters = iters + 1

            print("The function is minimized at", cur_x, cur_y, cost)


