import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import read_real_dataset, generate_dataset
from visualize_results import visualize_gd_results, draw_function


def predict(x, weights): 
    return np.matmul(x, weights)

def cost_func(y, y_hat):
    return np.mean(np.square(y_hat - y))

def count_gradient(x, weights, y):
    vec_w = np.zeros(weights.shape)
   
    for i, weight in enumerate(weights):
        y_hat = predict(x, weights)
        x_i = x[:, i]
        y_diff = y_hat - y
        vec_w[i] = np.dot(y_diff, x_i) # count partial derivative
   
    return vec_w


def gradient_descent_algorithm(x, weights, y, epochs, lr, regular_param, display_cost = True):
    epoch_results = [] #for visualisation
    
    for epoch in range(epochs):

        y_hat = predict(x, weights)

        epoch_results.append({'y_hat': y_hat, 'w': weights, 'e': epoch}) # save results
        
        if display_cost:
            cost = cost_func(y, y_hat)
            print(cost)

        gradient = count_gradient(x, weights, y)

        weights = weights - lr/len(x) * (gradient + regular_param * weights)
    
    return weights, epoch_results


def main():
    # uncomment for generate toy dataset
    # x, y_ideal, y = generate_dataset(3, 15, True)

    x, y = read_real_dataset() 
    weights = np.random.normal(0, 1, x.shape[1]) # generate random weights ~ N(0, 1)

    epochs, learning_rate, regularization_param = 8, 0.6, 0.1

    weights, epoch_results = gradient_descent_algorithm(x, weights, y, 
                                                        epochs, learning_rate, regularization_param)

    y_hat = predict(x, weights)
    
    print("w_1 (k) - {:.3f}".format(weights[1]))
    print("w_0 (b) - {:.3f}".format(weights[0]))

    visualize_gd_results(epoch_results, learning_rate, x, y)
    plt.show()
    

    draw_function(x, y, as_line=False)
    draw_function(x, y_hat, as_line=True, color='red')
    plt.show()




if __name__ == "__main__":
    main()