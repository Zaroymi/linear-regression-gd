from sklearn.linear_model import LinearRegression

from datasets import read_real_dataset
import matplotlib.pyplot as plt


def main():
    x, y = read_real_dataset()

    model = LinearRegression()

    model.fit(x, y)

    y_hat = model.predict(x)

    plt.scatter(x[:, 1], y, s=0.001, color='blue')
    plt.plot(x[:, 1], y_hat, color = 'red')
    
    w_0, w_1 = model.coef_

    print("w_1 (k) - {:.3f}".format(w_1))
    print("w_0 (b) - {:.3f}".format(w_0))
    
    plt.show()



if __name__ == "__main__":
    main()