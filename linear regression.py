import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
class LinearRegression():
    def __init__(self):
        pass
    def load_dataset(self):
        self.data = load_boston()
        self.x = self.data.data
        self.y = self.data.target[:,np.newaxis]
        self.learning_rate = 0.01
        # adding the bias term as our intercept
        # np.hstack concatenats the two nd.arrays column wise
        self.standardize()
        self.x = np.hstack((np.ones((len(self.x), 1)), self.x))
        self.weights = np.zeros((self.x.shape[1],1))
        # print(self.weights)
    def standardize(self):
        mu = np.mean(self.x,0)
        std = np.std(self.x,0)
        self.x = (self.x - mu) / std

        print("done normalizing the data")
    def compute_cost(self):
        pred = self.x.dot(self.weights)
        cost = (1/(2*(len(self.y))))*np.sum((self.y-pred)**2)
        return cost

    def gradient_descent(self, num_of_itteration):
        self.cost_history = np.zeros((num_of_itteration,1))
        n_samples = len(self.y)
        for i in range(num_of_itteration):
            # print((self.learning_rate/n_samples)* self.x.T @ ((self.x @ self.weights) - self.y))
            temp = self.x.T[0] @ ((self.x @ self.weights) - self.y)
            # print(temp, '     ', self.weights[0])
            # the intuition behind the vectorized gradient descent below is that, for each of our weights,
            # we multiply the loss for all our samples with the x value corresponding to that weight.
            # so in our case self.x is a 512,14 matrix, so the transpose of the matrix would be 14, 512. if we want to only update the weight for our first weight, we would do self.x.T[0] which would give
            # us a 1,512 matrix. our loss which is calcualted by using all the weights and samples, would have a dimension of (512,1). after multiplying the two, we would have a 1,1 matrix. and this
            # would be used to  update weight[0].
            self.weights = self.weights - (self.learning_rate/n_samples)* self.x.T @ ((self.x @ self.weights) - self.y)

            self.cost_history[i] = self.compute_cost()

        print("finished")
        self.make_predictions()
    def plot_loss(self, loss_history):
        plt.plot(range(len(loss_history)),loss_history, 'r')
        plt.title('Convergance of cost function')
        plt.xlabel("number of itterations")
        plt.ylabel("Cost")
        plt.show()

    def fit_line(self):
        lr.load_dataset()
        lr.gradient_descent(num_of_itteration= 1500)
        self.plot_loss(self.cost_history)
    def make_predictions(self):
        pred = self.weights.T @ self.x[200]
        print(f'predictions  {pred}    ground truth {self.y[200]}')



if __name__ == '__main__':
    lr = LinearRegression()
    lr.fit_line()