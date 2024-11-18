import numpy as np
import matplotlib as plt

class LogisticRegressionEP34:
    def __init__(self ,lr = 10**-2):
        self.lr = lr
        self.w = None
        self.b = None
        self.N = None
        self.p = None
        self.forecast = None
    
    def init_parameters(self):
        self.w = np.random.randn() * 0.1    
        self.b = np.random.randn() * 0.1 

    def forward(self, X):
        self.f =  1 /(1 + np.exp((-1*(X@self.w+self.b))))

    def predict(self, X):
        return 1 /(1 + np.exp((-1*(X@self.w+self.b))))

    def loss(self, X, y):
        loss_sum = 0
        y_predict = self.predict(X)
        for i in range(len(y)):
            loss_sum += (y[i] * np.log(y_predict[i])) + (1-y[i])*np.log(1 - y_predict[i])
        
        return (-1/self.N)*loss_sum
        
    def backward(self, X, y):
        y_predict = self.predict(X)  
        diff = y_predict - y
        self.l_grad_w = (-1 / self.N) * X.T @ diff
        self.l_grad_b = (-1 / self.N) * np.sum(diff)

    def step(self):
        self.w -= self.lr * self.l_grad_w
        self.b -= self.lr * self.l_grad_b


    def fit(self, X, y, iterations=10000, batch_size=None, show_step=1000, show_line=False):
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a numpy array.")
        
        if not isinstance(y, np.ndarray):
            raise ValueError("y must be a numpy array.")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X and y must match.")
        
        self.init_parameters()  

        indices = np.random.permutation(X.shape[0])
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(iterations):
            
            if batch_size:
                for start_idx in range(0, X.shape[0], batch_size):
                    end_idx = min(start_idx + batch_size, X.shape[0])
                    X_batch = X_shuffled[start_idx:end_idx]
                    y_batch = y_shuffled[start_idx:end_idx]
                    
                    self.forward(X_batch)  
                    dw, db = self.backward(X_batch, y_batch) 
                    self.step(dw, db)  
            else:
                
                self.forward(X_shuffled)
                dw, db = self.backward(X_shuffled, y_shuffled)
                self.step(dw, db)

            
            if i % show_step == 0:
                current_loss = self.loss(X, y)
                print(f"Iteration {i}, Loss: {current_loss}")

                if show_line:
                    self.show_line()


    def show_line(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Plot data points for two classes, as well as the line
        corresponding to the model.
        """
        if (X.shape[1] != 2):
            print("Not plotting: Data is not 2-dimensional")
            return
        idx0 = (y == 0)
        idx1 = (y == 1)
        X0 = X[idx0, :2]
        X1 = X[idx1, :2]
        plt.plot(X0[:, 0], X0[:, 1], 'gx')
        plt.plot(X1[:, 0], X1[:, 1], 'ro')
        min_x = np.min(X, axis=0)
        max_x = np.max(X, axis=0)
        xline = np.arange(min_x[0], max_x[0], (max_x[0] - min_x[0]) / 100)
        yline = (self.w[0]*xline + self.b) / (-self.w[1])
        plt.plot(xline, yline, 'b')
        plt.show()


