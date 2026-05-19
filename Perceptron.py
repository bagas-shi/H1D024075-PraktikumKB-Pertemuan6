import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, alpha=0.1, epoch=10):
        self.alpha = alpha
        self.epoch = epoch

    def weighted_sum(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.weighted_sum(X) >= 0.0, 1, -1)

    def plot_decision_boundary(self, X, t, epoch):
        plt.scatter(X[:, 0], X[:, 1], c=t.ravel(), marker='o',
                    edgecolors='k', cmap=plt.cm.RdYlBu)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1


        x_vals = np.linspace(x_min, x_max, 100)
        y_vals = -(self.w_[0] + self.w_[1] * x_vals) / self.w_[2]

        plt.plot(x_vals, y_vals, 'b-', label=f'Decision boundary (Epoch {epoch+1})')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title(f"Decision Boundary Pada Epoch {epoch+1}")
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.show()

    def fit(self, X, t):
        self.w_ = np.zeros(1 + X.shape[1])

        with open("HasilPerceptron.txt", "w") as f:
            f.write("Masalah OR dengan Perceptron\n")
            f.write("----------------------------\n")
            f.write(f"Input :\n{X}\n")
            f.write(f"Target:\n{t}\n")
            f.write(f"Bobot awal : {self.w_[1:]}\n")
            f.write(f"Bias awal : {self.w_[0]}\n")
            f.write(f"Learning rate : {self.alpha}\n")
            f.write(f"Max Epoch : {self.epoch}\n")

            for epoch in range(self.epoch):
                f.write(f"\nEpoch {epoch + 1}/{self.epoch}\n")
                f.write("----------\n")
                error = np.array([])

                for xi, target in zip(X, t):
                    y_pred = self.predict(xi)

                    error = np.append(error, target - y_pred)

                    update = self.alpha * error[-1]

                    self.w_[1:] += update * xi

                    self.w_[0] += update

                    f.write(f"Input: {xi}, Target: {target}, Predict: {y_pred}, "
                            f"Error: {error[-1]}, Bobot: {self.w_[1:]}, Bias: {self.w_[0]}\n")

                self.plot_decision_boundary(X, t, epoch)

                f.write(f"Sum Square Error(SSE): {sum(error ** 2)}\n")

                if sum(error ** 2) == 0 or epoch + 1 == self.epoch:
                    f.write("-" * 80 + "\n")
                    f.write(f"Pelatihan berhenti pada epoch ke-{epoch + 1} karena ")
                    f.write("Sum Square Error(SSE) mencapai target.\n" if epoch + 1 != self.epoch
                            else "max epoch tercapai.\n")
                    f.write(f"\nBobot akhir :{self.w_[1:]}\n")
                    f.write(f"Bias akhir :{self.w_[0]}")
                    break