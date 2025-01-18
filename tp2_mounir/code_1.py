import numpy as np
import matplotlib.pyplot as plt
class ArtificialNeurones:
    def __init__(self, alpha=0.1, n_iter=100):
        self.alpha = alpha
        self.n_iter = n_iter
        self.W = None
        self.b = None
    def model(self, X):
        z = np.dot(X, self.W) + self.b
        A = 1 / (1 + np.exp(-z))  
        return A
    def log_loss(self, A, y):
        cout = (1 / len(y)) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))
        return cout
    def gradients(self, A, X, y):
        dW = (1 / len(y)) * np.dot(X.T, A - y)
        db = (1 / len(y)) * np.sum(A - y)
        return dW, db
    def update(self, dW, db):
        self.W -= self.alpha * dW
        self.b -= self.alpha * db
    def train(self, X, y):
        self.W = np.random.randn(X.shape[1], 1)
        self.b = np.random.randn(1)   
        cout = []    
        for i in range(self.n_iter):
            A = self.model(X)
            cost = self.log_loss(A, y)
            cout.append(cost)  
            dW, db = self.gradients(A, X, y)
            self.update(dW, db)  
        return self.W, self.b, cout

    def predict(self, X):
        # Calcul des prédictions du modèle
        A = self.model(X)
        return A >= 0.5 

    def plot_cost(self, cost):
        plt.plot(cost)
        plt.xlabel('Itérations')
        plt.ylabel('Coût')
        plt.title('Évolution de la fonction Coût')
        plt.show()
X = np.random.randn(100, 2) 
y = np.random.randint(0, 2, size=(100, 1))  
alpha = 0.1  
n_iter = 100  
model = ArtificialNeurones(alpha, n_iter)
W, b, cost = model.train(X, y)
print("Valeur finale de W :", W)
print("Valeur finale de b :", b)
#print("le cout \t\t", cost)
model.plot_cost(cost)

def predict(X, W, b):
    A = 1 / (1 + np.exp(-(np.dot(X, W) + b)))  # Calcul de la sigmoide
    return A >= 0.5  


from sklearn.metrics import accuracy_score 

y_pred = predict(X, W, b)
print("Accuracy:", accuracy_score(y, y_pred))

X_nouveau = np.array([[2, 1], [2, 5], [3, 1.5], [2.5, 1.5]])

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
plt.scatter(X_nouveau[:, 0], X_nouveau[:, 1], c='red')
plt.show()

predictions = predict(X_nouveau, W, b)
print("Prédictions pour les nouveaux exemples:", predictions)

X0 = np.linspace(-1, 4, 100)
X1 = (-W[0] * X0 - b) / W[1]
plt.plot(X0, X1, c='orange', lw=3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
plt.scatter(X_nouveau[:, 0], X_nouveau[:, 1], c='red')
plt.show()

# Calcul des erreurs
errors = np.sum(y_pred.flatten() != y.flatten())
print(f"Nombre d'erreurs: {errors}")

