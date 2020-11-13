import pandas as pd
import matplotlib.pyplot as plt
from mlp import MLP

# Database treinamento_MLP:
dataset = pd.read_csv('database/treinamento_MLP.csv')
X = dataset.iloc[:, 0:4].values
d = dataset.iloc[:, 4:7].values

mlp = MLP(X, d, [15, 3])
mlp.train()

values_eqm = mlp.train()

# Database teste_MLP:
dataset = pd.read_csv('database/teste_MLP.csv')

X_teste = dataset[["x1", "x2", "x3", "x4"]].values
Y_teste = dataset[["d1", "d2", "d3"]].values

for i, x in enumerate(X_teste):
    y = mlp.evaluate(x)
    print(f'Expected: {Y_teste[i]}, Output: {y}')

# Plotando grafico do problema
plt.title('Gráfico Em x epoch')
plt.xlabel('EPOCH')
plt.ylabel('EQM')
plt.plot(values_eqm)
plt.show()
