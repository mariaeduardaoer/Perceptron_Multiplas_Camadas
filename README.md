# Perceptron Múltiplas Camadas

#### Projeto da cadeira de Sistemas Inteligentes

Utilização do algoritmo de aprendizado regra Delta visando a classificação de padrões pelo
Adaline para o seguinte problema:


### Classificação de Padrões

No processamento de bebidas, a aplicação de um determinado conservante é feita em função da combinação de quatro variáveis
por x1 (teor de água), x2 (grau de acidez), x3 (temperatura), x4 (tensão interfacial). Existem três tipos de conservantes 
que podem ser aplicados, os quais são definidos por A, B e C. Realizam-se ensaios em laboratório a fim de especificar qual
tipo deve ser aplicado em uma bebida específica.

Foi aplicada uma rede Perceptron de Múltiplas Camadas (PMC) como classificadora de padrões, visando identificar qual tipo
de conservantes seria introduzido em determinado lote de bebidas. Utilizou-se uma rede Perceptron com quatro entradas, 15
neurônios na camada intermediária e três saídas.

Foi feito o treinamento da rede PMC da seguinte forma:
- Algoritmo de aprendizado: backpropagation
- Inicialização das matrizes de pesos: valores aleatórios entre 0 e 1. 
- Função de ativação: logística (sigmóide)
- Taxa de aprendizado {𝜂}: 0.1
- Precisão {𝜖}: 1e-6.

