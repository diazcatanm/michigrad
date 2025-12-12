import numpy as np
from michigrad.engine import Value
from michigrad.visualize import show_graph

def graficar(root, format='svg', rankdir='LR', path='graph'):
    g = show_graph(L, rankdir="TB",format="png")
    g.render("graph", cleanup=True)
# Definición de los pesos
np.random.seed(42)
W0 = Value(np.random.random(), name='W₀')
W1 = Value(np.random.random(), name='W₁')
b = Value(np.random.random(), name='b')
print(W0)  # imprime Value(data=0.3745401188473625, grad=0, name=W₀)

# definición del dataset de entrenamiento
x0 = Value(.5, name="x₀")
x1 = Value(1., name="x₁")
y = Value(2., name="y")

# forward pass
yhat = x0*W0 + x1*W1 + b
yhat.name = "ŷ"
print(yhat)  # imprime Value(data=1.8699783076450025, grad=0, name=ŷ)

L = (y - yhat) ** 2
L.name = "L"
print(L)  # imprime Value(data=0.016905640482857615, grad=0, name=L)

# backward pass
L.backward()

print(L)  # imprime Value(data=0.016905640482857615, grad=1, name=L)
print(W0)  # imprime Value(data=0.3745401188473625, grad=-0.1300216923549975, name=W₀)

# update de los pesos en la dirección contraria al gradiente de los W

#show_graph(L, rankdir="TB",format="png")

#dot_forward = show_graph(perdida, format="svg", rankdir="LR")
#dot_forward.render("tp9_forward", cleanup=True)
#print("Grafo generado (forward): tp9_forward.svg")
graficar(L)