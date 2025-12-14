# ARCHIVO SOLO DE PRUEBAS

import math
import numpy as np
from michigrad.engine import Value
from michigrad.visualize import show_graph
from michigrad.nn import MLP, Linear, ReLU, Tanh, Sigmoid

def graficar(root, format='svg', rankdir='LR', path='graph'):
    g = show_graph(L, rankdir="TB",format="png")
    g.render("graph", cleanup=True)

def prueba_grafico():
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


# Pruebas de funciones de activación
def prueba_activacion():
    prueba_lineal = MLP(1, [(1, Linear)])
    prueba_lineal.layers[0].neurons[0].w[0] = Value(1.5)
    prueba_lineal.layers[0].neurons[0].b = Value(2)
    r = prueba_lineal([4])
    # 4 * 1.5 + 2 = 6 + 2 = 8
    assert(r.data == 8)

    prueba_relu = MLP(1, [(3, Linear)])
    prueba_relu.layers[0].neurons[0].w[0] = Value(1.0)
    prueba_relu.layers[0].neurons[0].b = Value(1)
    prueba_relu.layers[0].neurons[1].w[0] = Value(1.0)
    prueba_relu.layers[0].neurons[1].b = Value(0)
    prueba_relu.layers[0].neurons[2].w[0] = Value(1.0)
    prueba_relu.layers[0].neurons[2].b = Value(-1)
    r = prueba_relu([4])
    # Pre relu debería dar -1, 0, 1
    # Con relu debería dar 0, 0, 1
    assert([r[0].data == 0, r[1].data == 0, r[2].data == 0])

    prueba_tanh = MLP(1, [(1, Tanh)])
    prueba_tanh.layers[0].neurons[0].w[0] = Value(1.5)
    prueba_tanh.layers[0].neurons[0].b = Value(2)
    r = prueba_tanh([4])
    # Pre tanh debería dar 8
    assert(r.data == math.tanh(8))

    prueba_sigmoid = MLP(1, [(1, Sigmoid)])
    prueba_sigmoid.layers[0].neurons[0].w[0] = Value(1.5)
    prueba_sigmoid.layers[0].neurons[0].b = Value(2)
    r = prueba_sigmoid([4])
    # Pre sigmoid debería dar 8
    assert(r.data == Value(8).sigmoid().data)

prueba_activacion()
def prueba_nn():
    # NN compleja para probar que no se rompa ¿?
    nn = MLP(2, [(4, ReLU), (3, Tanh), (4, Sigmoid), (3, Linear)])
    print(nn([1, 2, 3]))
prueba_nn()
