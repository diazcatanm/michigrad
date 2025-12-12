import random
from michigrad.engine import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    # nin: n° de inputs
    def __init__(self, nin): 
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)

    def __call__(self, x):
        # Hace todo el cálculo excepto la función de activación
        return sum((wi*xi for wi, xi in zip(self.w, x)), self.b)

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron({len(self.w)})"

class Layer(Module):
    def __init__(self, nin, nout, funcion):
        self.funcion = funcion
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [self.funcion(n(x)) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    # nin: n° de neuronas de input
    # nout: una lista de TUPLAS conteniendo (numero, obj)
    # donde numero es el numero de neuronas de esa capa
    # y obj es la función de activación (definidas abajo)
    # Por ejemplo:
    # MLP(1, [(2, ReLU), (1, Linear)])
    # Define una MLP con una entrada, una capa con 2 neuronas y función ReLU, y una capa final con una neurona y función Linear
    def __init__(self, nin, nouts):
        cantidades, funciones = map(list, zip(*nouts))
        sz = [nin] + cantidades
        self.layers = []

        for i in range(len(nouts)):
            self.layers.append(Layer(sz[i], sz[i+1], funciones[i]))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

# Funciones de activación para usar como objetos.
def Linear(x):
    return x

def ReLU(x):
    return x.relu()

def Tanh(x):
    return x.tanh()

def Sigmoid(x):
    return x.sigmoid()