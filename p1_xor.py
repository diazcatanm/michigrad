import random
from michigrad.nn import Module, Layer
from michigrad.visualize import show_graph


# ------------------------------------------------------------
# MODELO XOR {2 -> 2 -> 1 sin fuc de activacion}
# ------------------------------------------------------------
class XORModule(Module):
    def __init__(self):
        self.l1 = Layer(2, 2, nonlin=False)
        self.l2 = Layer(2, 1, nonlin=False)

    def __call__(self, x):
        h = self.l1(x)
        out = self.l2(h)
        return out[0] if isinstance(out, list) else out

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()

# ------------------------------------------------------------
# VARIABLES GLOBALES
# ------------------------------------------------------------
xor = None

# Set de datos - tabla de verdad XOR
xs = [
    [0.0,0.0],
    [0.0,1.0],
    [1.0,0.0],
    [1.0,1.0]
    ]
ys = [0.0, 1.0, 1.0, 0.0]

# ------------------------------------------------------------
# FUNCIONES
# ------------------------------------------------------------
#Reinicia pesos delm odelo glbal
def reset_modelo(seed=40):
    global xor
    if seed is not None:
        random.seed(seed)
    xor = XORModule()

def entrenamiento(epocas=200, tasa_aprendizaje=0.01, log_cada=5):
    global xor
    if xor is None:
        reset_modelo(seed=40)
    
    loss = None

    for epoca in range(epocas):
        # Forward
        yhats = [xor(x) for x in xs]

        # Loss (MSE promedio)
        loss = sum(((y - yhat) ** 2 for y, yhat in zip(ys, yhats))) / 4

        # Zero grad
        xor.zero_grad()

        # Backward
        loss.backward()

        # Update
        for p in xor.parameters():
            p.data -= tasa_aprendizaje * p.grad

        # Log
        if log_cada and epoca % log_cada == 0:
            print(f"Época {epoca:02d} | Pérdida = {loss.data:.6f}")

    return loss

reset_modelo(seed=40)

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":

    epocas = 200 # seteado luego de evaluar script de calculo
    tasa_aprendizaje = 0.01 # seteado luego de evaluar script de calculo

    reset_modelo(seed=40)

    # ------------------------------------------------------------
    # GRAFICOS PARA 1ER SECUENCIA DE ENTRENAMIENTO
    # ------------------------------------------------------------

    #forward 1er sec de entrenamiento
    yhat0 = [xor(x) for x in xs] #fw
    loss0 = sum(((y - yhat)**2 for y, yhat in zip(ys, yhat0))) / 4 #loss

    #Grafico luego 1er forward
    grafico_fw = show_graph(loss0, format="svg", rankdir="LR")
    grafico_fw.render("tp9_forward", cleanup=True)

    #backward 1er sec de entramiento
    xor.zero_grad()
    loss0.backward()

    #Grafico  luego 1er backward
    grafico_bw = show_graph(loss0, format="svg", rankdir="LR")
    grafico_bw.render("tp9_backpropagation", cleanup=True)

    # ------------------------------------------------------------
    # ENTRENAMIENTO COMPLETO
    # ------------------------------------------------------------

    loss_final = entrenamiento(epocas=epocas, tasa_aprendizaje=tasa_aprendizaje, log_cada=5)


    print("\nLoss final:", loss_final.data)
    print("Predicciones finales:")
    for x, y in zip(xs, ys):
        yhat = xor(x).data
        print(f"x = {x} -> y_hat = {yhat:.4f} | y_true = {y}")
