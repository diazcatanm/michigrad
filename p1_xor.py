import random
from michigrad.nn import MLP, Linear, ReLU, Tanh, Sigmoid
from michigrad.visualize import show_graph


# ------------------------------------------------------------
# VARIABLES GLOBALES
# ------------------------------------------------------------

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

def generar_modelo(seed=40, funcion=Linear):
    if seed is not None:
        random.seed(seed)
    # 2 inputs, salida lineal, y una capa intermedia con función a elección
    return MLP(2, [(2, funcion), (1, Linear)])

def entrenamiento(mlp, epocas=200, tasa_aprendizaje=0.01, log_cada=5):    
    loss = None

    for epoca in range(epocas):
        # Forward. En un modelo perfecto, el resultado sería igual a ys
        yhats = [mlp(x) for x in xs]

        # Loss (MSE promedio)
        loss = sum(((y - yhat) ** 2 for y, yhat in zip(ys, yhats))) / len(ys)

        # Zero grad
        mlp.zero_grad()

        # Backward
        loss.backward()

        # Update
        for p in mlp.parameters():
            p.data -= tasa_aprendizaje * p.grad

        # Log
        if log_cada and epoca % log_cada == 0:
            print(f"Época {epoca:02d} | Pérdida = {loss.data:.6f} | Resultados = [{', '.join(f'{n.data:.3f}' for n in yhats)}]")

    return loss

def punto1():
    print("Punto 1")
    punto_1_o_3(epocas = 200, tasa = 0.01, funcion = Linear, prefijo = "Punto 1 - ", log_cada = 20)

def punto3():
    print("Punto 3")
    punto_1_o_3(epocas = 2000, tasa = 0.3, funcion = Sigmoid, prefijo = "Punto 3 - ", log_cada = 100)

def punto_1_o_3(epocas = 200, log_cada = 5, tasa = 0.01, funcion = Linear, prefijo = " - "):
    xor = generar_modelo(funcion = funcion)

    # ------------------------------------------------------------
    # GRAFICOS PARA 1ER SECUENCIA DE ENTRENAMIENTO
    # ------------------------------------------------------------

    #forward 1er sec de entrenamiento
    yhat0 = [xor(x) for x in xs] #fw
    loss0 = sum(((y - yhat)**2 for y, yhat in zip(ys, yhat0))) / 4 #loss

    #Grafico luego 1er forward
    grafico_fw = show_graph(loss0, format="svg", rankdir="LR")
    grafico_fw.render(prefijo + "forward", cleanup=True)

    #backward 1er sec de entramiento
    xor.zero_grad()
    loss0.backward()

    #Grafico  luego 1er backward
    grafico_bw = show_graph(loss0, format="svg", rankdir="LR")
    grafico_bw.render(prefijo + "backpropagation", cleanup=True)

    # ------------------------------------------------------------
    # ENTRENAMIENTO COMPLETO
    # ------------------------------------------------------------

    loss_final = entrenamiento(xor, epocas=epocas, tasa_aprendizaje=tasa, log_cada=log_cada)


    print("\nLoss final:", loss_final.data)
    print("Predicciones finales:")
    for x, y in zip(xs, ys):
        yhat = xor(x).data
        print(f"x = {x} -> y_hat = {yhat:.4f} | y_true = {y}")

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if __name__ == "__main__":
    punto1()
    punto3()
