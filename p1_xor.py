from michigrad.nn import MLP
from michigrad.visualize import show_graph

# Set de datos - tabla de verdad XOR
#
# p | q | xor
#---|---|----
# V | V | F
# V | F | V
# F | V | V
# F | F | F

x_entradas = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
]

# Salida segun xor:
# 0 -> [1, 0]
# 1 -> [0, 1]
y_clases = [
    [1.0, 0.0],  # XOR(0,0) = 0
    [0.0, 1.0],  # XOR(0,1) = 1
    [0.0, 1.0],  # XOR(1,0) = 1
    [1.0, 0.0],  # XOR(1,1) = 0
]

# ------------------------------------------------------------
# modelo disenado con una capa lineal con 2 neuronas
# MLP(2, [2]) = entrada de 2, una capa con 2 neuronas sin func de activacion
# ------------------------------------------------------------
modelo = MLP(2, [2])


# ------------------------------------------------------------
# func para fordward pass + calc de loss
# ------------------------------------------------------------
def forwardPass_loss():
    """
    Realiza el forward del modelo sobre las 4 entradas de XOR
    y calcula la pérdida total como suma del error cuadratico medio () sobre todos los ejemplos.
    Devuelve:
      - perdida_total: Value escalar
      - predicciones: lista de listas [Value, Value] (una por ejemplo)
    """
    # 1) Forward: predicciones para cada entrada
    predicciones = [modelo(x) for x in x_entradas]

    # 2) Pérdida: MSE por ejemplo y luego suma total
    perdidas_por_ejemplo = []
    for pred, clase_real in zip(predicciones, y_clases):
        # MSE: sum_i (pred_i - clase_real_i)^2
        terminos = [(p - t) ** 2 for p, t in zip(pred, clase_real)]
        perdida_ejemplo = sum(terminos)
        perdidas_por_ejemplo.append(perdida_ejemplo)

    perdida_total = sum(perdidas_por_ejemplo)
    return perdida_total, predicciones


# ------------------------------------------------------------
# func para graficar el grafo de 1 sec de entrenamiento
# ------------------------------------------------------------
def graficar_secuencia_entrenamiento():

    # 1) FORWARD + LOOS
    perdida, _ = forwardPass_loss()
    print(f"Pérdida inicial (antes de entrenar): {perdida.data:.6f}")

    # 2) Grafico dsp de FORWARD
    dot_forward = show_graph(perdida, format="svg", rankdir="LR")
    dot_forward.render("tp9_forward", cleanup=True)
    print("Grafo generado (forward): tp9_forward.svg")

    # 3) ZERO GRAD
    modelo.zero_grad()

    # 4) BACKWARD/BACKPROPAGATION
    perdida.backward()

    # 5) Grafico dsp de BACKPROPAGATION
    dot_backward = show_graph(perdida, format="svg", rankdir="LR")
    dot_backward.render("tp9_backpropagation", cleanup=True)
    print("Grafo generado (backpropagation): tp9_backpropagation.svg")


# ------------------------------------------------------------
# BUCLE DE ENTRENAMIENTO 
#   1. Forward
#   2. Loss
#   3. Zero grad
#   4. Backward
#   5. Update
# ------------------------------------------------------------

def bucle_entrenamiento(epocas=30, tasa_aprendizaje=0.1):
    for epoca in range(epocas):

        # 1) FORWARD + PÉRDIDA
        perdida_total, _ = forwardPass_loss()

        # 2) ZERO GRAD
        modelo.zero_grad()

        # 3) BACKWARD
        perdida_total.backward()

        # 4) UPDATE de parámetros
        for p in modelo.parameters():
            p.data += -tasa_aprendizaje * p.grad

        # 5) log de que la pérdida baja
        if epoca % 5 == 0:
            print(f"Época {epoca:02d} | Pérdida = {perdida_total.data:.6f}")


# ------------------------------------------------------------
# func para interpretar la clase predicha a partir de [y0, y1]
# ------------------------------------------------------------
def predecir_clase(prediccion):
    """
    Recibe: prediccion = [Value, Value]
    Devuelve:
      - clase_predicha: 0 o 1
      - valores: lista [y0, y1] (floats) para imprimir
    """
    valores = [v.data for v in prediccion]
    clase_predicha = 0 if valores[0] >= valores[1] else 1
    return clase_predicha, valores


# ------------------------------------------------------------
# prog ppal
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Estructura del modelo:")
    print(modelo)

    graficar_secuencia_entrenamiento()

    bucle_entrenamiento(epocas=30, tasa_aprendizaje=0.1)

    print("\n=== Predicciones finales del modelo lineal sobre XOR ===")
    for x, clase_real in zip(x_entradas, y_clases):
        pred = modelo(x)
        clase_predicha, valores = predecir_clase(pred)
        print(f"x = {x} -> pred = {valores} -> clase_predicha = {clase_predicha} | clase_real = {clase_real}")
