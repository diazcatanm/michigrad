from p1_xor import forwardPass_loss, bucle_entrenamiento

def calculo_tasaAprendizaje_epocas(lr=0.1, archivo=None):

    epocas_list = [10, 20, 30, 40, 50]

    def log(msg: str) -> None:
        if archivo is None:
            print(msg)
        else:
            archivo.write(msg + "\n")

    log(f"\n=== Tabla de pérdidas para tasa de aprendizaje = {lr} ===")
    log("lr\tÉpocas\tPérdida final")

    for ep in epocas_list:
        bucle_entrenamiento(epocas=ep, tasa_aprendizaje=lr)
        perdida_final, _ = forwardPass_loss()
        log(f"{lr}\t{ep}\t{perdida_final.data:.6f}")


if __name__ == "__main__":
    calculo_tasaAprendizaje_epocas()
