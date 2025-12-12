from p1_xor import reset_modelo, entrenamiento

# Tasas de aprendizaje a evaluar
tasas = [0.01, 0.1, 1.0]

# Épocas a evaluar  
epocas_list = [25, 50, 100, 200, 500, 1000]

def generar_tablas(archivo_path="tablas_tasas_aprendizaje.txt", seed=40):
    with open(archivo_path, "w", encoding="utf-8") as f:
        for tasa in tasas:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Resultados para tasa de aprendizaje = {tasa}\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"=== Tabla de pérdidas para tasa de aprendizaje = {tasa} ===\n")
            f.write("lr\tÉpocas\tPérdida final\n")

            for ep in epocas_list:
                # modelo nuevo para cada experimento, asi no acumula
                reset_modelo(seed=seed)

                loss_final = entrenamiento(epocas=ep, tasa_aprendizaje=tasa, log_cada=0)
                f.write(f"{tasa}\t{ep}\t{loss_final.data:.6f}\n")


if __name__ == "__main__":
    generar_tablas()
