import importlib
from prueba_calculo_tasaAprendizaje_epocas import calculo_tasaAprendizaje_epocas
import p1_xor


# Tasas de aprendizaje a evaluar
error_list = [0.01, 0.1, 1.0]


def main():
    # Archivo donde escribir los results
    with open("tablas_tasas_aprendizaje.txt", "w", encoding="utf-8") as f:
        for err in error_list:
            # Reset de modelo
            importlib.reload(p1_xor)

            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Resultados para tasa de aprendizaje = {err}\n")
            f.write("=" * 60 + "\n")

            calculo_tasaAprendizaje_epocas(err, archivo=f)


if __name__ == "__main__":
    main()


