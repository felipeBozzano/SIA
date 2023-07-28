import numpy as np
import random

class neurona():
    def __init__(self, pesos_iniciales: np.array, taza_de_aprendizaje: float) -> None:
        self.pesos = pesos_iniciales
        self.taza_de_aprendizaje = taza_de_aprendizaje
        self.error: float = None


    def entrenar_neurona(self, entradas_de_entrenamiento: np.array, salidas_de_entrenamiento: np.array):
        for entrada, salida in zip(entradas_de_entrenamiento, salidas_de_entrenamiento):
            producto_punto = self.agregacion(entrada)
            salida_obtenida = self.activacion(producto_punto)
            self.error = salida - salida_obtenida
            self.pesos = self.pesos + self.taza_de_aprendizaje * self.error * entrada
    

    def agregacion(self, entrada):
        return np.dot(self.pesos, entrada)
    

    def activacion(self, resultado) -> int:
        return 1 if resultado > 0 else -1
    

    def predecir(self, entrada: np.array) -> int:
        producto_punto = self.agregacion(entrada)
        salida = self.activacion(producto_punto)
        return salida


    def __str__(self) -> str:
        return f"Neurona -> pesos: {self.pesos} || taza de aprendizaje: {self.taza_de_aprendizaje} || error: {self.error}"


class perceptron():
    def __init__(self, neurona_1: neurona, neurona_2: neurona) -> None:
        self.neuronas = [neurona_1, neurona_2]


    def entrenar(self, entradas_de_entrenamiento: np.array, salidas_de_entrenamiento: np.array) -> None:
        salidas_de_entrenamiento = salidas_de_entrenamiento.T
        for _ in range(100):
            for neurona, salidas in zip(self.neuronas, salidas_de_entrenamiento):
                neurona.entrenar_neurona(entradas_de_entrenamiento, salidas)
    

    def entrenar_neurona(self, neurona: int, entradas_de_entrenamiento: np.array, salida_de_entrenamiento: np.array) -> None:
        self.neuronas[neurona].entrenar_neurona(entradas_de_entrenamiento, salida_de_entrenamiento)

    
    def predecir(self, entradas: np.array):
        for entrada in entradas:
            imprimir = f"{entrada} - ["
            for neurona in self.neuronas:
                imprimir += str(neurona.predecir(entrada)) + " "
            imprimir += "]"
            print(imprimir)
    

    def __str__(self) -> str:
        imprimir = "PERCEPTRON\n"
        for neurona in self.neuronas:
            imprimir += f"{neurona}\n"
        return imprimir
    
# python3 -i perceptron.py
if __name__ == "__main__":
    entradas_de_entrenamiento = np.array([[1, 1, 1, 1], [-1 , 1, 1, 1], [1, 1, -1, -1], [-1, -1, -1, -1], [1, -1, 1, 1], [1, 1, -1, 1], [1, 1, 1, -1]])
    salidas_de_entrenamiento = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1], [1, -1], [-1, 1], [1, -1]])
    entradas_de_testeo =np.array([[1, -1, -1, -1],
                                  [1, -1, 1, -1],
                                  [-1, -1, 1, 1],
                                  [-1, 1, -1, 1],
                                  [-1, -1, 1, -1],
                                  [-1, 1, 1, -1],
                                  [1, -1, -1, 1],
                                  [-1, -1, -1, 1],
                                  [-1, 1, -1, -1]])

    for _ in range(1000):
        # pesos_random_neurona_1 = random.np.array(1,4, min=-1, max=1)
        # pesos_random_neurona_2 = random.np.array(1,4, min=-1, max=1)
        # taza_aprendizaje_1 = random.int(min=0.1, max=0.9)
        # taza_aprendizaje_2 = random.int(min=0.1, max=0.9)

        neurona_1 = neurona(np.array([-1.0, 0.4, 0.8, 0.9]), 0.3)
        neurona_2 = neurona(np.array([-0.3, 0.2, 0.5, -0.4]), 0.3)
        perceptron_actual = perceptron(neurona_1, neurona_2)

        perceptron_actual.entrenar(entradas_de_entrenamiento, salidas_de_entrenamiento)
        print(perceptron_actual)

        perceptron_actual.predecir(entradas_de_testeo)

        # array_de_salida = perceptron_actual.predecir(entradas_de_testeo)
        # comparar salidas correctas contra array_de_salida
        # si esta ok cortas
        # sino ok, que siga otra iteracion


"""
SALIDAS CORRECTAS
[ 1 -1 -1 -1] - [1 -1 ]
[ 1 -1  1 -1] - [1 -1 ]
[-1 -1  1  1] - [1 1 ]
[-1  1 -1  1] - [-1 1 ]
[-1 -1  1 -1] - [1 -1 ]
[-1  1  1 -1] - [-1 -1 ]
[ 1 -1 -1  1] - [1 1 ]
[-1 -1 -1  1] - [1 1 ]
[-1  1 -1 -1] - [-1 1 ]
"""
