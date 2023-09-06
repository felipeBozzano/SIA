import numpy as np
import random


class neurona():
    def __init__(self, pesos_iniciales: list, taza_de_aprendizaje: float) -> None:
        self.pesos = pesos_iniciales
        self.taza_de_aprendizaje = taza_de_aprendizaje
        self.error: float = None
        self.errores_retropropagados = list()
    

    def agregacion(self, entrada):
        return np.dot(self.pesos, entrada)
    

    def activacion(self, resultado) -> int:
        try:
            return 1/(1 + np.exp(-resultado))
        except RuntimeWarning as e:
            print(e)
            return 1
    

    def predecir(self, entrada: np.array) -> int:
        producto_punto = self.agregacion(entrada)
        salida = self.activacion(producto_punto)
        return salida


    def __str__(self) -> str:
        return f"Neurona -> pesos: {self.pesos} || taza de aprendizaje: {self.taza_de_aprendizaje} || error: {self.error}"


class backpropagation():
    def __init__(self, neuronas_capa_oculta, neuronas_capa_salida) -> None:
        self.capa_oculta = neuronas_capa_oculta
        self.capa_salida = neuronas_capa_salida


    def entrenar(self, entradas_de_entrenamiento: np.array, salidas_de_entrenamiento: np.array) -> None:
        seguir = True
        print(self.__str__())
        contador = 0
        for _  in range(1,1001):
            activaciones_capa_oculta = list()
            activaciones_capa_salida = list()
            errores_capa_salida = list()
            errores_retropropagados = list()
            errores_capa_oculta = list()
            for entradas, salidas in zip(entradas_de_entrenamiento, salidas_de_entrenamiento):
                contador += 1
                # Paso hacia adelante capa oculta
                for neurona_oculta in self.capa_oculta:
                    agregacion = neurona_oculta.agregacion(entradas)
                    activacion = neurona_oculta.activacion(agregacion)
                    activaciones_capa_oculta.append(activacion)
                
                # Paso hacia adelante capa salida
                for neurona_salida, salida in zip(self.capa_salida, salidas):
                    agregacion = neurona_salida.agregacion(activaciones_capa_oculta)
                    activacion = neurona_salida.activacion(agregacion)
                    activaciones_capa_salida.append(activacion)
                    error_neurona_capa_salida = (1 - activacion)*(salida - activacion)
                    neurona_salida.error = error_neurona_capa_salida
                    errores_capa_salida.append(error_neurona_capa_salida)
                
                # Condicion de corte
                if -0.1 < self.capa_salida[1].error < 0.1:
                    seguir = False
                    break
                
                # Cálculo de los errores retropropagados
                for neurona_salida in self.capa_salida:
                    # error_retropropagado = np.dot(error_neurona_capa_salida, neurona_salida.pesos)
                    for peso in neurona_salida.pesos:
                        error_retropropagado = neurona_salida.error * peso
                        neurona_salida.errores_retropropagados.append(error_retropropagado)
                
                for i in range(0, len(self.capa_oculta)):
                    sumatoria = 0
                    for neurona_salida in self.capa_salida:
                        sumatoria += neurona_salida.error * neurona_salida.errores_retropropagados[i]
                    errores_retropropagados.append(sumatoria)
                
                for activacion_capa_oculta, error_retropropagado in zip(activaciones_capa_oculta, errores_retropropagados):
                    error_neurona_capa_oculta = (1 - activacion_capa_oculta) * error_retropropagado
                    errores_capa_oculta.append(error_neurona_capa_oculta)
                
                for neurona_oculta, error_neurona_oculta in zip(self.capa_oculta, errores_capa_oculta):
                    neurona_oculta.error = error_neurona_oculta
                
                for neurona_salida in self.capa_salida:
                    nuevos_pesos = []
                    for activacion_capa_oculta, peso_actual in zip(activaciones_capa_oculta, neurona_salida.pesos):
                        nuevos_pesos.append(peso_actual + neurona_salida.taza_de_aprendizaje * neurona_salida.error * activacion_capa_oculta)
                    neurona_salida.pesos = nuevos_pesos
                
                for neurona_oculta in self.capa_oculta:
                    nuevos_pesos = []
                    for entrada, peso_actual in zip(entradas, neurona_oculta.pesos):
                        nuevos_pesos.append(peso_actual + neurona_oculta.taza_de_aprendizaje * neurona_oculta.error * entrada)
                    neurona_oculta.pesos = nuevos_pesos
                    
                activaciones_capa_oculta.clear()
                activaciones_capa_salida.clear()
                errores_capa_salida.clear()
                errores_retropropagados.clear()
                errores_capa_oculta.clear()
                # seguir = False
                # print(self.__str__())
        print("CONTADOR: ", contador)


    def predecir(self, entradas: np.array):
        for entrada in entradas:
            imprimir = f"{entrada} - ["
            for neurona in self.neuronas:
                imprimir += str(neurona.predecir(entrada)) + " "
            imprimir += "]"
            print(imprimir)
    

    def predecir_entradas_de_testeo(self, entradas: np.array):
        salidas_ok = np.array([[ 1, -1],
                               [ 1, -1],
                               [ 1,  1],
                               [-1,  1],
                               [ 1, -1],
                               [-1, -1],
                               [ 1,  1],
                               [ 1,  1],
                               [-1,  1]])
        salidas = list()
        imprimir = ""
        for entrada in entradas:
            imprimir += f"{entrada} - ["
            salidas_actuales = list()
            for neurona in self.neuronas:
                salida = neurona.predecir(entrada)
                salidas_actuales.append(salida)
                imprimir += str(salida) + " "
            imprimir += "]\n"
            salidas.append(salidas_actuales)
        salidas_matriz = np.vstack(salidas)
        if np.array_equal(salidas_matriz, salidas_ok):
            return imprimir
        else:
            return None
    

    def __str__(self) -> str:
        imprimir = "BACKPROPAGATION\n"
        for neurona in self.capa_salida:
            imprimir += f"{neurona}\n"
        for neurona in self.capa_oculta:
            imprimir += f"{neurona}\n"
        return imprimir
    
if __name__ == "__main__":
    entradas_de_entrenamiento = np.array([[ 1,  1,  1,  1], 
                                          [-1,  1,  1,  1], 
                                          [ 1,  1, -1, -1], 
                                          [-1, -1, -1, -1], 
                                          [ 1, -1,  1,  1], 
                                          [ 1,  1, -1,  1], 
                                          [ 1,  1,  1, -1]])
    
    salidas_de_entrenamiento = np.array([[-1, -1], 
                                         [-1,  1], 
                                         [ 1, -1], 
                                         [ 1,  1], 
                                         [ 1, -1], 
                                         [-1,  1], 
                                         [ 1, -1]])
    
    entradas_de_testeo =np.array([[ 1, -1, -1, -1],
                                  [ 1, -1,  1, -1],
                                  [-1, -1,  1,  1],
                                  [-1,  1, -1,  1],
                                  [-1, -1,  1, -1],
                                  [-1,  1,  1, -1],
                                  [ 1, -1, -1,  1],
                                  [-1, -1, -1,  1],
                                  [-1,  1, -1, -1]])
    
    # neurona_1 = neurona([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)], 0.2)
    neurona_1 = neurona([0.07041993544337499, -0.029085890427333824, -0.049526424118370584, -0.04468778985713509], 0.2)
    # neurona_2 = neurona([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)], 0.2)
    neurona_2 = neurona([-0.05023876351731811, -0.07020683364671161, -0.017421182495140644, -0.04219661513444326], 0.2)

    neurona_3 = neurona([-0.011961687573127655, -0.06454545135062761], 0.2)
    neurona_4 = neurona([random.uniform(-0.1, 0.1),random.uniform(-0.1, 0.1)], 0.1)

    capa_oculta = [neurona_1, neurona_2]
    capa_salida = [neurona_3, neurona_4]

    backpropagation_actual = backpropagation(capa_oculta, capa_salida)

    original = backpropagation_actual.__str__()
    
    backpropagation_actual.entrenar(entradas_de_entrenamiento, salidas_de_entrenamiento)
    # print("Original")
    # print(original)
    print(backpropagation_actual)

    # backpropagation_actual.predecir(entradas_de_testeo)

    # for _ in range(1000):
    #     pesos_random_neurona_1 = np.random.uniform(-1, 1, size=4)
    #     pesos_random_neurona_2 = np.random.uniform(-1, 1, size=4)
    #     taza_aprendizaje_1 = random.uniform(0.001, 0.1)
    #     taza_aprendizaje_2 = random.uniform(0.001, 0.1)

    #     neurona_1 = neurona(pesos_random_neurona_1, taza_aprendizaje_1)
    #     neurona_2 = neurona(pesos_random_neurona_2, taza_aprendizaje_2)
    #     perceptron_actual = perceptron(neurona_1, neurona_2)

    #     perceptron_actual.entrenar(entradas_de_entrenamiento, salidas_de_entrenamiento)
    #     # print(perceptron_actual)

    #     sirve = perceptron_actual.predecir_entradas_de_testeo(entradas_de_testeo)

    #     if sirve is not None:
    #         print(f"Pesos iniciales neurona 1: {pesos_random_neurona_1}")
    #         print(f"Pesos iniciales neurona 2: {pesos_random_neurona_2}")
    #         print(perceptron_actual)
    #         print(sirve)
    #         break



"""
SALIDAS CORRECTAS
[ 1 -1 -1 -1] - [ 1 -1 ]
[ 1 -1  1 -1] - [ 1 -1 ]
[-1 -1  1  1] - [ 1  1 ]
[-1  1 -1  1] - [-1  1 ]
[-1 -1  1 -1] - [ 1 -1 ]
[-1  1  1 -1] - [-1 -1 ]
[ 1 -1 -1  1] - [ 1  1 ]
[-1 -1 -1  1] - [ 1  1 ]
[-1  1 -1 -1] - [-1  1 ]
"""
