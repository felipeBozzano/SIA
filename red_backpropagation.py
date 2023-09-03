import numpy as np


class neurona():
    def __init__(self, pesos_iniciales: list, taza_de_aprendizaje: float) -> None:
        self.pesos = pesos_iniciales
        self.taza_de_aprendizaje = taza_de_aprendizaje
        self.error: float = None
    

    def agregacion(self, entrada):
        return np.dot(self.pesos, entrada)
    

    def activacion(self, resultado) -> int:
        return 1/(1 + np.exp(-resultado))
    

    def predecir(self, entrada: np.array) -> int:
        producto_punto = self.agregacion(entrada)
        salida = self.activacion(producto_punto)
        return salida


    def __str__(self) -> str:
        return f"Neurona -> pesos: {self.pesos} || taza de aprendizaje: {self.taza_de_aprendizaje} || error: {self.error}"


class backpropagation():
    def __init__(self, neurona_1: neurona, neurona_2: neurona, neurona_3: neurona, neurona_4: neurona) -> None:
        self.capa_oculta = [neurona_1, neurona_2]
        self.capa_salida = [neurona_3, neurona_4]


    def entrenar(self, entradas_de_entrenamiento: np.array, salidas_de_entrenamiento: np.array) -> None:
        seguir = True
        for _  in range(1,10000):
            activaciones_capa_oculta = list()
            activaciones_capa_salida = list()
            errores_capa_salida = list()
            errores_retropropagado = list()
            errores_capa_oculta = list()
            for entradas, salidas in zip(entradas_de_entrenamiento, salidas_de_entrenamiento):
                for neurona_oculta in self.capa_oculta:
                    agregacion = neurona_oculta.agregacion(entradas)
                    activacion = neurona_oculta.activacion(agregacion)
                    activaciones_capa_oculta.append(activacion)
                for neurona_salida, salida in zip(self.capa_salida, salidas):
                    agregacion = neurona_salida.agregacion(activaciones_capa_oculta)
                    activacion = neurona_salida.activacion(agregacion)
                    activaciones_capa_salida.append(activacion)
                    error_neurona_capa_salida = (1 - activacion)*(salida - activacion)
                    neurona_salida.error = error_neurona_capa_salida
                    errores_capa_salida.append(error_neurona_capa_salida)
                    error_retropropagado = np.dot([error_neurona_capa_salida, error_neurona_capa_salida], neurona_salida.pesos)
                    errores_retropropagado.append(error_retropropagado)
                if self.capa_salida[0].error == 0 and self.capa_salida[1].error == 0:
                    seguir = False
                    break
                else: 
                    for activacion_capa_oculta, error_retropropagado in zip(activaciones_capa_oculta, errores_retropropagado):
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
                errores_retropropagado.clear()
                errores_capa_oculta.clear()
                seguir = False


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
    
    neurona_1 = neurona([ 0.08174903, -0.8377704, 0.33671084, 0.57375835],  0.03673239027963381)
    neurona_2 = neurona([-0.4094063,   0.53426245, -0.44470761,  0.98560924], 0.05788280232040685)

    neurona_3 = neurona([ 0.8174903, -0.8377704],  0.03673239027963381)
    neurona_4 = neurona([-0.4094063,   0.53426245], 0.05788280232040685)
    backpropagation_actual = backpropagation(neurona_1, neurona_2, neurona_3, neurona_4)
    
    backpropagation_actual.entrenar(entradas_de_entrenamiento, salidas_de_entrenamiento)
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
