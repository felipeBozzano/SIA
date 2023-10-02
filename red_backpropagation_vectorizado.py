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
            return (np.exp(resultado) - np.exp(-resultado))/(np.exp(resultado) + np.exp(-resultado))
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


    """
    PROBAR POR EPOCAS. ALEATORIZAR LAS ENTRADAS EN CADA EPOCA.
    """
    def entrenar(self, entradas_de_entrenamiento: np.array, salidas_de_entrenamiento: np.array) -> None:
        contador = 0
        while contador < 300 or any(
            self.capa_salida[i].error is None or (
                self.capa_salida[i].error < -0.01 or self.capa_salida[i].error > 0.01
            )
            for i in (0, 1)
        ):
            
            if contador == 1000:
                print(contador)
                return

            for entradas, salidas in zip(entradas_de_entrenamiento, salidas_de_entrenamiento):
                # Listas donde se guardan los datos de la red
                activaciones_capa_oculta = list()
                activaciones_capa_salida = list()
                errores_capa_salida = list()
                errores_retropropagados = list()
                errores_capa_oculta = list()

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
                
                # Cálculo de los errores retropropagados
                for neurona_salida in self.capa_salida:
                    for peso in neurona_salida.pesos:
                        error_retropropagado = neurona_salida.error * peso
                        neurona_salida.errores_retropropagados.append(error_retropropagado)
                
                for i in range(0, len(self.capa_oculta)):
                    sumatoria = 0
                    for neurona_salida in self.capa_salida:
                        sumatoria += neurona_salida.errores_retropropagados[i]
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

            contador += 1
            

    def predecir(self, entradas_de_testeo: np.array):
        activaciones_capa_oculta = list()
        activaciones_capa_salida = list()
        for entradas in entradas_de_testeo:
            # Paso hacia adelante capa oculta
            for neurona_oculta in self.capa_oculta:
                agregacion = neurona_oculta.agregacion(entradas)
                activacion = neurona_oculta.activacion(agregacion)
                activaciones_capa_oculta.append(activacion)
                        
            # Paso hacia adelante capa salida
            for neurona_salida in self.capa_salida:
                agregacion = neurona_salida.agregacion(activaciones_capa_oculta)
                activacion = neurona_salida.activacion(agregacion)
                activaciones_capa_salida.append(activacion)
            print(activaciones_capa_salida)

            activaciones_capa_oculta.clear()
            activaciones_capa_salida.clear()
    

    def predecir_entradas_de_testeo(self, entradas: np.array):
        prediccion = list()

        for entradas in entradas_de_testeo:
            activaciones_capa_oculta = list()
            activaciones_capa_salida_temporal = list()

            # Paso hacia adelante capa oculta
            for neurona_oculta in self.capa_oculta:
                agregacion = neurona_oculta.agregacion(entradas)
                activacion = neurona_oculta.activacion(agregacion)
                activaciones_capa_oculta.append(activacion)
                        
            # Paso hacia adelante capa salida
            for neurona_salida in self.capa_salida:
                agregacion = neurona_salida.agregacion(activaciones_capa_oculta)
                activacion = neurona_salida.activacion(agregacion)
                activaciones_capa_salida_temporal.append(activacion)

            prediccion.append(activaciones_capa_salida_temporal)

        salidas_matriz = np.vstack(prediccion)
        salidas_matriz_corregidas = np.where(salidas_matriz > 0, 1, -1)
        return salidas_matriz_corregidas

    
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
    
    salidas_ok = np.array([[ 1, -1],
                           [ 1, -1],
                           [ 1,  1],
                           [-1,  1],
                           [ 1, -1],
                           [-1, -1],
                           [ 1,  1],
                           [ 1,  1],
                           [-1,  1]])
    
    for _ in range(0,4):
        neurona_1  = neurona([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)], random.uniform(0.1, 0.5))
        neurona_2  = neurona([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)], random.uniform(0.1, 0.5))
        neurona_3  = neurona([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)], random.uniform(0.1, 0.5))
        neurona_4  = neurona([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)], random.uniform(0.1, 0.5))
        neurona_5  = neurona([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)], random.uniform(0.1, 0.5))
        neurona_6  = neurona([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)], random.uniform(0.1, 0.5))
        neurona_7  = neurona([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)], random.uniform(0.1, 0.5))
        neurona_8  = neurona([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)], random.uniform(0.1, 0.5))

        capa_oculta = [neurona_1, neurona_2, neurona_3, neurona_4, neurona_5, neurona_6,]
        # neurona_7, neurona_8

        neurona_9  = neurona([random.uniform(-0.1, 0.1) for _ in range(0, len(capa_oculta))], random.uniform(0.1, 0.5))
        neurona_10 = neurona([random.uniform(-0.1, 0.1) for _ in range(0, len(capa_oculta))], random.uniform(0.1, 0.5))

        capa_salida = [neurona_9, neurona_10]

        backpropagation_actual = backpropagation(capa_oculta, capa_salida)
        backpropagation_original = backpropagation_actual
        backpropagation_actual.entrenar(entradas_de_entrenamiento, salidas_de_entrenamiento)
        print(backpropagation_actual)

        # if -0.01 <= backpropagation_actual.capa_salida[0].error <= 0.01 and -0.01 <= backpropagation_actual.capa_salida[1].error <= 0.01:
        #     prediccion = backpropagation_actual.predecir_entradas_de_testeo(entradas_de_testeo)
        #     # print(prediccion)
        #     if np.array_equal(prediccion, salidas_ok):
        #         print(backpropagation_original)
        #         print(backpropagation_actual)
        #         print(prediccion)
        
# neurona_1 = neurona([-0.07341374569337139, -0.050207966005393394, 0.02739102009240485, -0.06874419453865008], 0.10244615045314533)
# neurona_2 = neurona([-0.08163492101577363, -0.04180432041060687, 0.060581632279734454, -0.05373163121883531], 0.14307583875242635)
# neurona_3 = neurona([0.07657907271580519, 0.09947905750351202, 0.05321644670923045, -0.04779252780428256], 0.103820502456266)
# neurona_4 = neurona([-0.08715259679109706, -0.036401209052637815, 0.08714161229117154, -0.04067076238709908], 0.12696759781870642)

# neurona_5 = neurona([-0.08222715735401039, -0.07375113103458426, -0.007484636787434454, 0.06477876605200866], 0.15092686797766544)
# neurona_6 = neurona([-0.09281134356946108, 0.0954204849156626, 0.04854790566079456, -0.09091727217617396], 0.15235408398276823)

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
