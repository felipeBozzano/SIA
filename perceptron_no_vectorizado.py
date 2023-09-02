class Neurona:
    def __init__(self, pesos_iniciales, taza_de_aprendizaje):
        self.pesos = pesos_iniciales
        self.taza_de_aprendizaje = taza_de_aprendizaje
        self.error = None

    def entrenar_neurona(self, entradas_de_entrenamiento, salidas_de_entrenamiento):
        while self.error != 0:
            for entrada, salida in zip(entradas_de_entrenamiento, salidas_de_entrenamiento):
                if self.error == 0:
                    continue
                producto_punto = self.agregacion(entrada)
                salida_obtenida = self.activacion(producto_punto)
                self.error = salida - salida_obtenida
                self.pesos = [w + self.taza_de_aprendizaje * self.error * x for w, x in zip(self.pesos, entrada)]

    def agregacion(self, entrada):
        return sum(w * x for w, x in zip(self.pesos, entrada))

    def activacion(self, resultado):
        return 1 if resultado > 0 else -1

    def predecir(self, entrada):
        producto_punto = self.agregacion(entrada)
        salida = self.activacion(producto_punto)
        return salida

    def __str__(self):
        return f"Neurona -> pesos: {self.pesos} || taza de aprendizaje: {self.taza_de_aprendizaje} || error: {self.error}"

class Perceptron:
    def __init__(self, neurona_1, neurona_2):
        self.neuronas = [neurona_1, neurona_2]

    def entrenar(self, entradas_de_entrenamiento, salidas_de_entrenamiento):
        salidas_de_entrenamiento = list(map(list, zip(*salidas_de_entrenamiento)))
        for neurona, salidas in zip(self.neuronas, salidas_de_entrenamiento):
            neurona.entrenar_neurona(entradas_de_entrenamiento, salidas)

    def entrenar_neurona(self, neurona, entradas_de_entrenamiento, salida_de_entrenamiento):
        self.neuronas[neurona].entrenar_neurona(entradas_de_entrenamiento, salida_de_entrenamiento)

    def predecir(self, entradas):
        for entrada in entradas:
            imprimir = f"{entrada} - ["
            for neurona in self.neuronas:
                imprimir += str(neurona.predecir(entrada)) + " "
            imprimir += "]"
            print(imprimir)

    def predecir_entradas_de_testeo(self, entradas):
        salidas_ok = [[1, -1],
                      [1, -1],
                      [1, 1],
                      [-1, 1],
                      [1, -1],
                      [-1, -1],
                      [1, 1],
                      [1, 1],
                      [-1, 1]]
        salidas = []
        imprimir = ""
        for entrada in entradas:
            imprimir += f"{entrada} - ["
            salidas_actuales = []
            for neurona in self.neuronas:
                salida = neurona.predecir(entrada)
                salidas_actuales.append(salida)
                imprimir += str(salida) + " "
            imprimir += "]\n"
            salidas.append(salidas_actuales)
        if salidas == salidas_ok:
            return imprimir
        else:
            return None

    def __str__(self):
        imprimir = "PERCEPTRON\n"
        for neurona in self.neuronas:
            imprimir += f"{neurona}\n"
        return imprimir

if __name__ == "__main__":
    entradas_de_entrenamiento = [[1, 1, 1, 1], 
                                 [-1, 1, 1, 1], 
                                 [1, 1, -1, -1], 
                                 [-1, -1, -1, -1], 
                                 [1, -1, 1, 1], 
                                 [1, 1, -1, 1], 
                                 [1, 1, 1, -1]]
    
    salidas_de_entrenamiento = [[-1, -1], 
                                [-1, 1], 
                                [1, -1], 
                                [1, 1], 
                                [1, -1], 
                                [-1, 1], 
                                [1, -1]]
    
    entradas_de_testeo = [[1, -1, -1, -1],
                          [1, -1, 1, -1],
                          [-1, -1, 1, 1],
                          [-1, 1, -1, 1],
                          [-1, -1, 1, -1],
                          [-1, 1, 1, -1],
                          [1, -1, -1, 1],
                          [-1, -1, -1, 1],
                          [-1, 1, -1, -1]]
    
    neurona_1 = Neurona([0.08174903, -0.8377704, 0.33671084, 0.57375835], 0.03673239027963381)
    neurona_2 = Neurona([-0.4094063, 0.53426245, -0.44470761, 0.98560924], 0.05788280232040685)
    perceptron_actual = Perceptron(neurona_1, neurona_2)
    
    perceptron_actual.entrenar(entradas_de_entrenamiento, salidas_de_entrenamiento)
    print(perceptron_actual)
    
    perceptron_actual.predecir(entradas_de_testeo)
