# Link del colab

### https://colab.research.google.com/drive/1jH8-4Ai4Ryczjf0B6eLlFEUqo1vODZdV?usp=sharing

## Sección 1: El Núcleo de la Investigación Teórica


* **Enunciado Matemático:** Puedes presentar que, para una población con media $\mu$ y varianza finita $\sigma^2$, la distribución de las
* medias muestrales $\bar{X}$ se aproxima a una distribución Normal $N(\mu, \sigma^2/n)$ conforme el tamaño de la muestra $n$ aumenta.
* **Media de las medias:** Explica que la media de la distribución muestral es igual a la media poblacional porque la media muestral es un
* estimador insesgado; al promediar muchas muestras, los errores se cancelan y se llega al parámetro real.
* **Error Estándar ($\sigma_{\bar{x}}$):** Defínelo como la medida de qué tan alejado está el promedio muestral de la realidad poblacional.
* Su fórmula es:

$$\sigma_{\bar{x}} = \frac{\sigma}{\sqrt{n}}$$

* **Condiciones de tamaño ($n$):** Menciona la "regla de oro" de $n \ge 30$ para asumir normalidad. Sin embargo, aclara que si la población
* es extremadamente asimétrica (como una distribución de Pareto), podrías necesitar un $n$ mayor para que la campana sea "limpia".

## Sección 2: Demostración Práctica Basada en Código


* **Población Base:** El código utiliza una distribución exponencial (típica de fallos de hardware o tiempos de permanencia) con $\lambda = 0.5$, la
* cual es altamente asimétrica.
* **Simulación de Monte Carlo:** El script realiza un bucle para extraer 1,000 muestras aleatorias de tamaño $n = 30$.
* **Transformación Visual:** Podrás mostrar cómo la gráfica naranja (población original caótica) se transforma en una campana de Gauss
azul (distribución de medias muestrales).

### Primer código

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
*Creación de la Población "Universo" (Altamente asimétrica)*
np.random.seed(42)
poblacion_exponencial = np.random.exponential(scale=2.0, size=100000)
mu_pob = np.mean(poblacion_exponencial)
sigma_pob = np.std(poblacion_exponencial)
print(f"--- Parámetros Poblacionales Reales ---")
print(f"Media (μ): {mu_pob:.4f}")
print(f"Desviación Estándar (σ): {sigma_pob:.4f}")
*Visualización de la Población Original*
plt.figure(figsize=(8, 4))
sns.histplot(poblacion_exponencial, bins=50, kde=True, color='orange')
plt.title("Distribución Poblacional (Exponencial) - Claramente NO Normal")
plt.xlabel("Tiempo de Permanencia")
plt.ylabel("Frecuencia")
plt.axvline(mu_pob, color='red', linestyle='dashed', label=f'Media: {mu_pob:.2f}')
plt.legend()
plt.show()

### Segundo código

*Parámetros de la simulación*
tamaño_muestra = 30 # n
numero_muestras = 1000 # k
*Array para almacenar las medias de cada muestra*
medias_muestrales = []
*Bucle de Monte Carlo simple*
for _ in range(numero_muestras):
*Extraer muestra aleatoria sin reemplazo*
  muestra = np.random.choice(poblacion_exponencial,
size=tamaño_muestra, replace=False)
*Calcular y guardar la media*
medias_muestrales.append(np.mean(muestra))
*Estadísticos de la Distribución Muestral*
media_de_medias = np.mean(medias_muestrales)
error_estandar_empirico = np.std(medias_muestrales)
error_estandar_teorico = sigma_pob / np.sqrt(tamaño_muestra)
print(f"--- Estadísticos de las Medias Muestrales (n={tamaño_muestra}) ---")
print(f"Media de las Medias Muestrales (E[X̄ ]): {media_de_medias:.4f}")
print(f"Error Estándar Empírico (σ_̄x): {error_estandar_empirico:.4f}")
print(f"Error Estándar Teórico (σ/√n): {error_estandar_teorico:.4f}")
*Visualización de la convergencia a la Normal*
plt.figure(figsize=(8, 4))
sns.histplot(medias_muestrales, bins=30, kde=True, color='blue')
plt.title(f"Distribución de las Medias Muestrales (n={tamaño_muestra}) - ¡Convergencia a la Normal!")
plt.xlabel("Valor de las Medias Muestrales")
plt.ylabel("Frecuencia")
plt.axvline(media_de_medias, color='red', linestyle='dashed', label=f'E[X̄ ]:{media_de_medias:.2f}')
plt.legend()
plt.show()


## Sección 3: El TLC en Ciencias de la Computación (Enfoque: Machine Learning)

* **Modelado de Errores y Residuos:** Las fuentes explican que el TLC es el "puente" que permite usar la distribución Normal $Z$ para casi todo.
* En ML, esto justifica por qué asumimos que los errores de predicción son normales, permitiendo calcular intervalos de confianza precisos sobre
* la métrica de exactitud del modelo.
* **Selección de Características:** La fuente detalla el uso de pruebas estadísticas (como Chi-cuadrado, $\chi^2$) en Python para evaluar si existe
* una asociación significativa entre variables en un dataset. Esto permite optimizar algoritmos descartando variables independientes, lo que
* mejora la velocidad de procesamiento y la precisión del código final.


## **Bibliografía:**
  * R. E. Walpole, R. H. Myers, S. L. Myers, y K. Ye, *Probabilidad y estadística para ingeniería y ciencias*, 9na ed. Pearson Educación, 2012.
  * SciPy Developers, "scipy.stats Documentation," SciPy.org, 2024. [Online]. Disponible: https://docs.scipy.org/doc/scipy/reference/stats.html.
