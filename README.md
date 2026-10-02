# Detección de Anomalías Cinemáticas en Eventos de Dispersión Drell-Yan

Este repositorio contiene un pipeline analítico end-to-end diseñado para procesar, limpiar y modelar medio millón de eventos de colisión de partículas provenientes del detector ATLAS del CERN[cite: 2, 7]. El objetivo del proyecto es identificar eventos anómalos multivariados utilizando técnicas de Machine Learning sobre un conjunto de datos altamente desbalanceado.

## Resumen del Proyecto

El proceso físico de dispersión Drell-Yan genera típicamente pares de leptones[cite: 2]. Tras procesar 500,000 eventos[cite: 4, 8], el análisis exploratorio confirmó un desbalance masivo en los datos: más del 85% de las colisiones resultan en 2 leptones, mientras que los eventos anómalos (3, 4 o 5 leptones) representan una fracción mínima (hasta un 0.0002%)[cite: 6]. 

Para abordar este problema de detección de anomalías sin depender de reglas manuales, se construyó una arquitectura basada en aprendizaje no supervisado que analiza la rareza cinemática (momento transversal y pseudo-rapidez) de cada evento.

## Arquitectura y Stack Tecnológico

*   **Lenguaje:** Python 3.10
*   **Ingeniería de Datos (ETL):** `uproot`, `pandas`, `numpy`
*   **Machine Learning:** `scikit-learn` (Isolation Forest)
*   **Visualización:** `matplotlib` (Generación de gráficos headless)

## Estructura del Pipeline

1.  **`etl_pipeline.py` (Extracción y Transformación):** 
    Lee los archivos binarios `.root` nativos del CERN utilizando `uproot`[cite: 3]. Ejecuta la limpieza de variables irrelevantes (factores de escala, pesos, y variables de vértice)[cite: 4] y exporta un conjunto de datos tabular estructurado y optimizado para entrenamiento.
2.  **`eda_anomalias.py` (Perfilado de Datos):** 
    Cuantifica la distribución de clases y genera visualizaciones en escala logarítmica para demostrar la asimetría de la distribución y el nivel de desbalance.
3.  **`ml_anomalias.py` (Modelado Predictivo):** 
    Implementa un modelo *Isolation Forest* que procesa el espacio multidimensional de las variables cinemáticas estandarizadas para aislar matemáticamente los eventos atípicos.

## Resultados y Discusión del Modelo

El modelo de aprendizaje no supervisado fue evaluado contra la variable objetivo real (cantidad de leptones). Los resultados de la matriz de confusión mostraron:
*   Identificación exitosa de 127 verdaderas anomalías sin intervención de reglas lógicas[cite: 8].
*   Detección de 7,353 falsas anomalías (ruido cinemático) y una precisión global en la clase minoritaria del 2%[cite: 8].

**Interpretación de Negocio/Dominio:** 
El alto volumen de falsos positivos indica que el modelo detectó correctamente eventos con características cinemáticas extremas (valores atípicos en la cola de distribución del momento transversal). Sin embargo, la rareza matemática en la cinemática no garantiza estrictamente la generación física de leptones adicionales. 

Este primer modelo de *baseline* demuestra la viabilidad de procesar la base de datos completa y establece el fundamento para una iteración v2.0 utilizando enfoques de aprendizaje supervisado (como XGBoost o Random Forest) para aislar la firma exacta de estos decaimientos secundarios.

## Instrucciones de Ejecución

Para replicar este entorno de forma aislada:

```bash
# 1. Crear y activar el entorno
conda create --name drell_yan python=3.10 -y
conda activate drell_yan

# 2. Instalar dependencias
conda install -c conda-forge uproot pandas numpy matplotlib scikit-learn -y

# 3. Ejecutar el pipeline secuencialmente
python etl_pipeline.py
python eda_anomalias.py
python ml_anomalias.py