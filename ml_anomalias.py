import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

print("1. Cargando la base de datos...")
df = pd.read_csv('DatosDepuradosOrganizados.csv')

# 2. Ingeniería de Variables (Feature Engineering)
# Para evitar errores con columnas que contienen listas (arreglos de ROOT),
# seleccionaremos estrictamente las variables escalares (numéricas).
df_numeric = df.select_dtypes(include=[np.number]).dropna()

# Separamos nuestra variable objetivo 'lep_n' (solo para validar, NO para entrenar)
# Eliminamos 'eventNumber' porque es solo un ID que no aporta valor predictivo
columnas_a_excluir = ['eventNumber', 'lep_n']
X = df_numeric.drop(columns=[col for col in columnas_a_excluir if col in df_numeric.columns])

print(f"Entrenando modelo con {X.shape[1]} variables cinemáticas y {X.shape[0]} eventos...")

# 3. Preprocesamiento (Escalado)
# Los algoritmos de ML requieren que todas las variables tengan la misma escala
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Entrenamiento del Modelo de Machine Learning (Isolation Forest)
# contamination = 0.015 (Le decimos al modelo que esperamos ~1.5% de anomalías, 
# basado en tu análisis previo de 3+ leptones)
modelo_if = IsolationForest(n_estimators=100, contamination=0.015, random_state=42)

print("Entrenando el Bosque de Aislamiento (Esto puede tomar unos segundos)...")
df_numeric['prediccion_anomalia'] = modelo_if.fit_predict(X_scaled)

# El modelo devuelve -1 para Anomalías y 1 para Eventos Normales
# Lo mapeamos: 1 (Anomalía), 0 (Normal) para comparar fácilmente
df_numeric['es_anomalia_ml'] = df_numeric['prediccion_anomalia'].apply(lambda x: 1 if x == -1 else 0)

# 5. Validación del Negocio (Traduciendo el ML a la Física)
# Definimos la verdad fundamental: Anomalía = eventos con más de 2 leptones
df_numeric['es_anomalia_fisica'] = df_numeric['lep_n'].apply(lambda x: 1 if x > 2 else 0)

print("\n======================================================")
print(" RESULTADOS DEL MODELO DE MACHINE LEARNING")
print("======================================================")

# Matriz de Confusión
matriz = confusion_matrix(df_numeric['es_anomalia_fisica'], df_numeric['es_anomalia_ml'])
print("\nMatriz de Confusión:")
print(f"Verdaderos Normales (Encontrados correctamente): {matriz[0][0]}")
print(f"Falsas Anomalías (Ruido): {matriz[0][1]}")
print(f"Anomalías No Detectadas: {matriz[1][0]}")
print(f"VERDADERAS ANOMALÍAS (El modelo detectó física nueva): {matriz[1][1]}")

print("\nReporte de Clasificación:")
print(classification_report(df_numeric['es_anomalia_fisica'], df_numeric['es_anomalia_ml'], 
                            target_names=['Eventos Normales', 'Eventos Anómalos']))