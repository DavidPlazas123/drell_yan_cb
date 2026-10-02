import matplotlib.pyplot as plt
import pandas as pd

print("Cargando la base de datos limpia...")
# 1. Cargar los datos procesados por el pipeline ETL
df = pd.read_csv('DatosDepuradosOrganizados.csv')

# 2. Calcular la distribución (Frecuencia de eventos)
conteo_leptones = df['lep_n'].value_counts().sort_index()
total_eventos = len(df)

print("\n--- Perfilado de Datos (Data Profiling) ---")
for leptones, conteo in conteo_leptones.items():
    porcentaje = (conteo / total_eventos) * 100
    print(f"Eventos con {leptones} leptones: {conteo} ({porcentaje:.4f}%)")

# 3. Visualización de alto impacto (Estilo corporativo)
plt.figure(figsize=(10, 6))
# Usamos un color profesional (Azul marino)
bars = plt.bar(conteo_leptones.index, conteo_leptones.values, color='#1f77b4', edgecolor='black')

# Aplicamos escala logarítmica tal como lo hiciste en tu investigación original
plt.yscale('log')

# Estilizado de la gráfica
plt.title('Detección de Anomalías: Distribución de Leptones por Evento', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Número de Leptones Detectados (lep_n)', fontsize=12)
plt.ylabel('Cantidad de Eventos (Escala Logarítmica)', fontsize=12)
plt.xticks(conteo_leptones.index)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Añadir las etiquetas de texto encima de cada barra para rápida lectura
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval * 1.2, int(yval), 
             va='bottom', ha='center', fontsize=10, fontweight='bold', color='#333333')

# Guardar la gráfica en alta calidad y mostrarla
plt.tight_layout()
plt.savefig('distribucion_anomalias.png', dpi=300)
print("\nGráfica guardada exitosamente como 'distribucion_anomalias.png'")
plt.show()