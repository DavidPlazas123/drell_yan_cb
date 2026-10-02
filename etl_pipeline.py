import numpy as np
import pandas as pd
import uproot

# ==========================================
# 1. Lectura de archivos .ROOT del CERN
# ==========================================
with uproot.open("mc_173043.DYmumuM08to15.root") as archive_print:
    mini = archive_print["mini"]
    branches = mini.keys()
    
    data = {}
    for branch_name in branches:
        branch_data = mini[branch_name].array(library="np")
        
        if isinstance(branch_data, np.ndarray):
            data[branch_name] = branch_data
        else:
            data[branch_name] = [list(x) for x in branch_data]
            
# Conversión a DataFrame tabular
df = pd.DataFrame.from_dict(data, orient='index').transpose()

# ==========================================
# 2. Limpieza de la base de datos
# ==========================================
# Selección de las primeras 33 columnas de interés
dfCopy = df.iloc[:, :33]

# Eliminación de factores de escala y variables de vértice
dfCopy2 = dfCopy.drop([
    'mcWeight', 'pvxp_n', 'vxp_z', 'hasGoodVertex', 'scaleFactor_PILEUP', 
    'scaleFactor_ELE', 'scaleFactor_MUON', 'scaleFactor_BTAG', 
    'scaleFactor_TRIGGER', 'scaleFactor_JVFSF', 'scaleFactor_ZVERTEX', 
    'trigE', 'trigM'
], axis=1)

# Eliminación de variables de canal y tracking
dfCopy3 = dfCopy2.drop([
    'channelNumber', 'passGRL', 'lep_truthMatched', 'lep_trigMatched', 
    'lep_flag', 'lep_ptcone30', 'lep_etcone20', 'lep_trackd0pvunbiased', 
    'lep_tracksigd0pvunbiased'
], axis=1)

# ==========================================
# 3. Exportación de la nueva base depurada
# ==========================================
file_path = 'DatosDepuradosOrganizados.csv'
dfCopy3.to_csv(file_path, index=False)
print(f"Base de datos limpia exportada a: {file_path}")

# ==========================================
# 4. Separación y agrupación de datos
# ==========================================
# Agrupación por número de leptones para análisis de simetría
df_nlep = dfCopy3[['eventNumber', 'lep_n']]
df_nlep_graph = df_nlep.groupby('lep_n').count()

print("\nConteo de eventos por número de leptones:")
print(df_nlep_graph)