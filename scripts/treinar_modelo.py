import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Carregar dataset
caminho_csv = 'data/dataset_imoveis.csv'
df = pd.read_csv(caminho_csv)

# Inicializar dicionário de LabelEncoders
le_dict = {}

# Colunas categóricas para codificar
colunas_cat = ['estado', 'cidade', 'bairro', 'tipo', 'mobilia']
for col in colunas_cat:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# Criar estrutura hierárquica de estados, cidades e bairros
estados_cidades_bairros = {}
for _, row in df.iterrows():
    estado = row['estado']
    cidade = row['cidade']
    bairro = row['bairro']
    estado_str = le_dict['estado'].inverse_transform([int(estado)])[0]
    cidade_str = le_dict['cidade'].inverse_transform([int(cidade)])[0]
    bairro_str = le_dict['bairro'].inverse_transform([int(bairro)])[0]

    if estado_str not in estados_cidades_bairros:
        estados_cidades_bairros[estado_str] = {}
    if cidade_str not in estados_cidades_bairros[estado_str]:
        estados_cidades_bairros[estado_str][cidade_str] = set()
    estados_cidades_bairros[estado_str][cidade_str].add(bairro_str)

# Converter sets para listas
for estado in estados_cidades_bairros:
    for cidade in estados_cidades_bairros[estado]:
        estados_cidades_bairros[estado][cidade] = list(estados_cidades_bairros[estado][cidade])

# Features e alvo
X = df.drop(columns=['valor'])
y = df['valor']

# Treinar modelo
modelo = RandomForestRegressor(n_estimators=150, random_state=42)
modelo.fit(X, y)

# Salvar artefatos
joblib.dump(modelo, 'models/modelo_avaliacao.joblib')
joblib.dump(le_dict, 'models/label_encoders.joblib')
joblib.dump(X.columns.tolist(), 'models/colunas_modelo.joblib')
joblib.dump(estados_cidades_bairros, 'models/estados_cidades_bairros.joblib')

print("✅ Modelo treinado e estrutura de estados/cidades/bairros salva com sucesso!")
