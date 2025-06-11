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

print("✅ Modelo treinado e salvo com sucesso!")
