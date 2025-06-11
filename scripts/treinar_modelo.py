import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from joblib import dump
import os

# Gerar dataset fictício
np.random.seed(42)
bairros = ['Centro', 'Moema', 'Itaim', 'Bela Vista', 'Tatuapé']
cidades = ['São Paulo']
tipos = ['Apartamento', 'Casa']
mobilias = ['Nenhuma', 'Armários', 'Completa']

dados = {
    'bairro': np.random.choice(bairros, 500),
    'cidade': np.random.choice(cidades, 500),
    'm2': np.random.randint(30, 200, 500),
    'quartos': np.random.randint(1, 5, 500),
    'banheiros': np.random.randint(1, 4, 500),
    'garagem': np.random.randint(0, 3, 500),
    'tipo': np.random.choice(tipos, 500),
    'ano': np.random.randint(1970, 2022, 500),
    'mobilia': np.random.choice(mobilias, 500)
}
df = pd.DataFrame(dados)

# Valor simulado
preco_base = 3000
fator_bairro = df['bairro'].map({
    'Centro': 1.0, 'Moema': 1.5, 'Itaim': 1.4, 'Bela Vista': 1.2, 'Tatuapé': 1.1
})
fator_mobilia = df['mobilia'].map({
    'Nenhuma': 0,
    'Armários': 10000,
    'Completa': 20000
})
df['valor'] = df['m2'] * preco_base * fator_bairro + df['quartos']*20000 + df['banheiros']*15000 + df['garagem']*10000 + fator_mobilia

# Label encoding
label_cols = ['bairro', 'cidade', 'tipo', 'mobilia']
le_dict = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Treinar modelo
X = df.drop(columns='valor')
y = df['valor']
colunas_modelo = X.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Avaliação e salvar
mae = mean_absolute_error(y_test, modelo.predict(X_test))
print(f'MAE: R${mae:,.2f}')

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
df.to_csv('data/dataset_imoveis.csv', index=False)
dump(modelo, 'models/modelo_avaliacao.joblib')
dump(le_dict, 'models/label_encoders.joblib')
dump(colunas_modelo, 'models/colunas_modelo.joblib')
