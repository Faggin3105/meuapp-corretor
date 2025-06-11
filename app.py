import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar modelo e codificadores
modelo = joblib.load("models/modelo_avaliacao.joblib")
le_dict = joblib.load("models/label_encoders.joblib")
colunas_modelo = joblib.load("models/colunas_modelo.joblib")
estados_cidades_bairros = joblib.load("models/estados_cidades_bairros.joblib")

# Menu lateral
st.set_page_config(page_title="Avaliação de Imóveis da Orla", layout="centered")
menu = st.sidebar.selectbox("Escolha uma funcionalidade:", [
    "Avaliação de Imóveis",
    "Criação de Contratos",
    "Posição Solar",
    "Calculadora Financeira",
    "Simulador de Investimento",
    "Consulta de Índices",
    "Notícias do Mercado",
    "Dicionário do Corretor",
    "Biblioteca do Corretor",
    "Universidade do Corretor",
    "Análise de Perfil de Cliente",
    "Agenda de Eventos"
])

if menu == "Avaliação de Imóveis":
    st.title("Avaliação de Imóveis da Orla")

    # Seleção dinâmica: Estado → Cidade → Bairro
    estado = st.selectbox("Estado", list(estados_cidades_bairros.keys()), key="estado")
    cidades = list(estados_cidades_bairros[estado].keys())
    cidade = st.selectbox("Cidade", cidades, key="cidade")
    bairros = estados_cidades_bairros[estado][cidade]
    bairro = st.selectbox("Bairro", bairros, key="bairro")

    # Demais entradas
    tipo = st.selectbox("Tipo do Imóvel", le_dict['tipo'].classes_)
    m2 = st.slider("Área (m²)", min_value=10, max_value=2000, step=10)
    quartos = st.slider("Quartos", min_value=0, max_value=10, step=1)
    banheiros = st.slider("Banheiros", min_value=0, max_value=10, step=1)
    garagem = st.slider("Vagas de Garagem", min_value=0, max_value=10, step=1)
    ano = st.slider("Ano de Construção", min_value=1900, max_value=2025, step=1)
    mobilia = st.selectbox("Mobilia", le_dict['mobilia'].classes_)

    if st.button("Avaliar Imóvel"):
        try:
            entrada = pd.DataFrame([{
                'estado': le_dict['estado'].transform([estado])[0],
                'cidade': le_dict['cidade'].transform([cidade])[0],
                'bairro': le_dict['bairro'].transform([bairro])[0],
                'tipo': le_dict['tipo'].transform([tipo])[0],
                'm2': m2,
                'quartos': quartos,
                'banheiros': banheiros,
                'garagem': garagem,
                'ano': ano,
                'mobilia': le_dict['mobilia'].transform([mobilia])[0]
            }])[colunas_modelo]

            valor_estimado = modelo.predict(entrada)[0]
            valor_m2 = valor_estimado / m2 if m2 else 0

            st.success(f"💰 Valor estimado: R$ {valor_estimado:,.2f}")
            st.info(f"📐 Valor por m²: R$ {valor_m2:,.2f}")

            # Exibir gráfico de distribuição por faixa de preço
            df = pd.read_csv("data/dataset_imoveis.csv")
            df['faixa'] = pd.cut(df['valor'], bins=[0,200000,400000,600000,800000,1000000,1500000],
                                  labels=["Até 200k", "200k-400k", "400k-600k", "600k-800k", "800k-1M", "1M+"])
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.set_style("whitegrid")
            sns.countplot(data=df, x='faixa', palette='coolwarm', ax=ax)
            ax.set_title("Distribuição de Imóveis por Faixa de Preço", fontsize=14, fontweight='bold')
            ax.set_xlabel("Faixa de Preço", fontsize=12)
            ax.set_ylabel("Quantidade de Imóveis", fontsize=12)
            for container in ax.containers:
                ax.bar_label(container, fmt='%d', label_type='edge', fontsize=10)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Erro na avaliação: {str(e)}")
else:
    st.title(f"{menu}")
    st.info("Funcionalidade em construção... em breve!")