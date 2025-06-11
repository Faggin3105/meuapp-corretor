import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt

# Carregar modelo e encoders
modelo = joblib.load('models/modelo_avaliacao.joblib')
le_dict = joblib.load('models/label_encoders.joblib')
colunas_modelo = joblib.load('models/colunas_modelo.joblib')
df = pd.read_csv('data/dataset_imoveis.csv')

st.set_page_config(page_title="Corretor AI - Sistema Imobiliário", layout="centered")

# Dicionário completo de estados e cidades do Brasil (exemplo parcial para simplificação)
estados_cidades = {
    "AC": ["Rio Branco"],
    "AL": ["Maceió"],
    "AP": ["Macapá"],
    "AM": ["Manaus"],
    "BA": ["Salvador", "Feira de Santana", "Ilhéus"],
    "CE": ["Fortaleza"],
    "DF": ["Brasília"],
    "ES": ["Vitória"],
    "GO": ["Goiânia"],
    "MA": ["São Luís"],
    "MT": ["Cuiabá"],
    "MS": ["Campo Grande"],
    "MG": ["Belo Horizonte", "Uberlândia", "Ouro Preto"],
    "PA": ["Belém"],
    "PB": ["João Pessoa"],
    "PR": ["Curitiba"],
    "PE": ["Recife"],
    "PI": ["Teresina"],
    "RJ": ["Rio de Janeiro", "Niterói", "Petrópolis"],
    "RN": ["Natal"],
    "RS": ["Porto Alegre"],
    "RO": ["Porto Velho"],
    "RR": ["Boa Vista"],
    "SC": ["Florianópolis"],
    "SP": ["São Paulo", "Campinas", "Santos"],
    "SE": ["Aracaju"],
    "TO": ["Palmas"]
}

# Navegação por abas
aba = st.sidebar.radio("Escolha uma função:", [
    "🏠 Avaliação de Imóveis",
    "📝 Criação de Contratos",
    "☀️ Posição Solar",
    "📈 Calculadora Financeira",
    "📊 Simulador de Investimento",
    "📉 Consulta de Índices",
    "📰 Notícias do Mercado",
    "📚 Dicionário do Corretor",
    "📖 Biblioteca",
    "🏫 Universidade",
    "👤 Análise de Perfil",
    "📅 Agenda de Eventos"
])

if aba == "🏠 Avaliação de Imóveis":
    st.title("🏠 Avaliador de Imóveis com IA")
    st.write("Preencha os dados abaixo para estimar o valor do imóvel")

    with st.form("form_avaliacao"):
        col1, col2 = st.columns(2)
        with col1:
            estado = st.selectbox("Estado", list(estados_cidades.keys()))
            cidade = st.selectbox("Cidade", estados_cidades[estado])
            bairro = st.selectbox("Bairro", ['Centro', 'Moema', 'Itaim', 'Bela Vista', 'Tatuapé'])
            tipo = st.selectbox("Tipo de Imóvel", ['Apartamento', 'Casa', 'Lote/Área', 'Comercial'])
            m2 = st.slider("Área (m²)", 30, 2000, 85)
            ano = st.slider("Ano de Construção", 1970, 2024, 2010)
        with col2:
            quartos = st.slider("Quartos", 0, 10, 2)
            banheiros = st.slider("Banheiros", 0, 6, 2)
            garagem = st.slider("Vagas de Garagem", 0, 6, 1)
            mobilia = st.selectbox("Mobilia", ['Nenhuma', 'Armários', 'Completa'])

        submit = st.form_submit_button("Avaliar")

    if submit:
        mobilia_valor = le_dict['mobilia'].transform([mobilia])[0]

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
            'mobilia': mobilia_valor
        }])

        entrada = entrada[colunas_modelo]

        valor_estimado = modelo.predict(entrada)[0]
        valor_m2 = valor_estimado / m2

        st.success(f"💰 Valor estimado do imóvel: R$ {valor_estimado:,.2f}")
        st.info(f"📏 Valor por m²: R$ {valor_m2:,.2f}")

        st.subheader(f"📊 Estatísticas de imóveis em {bairro}")
        bairro_cod = le_dict['bairro'].transform([bairro])[0]
        df_bairro = df[df['bairro'] == bairro_cod]

        valor_min = df_bairro['valor'].min()
        valor_max = df_bairro['valor'].max()
        valor_med = df_bairro['valor'].mean()

        st.markdown(f"**Faixa de valores:** R$ {valor_min:,.2f} — R$ {valor_max:,.2f}")
        st.markdown(f"**Valor médio no bairro:** R$ {valor_med:,.2f}")

        hist = alt.Chart(df_bairro).mark_bar().encode(
            alt.X("valor", bin=alt.Bin(maxbins=30), title="Valor (R$)"),
            alt.Y("count()", title="Número de imóveis")
        ).properties(width=700, height=300, title="Distribuição de Preços no Bairro")

        st.altair_chart(hist, use_container_width=True)

elif aba == "📝 Criação de Contratos":
    st.title("📝 Módulo em construção")
elif aba == "☀️ Posição Solar":
    st.title("☀️ Módulo em construção")
elif aba == "📈 Calculadora Financeira":
    st.title("📈 Módulo em construção")
elif aba == "📊 Simulador de Investimento":
    st.title("📊 Módulo em construção")
elif aba == "📉 Consulta de Índices":
    st.title("📉 Módulo em construção")
elif aba == "📰 Notícias do Mercado":
    st.title("📰 Módulo em construção")
elif aba == "📚 Dicionário do Corretor":
    st.title("📚 Módulo em construção")
elif aba == "📖 Biblioteca":
    st.title("📖 Módulo em construção")
elif aba == "🏫 Universidade":
    st.title("🏫 Módulo em construção")
elif aba == "👤 Análise de Perfil":
    st.title("👤 Módulo em construção")
elif aba == "📅 Agenda de Eventos":
    st.title("📅 Módulo em construção")

