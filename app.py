from flask import Flask, render_template, request
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

# Carregar os modelos e dados
modelo = joblib.load("models/modelo_avaliacao.joblib")
le_dict = joblib.load("models/label_encoders.joblib")
colunas_modelo = joblib.load("models/colunas_modelo.joblib")
estados_cidades_bairros = joblib.load("models/estados_cidades_bairros.joblib")

@app.route("/")
def index():
    estados = list(estados_cidades_bairros.keys())
    return render_template("index.html", estados=estados)

@app.route("/resultado", methods=["POST"])
def resultado():
    estado = request.form["estado"]
    cidade = request.form["cidade"]
    bairro = request.form["bairro"]
    tipo = request.form["tipo"]
    m2 = int(request.form["m2"])
    quartos = int(request.form["quartos"])
    banheiros = int(request.form["banheiros"])
    garagem = int(request.form["garagem"])
    ano = int(request.form["ano"])
    mobilia = request.form["mobilia"]

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

        # Gerar gráfico
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
        plt.tight_layout()
        plot_path = "static/plot.png"
        fig.savefig(plot_path)
        plt.close()

        return render_template("resultado.html", valor=valor_estimado, valor_m2=valor_m2, imagem=plot_path)

    except Exception as e:
        return f"Erro: {str(e)}"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)