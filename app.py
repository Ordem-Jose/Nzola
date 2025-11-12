from flask import Flask, render_template, request, jsonify
from recomendador import recomendar_destinos, carregar_destinos_robustos

app = Flask(__name__)

# Carrega os destinos uma vez
destinos_df = carregar_destinos_robustos()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        preferencias = request.form.get("preferencias")
        user_lat = float(request.form.get("latitude"))
        user_lon = float(request.form.get("longitude"))

        resultados = recomendar_destinos(preferencias, destinos_df, user_lat, user_lon)
        resultados_dict = resultados.to_dict(orient="records")
        destinos_recomendados = [r['Nome'] for r in resultados_dict]

        return render_template(
            "resultados.html",
            resultados=resultados_dict,
            destinos_recomendados=destinos_recomendados
        )
    return render_template("index.html")

@app.route("/segunda_opiniao", methods=["POST"])
def segunda_opiniao():
    data = request.get_json()
    categoria = data.get("categoria")
    user_lat = float(data.get("latitude"))
    user_lon = float(data.get("longitude"))
    destinos_selecionados = data.get("destinos")

    # Filtra os destinos enviados e aplica o recomendador com PyTorch ou pontuação
    df_filtrado = destinos_df[destinos_df["Nome"].isin(destinos_selecionados)]
    resultados = recomendar_destinos(categoria, df_filtrado, user_lat, user_lon, top_n=3)
    
    # Formata o resultado para enviar como string
    sugestao = "<br>".join([
        f"{r['Nome']} ({r['Provincia']}, {r['Tipo_Ecosistema']}) - Pontuação final: {r['Pontuacao']:.2f}"
        for _, r in resultados.iterrows()
    ])

    return jsonify({"sugestao": sugestao})

if __name__ == "__main__":
    app.run(debug=True)
