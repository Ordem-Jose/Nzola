# app.py
from flask import Flask, render_template, request
from recomendador import recomendar_destinos, carregar_destinos

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recomendar", methods=["POST"])
def recomendar():
    preferencias = request.form["preferencias"]
    max_impacto = float(request.form["max_impacto"])
    destinos_df = carregar_destinos()
    resultados = recomendar_destinos(preferencias, destinos_df, max_impacto)

    return render_template("resultados.html", resultados=resultados.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)
