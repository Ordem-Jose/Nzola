# recomendador.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recomendar_destinos(preferencias, destinos_df, max_impacto=3.0):
    """
    Retorna destinos sustentáveis e compatíveis com as preferências do utilizador.
    """

    # Filtra por sustentabilidade
    destinos_filtrados = destinos_df[destinos_df["Impacto_Ambiental"] <= max_impacto]

    # Calcula similaridade com base nas tags
    tfidf = TfidfVectorizer()
    matriz_tfidf = tfidf.fit_transform(destinos_filtrados["Tags"])
    consulta = tfidf.transform([preferencias])
    similaridades = cosine_similarity(consulta, matriz_tfidf).flatten()

    # Cria DataFrame com resultados
    destinos_filtrados["Similaridade"] = similaridades
    recomendacoes = destinos_filtrados.sort_values(by="Similaridade", ascending=False).head(5)
    return recomendacoes[["Destino", "Província", "Ecossistema", "Custo_Médio", "Impacto_Ambiental", "Similaridade"]]

def carregar_destinos():
    # Dados de exemplo — substituir por CSV real
    return pd.DataFrame([
        {"Destino": "Parque Nacional da Kissama", "Província": "Bengo", "Ecossistema": "Savanas", "Custo_Médio": 50, "Impacto_Ambiental": 2.0, "Tags": "safari natureza ecoturismo"},
        {"Destino": "Reserva de Cangandala", "Província": "Malanje", "Ecossistema": "Floresta", "Custo_Médio": 40, "Impacto_Ambiental": 3.5, "Tags": "floresta biodiversidade conservação"},
        {"Destino": "Praia Morena", "Província": "Benguela", "Ecossistema": "Litoral", "Custo_Médio": 30, "Impacto_Ambiental": 2.5, "Tags": "praia mar turismo"},
        {"Destino": "Serra da Leba", "Província": "Huíla", "Ecossistema": "Montanha", "Custo_Médio": 60, "Impacto_Ambiental": 1.5, "Tags": "montanha paisagem aventura"},
        {"Destino": "Morro do Moco", "Província": "Huambo", "Ecossistema": "Montanha", "Custo_Médio": 45, "Impacto_Ambiental": 1.0, "Tags": "montanha trilha natureza"}
    ])
