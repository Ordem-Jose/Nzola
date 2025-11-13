import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def recomendar_destinos(preferencias, destinos_df, user_lat, user_lon, max_impacto=3.0, top_n=5):
    destinos_filtrados = destinos_df[destinos_df["Impacto_Ambiental"] <= max_impacto].copy()

    # Similaridade de tags
    tfidf = TfidfVectorizer()
    matriz_tfidf = tfidf.fit_transform(destinos_filtrados["Tags"])
    consulta = tfidf.transform([preferencias])
    destinos_filtrados["Similaridade"] = cosine_similarity(consulta, matriz_tfidf).flatten()

    # Distância geográfica
    destinos_filtrados["Distancia_km"] = destinos_filtrados.apply(
        lambda row: haversine(user_lat, user_lon, row["Latitude"], row["Longitude"]), axis=1
    )

    # Combina similaridade e proximidade
    destinos_filtrados["Distancia_norm"] = 1 - (destinos_filtrados["Distancia_km"] / destinos_filtrados["Distancia_km"].max())
    destinos_filtrados["Pontuacao"] = 0.6*destinos_filtrados["Similaridade"] + 0.4*destinos_filtrados["Distancia_norm"]

    recomendacoes = destinos_filtrados.sort_values(by="Pontuacao", ascending=False).head(top_n)
    return recomendacoes[[
        "Nome","Provincia","Tipo_Ecosistema","Custo_Medio","Impacto_Ambiental",
        "Visitantes_Anuais","Tags","Distancia_km","Similaridade","Pontuacao"
    ]]

def carregar_destinos_robustos():
    # Lê os dados do CSV
    df = pd.read_csv("destinos_turisticos.csv")
    return df