# 🌍 Nzola — Plataforma Inteligente de Turismo Sustentável

## 🧠 Visão Geral

O **Nzola** é uma plataforma web interativa que utiliza **Inteligência Artificial, Machine Learning e Geolocalização** para recomendar **destinos turísticos sustentáveis** com base em dados reais e preferências do usuário.

O sistema foi desenvolvido no âmbito do **Capstone Final do Bootcamp de Engenharia de Dados e IA**, com foco em **turismo responsável e sustentável**.


## 🚀 Arquitetura Geral
Nzola/
├── __pycache__/
├── app.py
├── recomendador.py
├── pytorch_recomendador.py
├── csvrapido.py
├── csvs/
│   ├── destinos.csv
│   ├── perfil_turista.csv
│   └── feedback.csv
├── templates/
│   ├── index.html
│   └── resultados.html
├── static/
│   └── style.css
└── docs/
    ├── README.md
    ├── vision.md
    ├── Proposta de idea.md
    ├── Folha tecnica.md
    └── destinos.csv
```Nzola/
├── __pycache__/
├── app.py
├── recomendador.py
├── pytorch_recomendador.py
├── csvrapido.py
├── csvs/
│   ├── destinos.csv
│   ├── perfil_turista.csv
│   └── feedback.csv
├── templates/
│   ├── index.html
│   └── resultados.html
├── static/
│   └── style.css
└── docs/
    ├── README.md
    ├── vision.md
    ├── Proposta de idea.md
    ├── Folha tecnica.md
    └── destinos.csv
|
 Funcionalidades Principais

- 🌎 **Geolocalização automática:** o navegador obtém as coordenadas do usuário.
- 🤖 **Recomendação inteligente:** sugestões de destinos baseadas em preferências, localização e impacto ambiental.
- 🧩 **Sistema de “segunda opinião” (PyTorch):** modelo neural que reavalia as recomendações e fornece uma escolha mais personalizada.
- 📊 **Análise de dados:** os destinos são avaliados quanto a custo, ecossistema, impacto ambiental e acessibilidade.
- 🖥 **Interface responsiva:** página dinâmica com HTML, CSS e JavaScript, comunicação via AJAX com o Flask.

---

## 🖥 1. Backend

### 🔹 Tecnologias
- **Python + Flask** — Framework web principal.
- **pandas** — Manipulação dos dados (DataFrames com destinos, custos, tags, etc).
- **scikit-learn (sklearn)** — Cálculo de similaridade (TF-IDF e cosine similarity).
- **PyTorch** — Implementação da “segunda camada” (modelo neural MLP).
- **geopy / haversine** — Cálculo de distâncias geográficas.
- **json** — Troca de dados entre backend e frontend.

### 🔹 Principais rotas
| Rota | Função |
|------|--------|
| `/` | Página inicial (formulário e geolocalização) |
| `/resultados` | Mostra recomendações iniciais (sklearn) |
| `/segunda_opiniao` | Modelo PyTorch gera a recomendação final |

---

## 🌐 2. Frontend

### 🔹 Tecnologias
- **HTML5 + CSS3 + JavaScript** — Interface responsiva e interativa.
- **Jinja2** — Motor de templates do Flask (renderização dinâmica).
- **AJAX (fetch API)** — Envia e recebe dados sem recarregar a página.
- **Session Storage** — Guarda latitude e longitude localmente no navegador.

---

## 🗺 3. Geolocalização

- Implementada com **JavaScript (navigator.geolocation)**.
- Flask recebe as coordenadas do usuário e calcula distâncias reais até cada destino.
- Distâncias são normalizadas e usadas como variável de entrada no modelo de recomendação.

---

## 🤖 4. Inteligência Artificial / Machine Learning

### 🔹 Primeira camada — Recomendador Base
- Baseado em **TF-IDF** e **similaridade de cosseno** (scikit-learn).  
- Identifica destinos similares conforme tags e preferências do usuário.  
- Retorna uma lista inicial classificada por compatibilidade e proximidade.

### 🔹 Segunda camada — Módulo PyTorch
- Modelo de rede neural (MLP) leve, com pesos simulando uma “segunda opinião”.  
- Recebe as top recomendações e ajusta com base em dados contextuais.  
- Permite ao usuário solicitar uma análise mais personalizada ("Ajude-me a decidir").  

---

## 🗃 5. Dados

- Estruturados em **DataFrames pandas**.
- Atributos:
  - Nome do destino
  - Província
  - Tipo de ecossistema (floresta, praia, montanha, urbano)
  - Custo médio
  - Impacto ambiental
  - Número de visitantes anuais
  - Tags (trilha, praia, cultural, safari, etc.)
- Armazenados em formato **CSV** e carregados dinamicamente pelo Flask.

---

## ⚙️ 6. Ambiente de Desenvolvimento

- **Python 3.9+**
- **VSCode** como IDE principal.
- Execução local no **Windows 11**
- Dependências listadas em `requirements.txt`

---

## 📦 Instalação e Execução

```bash
# Clonar o repositório
git clone https://github.com/<usuario>/nzola.git
cd nzola