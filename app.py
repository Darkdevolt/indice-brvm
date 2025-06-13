import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="Analyse BRVM", layout="wide")

@st.cache_data
def scraper_brvm():
    url = "https://www.brvm.org/fr/cours-actions/0"
    response = requests.get(url, verify=False)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")

    if not table:
        return pd.DataFrame()

    headers = [th.text.strip() for th in table.find_all("th")]
    rows = []

    for tr in table.find_all("tr")[1:]:
        cols = [td.text.strip().replace("\xa0", "") for td in tr.find_all("td")]
        if cols:
            rows.append(cols)

    df = pd.DataFrame(rows, columns=headers)
    return df

st.title("📊 Analyse des données de la BRVM")
st.write("⚠️ Connexion non sécurisée (SSL désactivé temporairement).")

try:
    df = scraper_brvm()

    if df.empty:
        st.warning("⚠️ Tableau introuvable ou vide.")
    else:
        st.success("✅ Données récupérées avec succès.")
        st.write("Colonnes récupérées :", df.columns.tolist())

        with st.expander("🔎 Afficher les données brutes"):
            st.dataframe(df)

        # Vérifie la présence des colonnes avant traitement
        colonnes_requises = ["Valeur", "Dernier", "Variation", "Volume", "Capitalisation boursière"]
        colonnes_disponibles = [col for col in colonnes_requises if col in df.columns]

        if "Variation" in df.columns:
            for col in colonnes_disponibles:
                df[col] = df[col].str.replace(",", "").str.replace("%", "").str.replace(" ", "").replace("", "0").astype(float)

            top_variation = df.sort_values(by="Variation", ascending=False).head(10)
            worst_variation = df.sort_values(by="Variation").head(10)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🔼 Top 10 hausses")
                st.dataframe(top_variation[["Valeur", "Dernier", "Variation"]])
            with col2:
                st.markdown("### 🔽 Top 10 baisses")
                st.dataframe(worst_variation[["Valeur", "Dernier", "Variation"]])

            st.markdown("---")
            st.bar_chart(top_variation.set_index("Valeur")["Variation"])
        else:
            st.error("❌ Colonnes nécessaires non présentes dans la table récupérée.")

except Exception as e:
    st.error("❌ Une erreur s’est produite lors du chargement des données.")
    st.exception(e)
