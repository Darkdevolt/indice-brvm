import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import urllib3

# 🔒 Désactive les avertissements SSL (⚠️ à éviter en production)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

st.set_page_config(page_title="Analyse BRVM", layout="wide")

@st.cache_data
def scraper_brvm():
    url = "https://www.brvm.org/fr/cours-actions/0"
    
    # ⚠️ SSL désactivé ici pour contourner l’erreur sur Streamlit Cloud
    response = requests.get(url, verify=False)

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")

    headers = [th.text.strip() for th in table.find_all("th")]
    rows = []

    for tr in table.find_all("tr")[1:]:
        cols = [td.text.strip().replace("\xa0", "") for td in tr.find_all("td")]
        if cols:
            rows.append(cols)

    df = pd.DataFrame(rows, columns=headers)
    return df

st.title("📊 Analyse des données de la BRVM")
st.write("⚠️ *Connexion non sécurisée au site de la BRVM (SSL désactivé temporairement).*")

try:
    df = scraper_brvm()

    if not df.empty:
        st.success("✅ Données récupérées avec succès.")

        with st.expander("🔎 Afficher les données brutes"):
            st.dataframe(df)

        st.subheader("📈 Analyse rapide")

        numeric_columns = ['Dernier', 'Variation', 'Volume', 'Capitalisation boursière']
        for col in numeric_columns:
            df[col] = df[col].str.replace(",", "").str.replace("%", "").str.replace(" ", "").replace("", "0").astype(float)

        # Classement par performance
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
        st.warning("⚠️ Données introuvables ou tableau vide.")
except Exception as e:
    st.error("❌ Une erreur s’est produite lors du chargement des données.")
    st.exception(e)
