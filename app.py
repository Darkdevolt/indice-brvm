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

st.title("📊 Analyse Automatique des Données de la BRVM")
st.write("⚠️ Connexion SSL désactivée temporairement pour récupérer les données depuis le site officiel.")

try:
    df = scraper_brvm()

    if df.empty:
        st.warning("⚠️ Aucune donnée trouvée. La table peut avoir changé ou être temporairement indisponible.")
    else:
        st.success("✅ Données récupérées avec succès depuis le site de la BRVM.")
        
        st.subheader("🧾 Colonnes détectées")
        st.write(df.columns.tolist())

        st.subheader("🔍 Aperçu des premières lignes")
        st.dataframe(df.head())

        # Correspondance possible avec les noms des colonnes
        # On détecte dynamiquement les noms valides
        possible_cols = df.columns.str.lower()

        col_nom = next((col for col in df.columns if "valeur" in col.lower() or "nom" in col.lower()), None)
        col_cours = next((col for col in df.columns if "dernier" in col.lower() or "cours" in col.lower()), None)
        col_var = next((col for col in df.columns if "variation" in col.lower()), None)

        if all([col_nom, col_cours, col_var]):
            st.markdown("### ✅ Colonnes utilisées pour l'analyse")
            st.write(f"- Nom/Valeur : `{col_nom}`")
            st.write(f"- Cours Dernier : `{col_cours}`")
            st.write(f"- Variation : `{col_var}`")

            # Nettoyage
            df[col_var] = df[col_var].str.replace(",", "").str.replace("%", "").str.replace(" ", "").replace("", "0").astype(float)
            df[col_cours] = df[col_cours].str.replace(",", "").str.replace(" ", "").replace("", "0").astype(float)

            # Classement des variations
            top = df.sort_values(by=col_var, ascending=False).head(10)
            flop = df.sort_values(by=col_var).head(10)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🔼 Top 10 Performers")
                st.dataframe(top[[col_nom, col_cours, col_var]])
            with col2:
                st.markdown("### 🔽 Flop 10")
                st.dataframe(flop[[col_nom, col_cours, col_var]])

            st.markdown("---")
            st.bar_chart(top.set_index(col_nom)[col_var])

        else:
            st.error("❌ Colonnes essentielles manquantes. Impossible d’analyser.")
except Exception as e:
    st.error("❌ Une erreur s’est produite lors du chargement des données.")
    st.exception(e)
