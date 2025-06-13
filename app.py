import streamlit as st
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time

st.title("Analyse des actions BRVM en temps réel")

@st.cache_data
def scrap_brvm():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    url = "https://www.brvm.org/fr/cours-actions/0"
    driver.get(url)
    time.sleep(5)  # Laisse le temps de charger les données JS

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()

    table = soup.find('table', {'id': 'brvm_table_cours_action'})
    rows = table.find_all('tr')[1:]  # Skip header

    data = []
    for row in rows:
        cols = row.find_all('td')
        if len(cols) >= 6:
            titre = cols[0].text.strip()
            dernier_cours = cols[1].text.strip().replace(',', '.')
            variation = cols[4].text.strip().replace(',', '.')
            volume = cols[5].text.strip().replace(',', '').replace(' ', '')

            data.append({
                'Titre': titre,
                'Dernier Cours': float(dernier_cours) if dernier_cours else 0,
                'Variation (%)': float(variation.replace('%', '')) if variation else 0,
                'Volume': int(volume) if volume else 0
            })

    df = pd.DataFrame(data)
    return df

df = scrap_brvm()

st.subheader("📊 Données BRVM")
st.dataframe(df)

st.subheader("📈 Analyse des variations")
top_gains = df.sort_values(by='Variation (%)', ascending=False).head(5)
top_losses = df.sort_values(by='Variation (%)').head(5)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Top 5 Hausse")
    st.dataframe(top_gains)

with col2:
    st.markdown("### Top 5 Baisse")
    st.dataframe(top_losses)

st.subheader("🔍 Filtrer par volume échangé")
volume_min = st.slider("Volume minimum", min_value=0, max_value=int(df['Volume'].max()), value=1000)
filtered = df[df['Volume'] >= volume_min]
st.dataframe(filtered)
