import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

def get_brvm_data():
"""
Récupère les données des actions et indices BRVM depuis Sika Finance
Structure attendue: Nom, Haut, Bas, Dernier, Volume, Variation jour
"""
try:
# Headers pour simuler un navigateur
headers = {
'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
'Accept-Language': 'fr-FR,fr;q=0.9,en;q=0.8',
'Accept-Encoding': 'gzip, deflate',
'Connection': 'keep-alive',
}

```
    # Récupération de la page palmares
    url = "<https://www.sikafinance.com/marches/palmares>"
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    # Parse du HTML
    soup = BeautifulSoup(response.content, 'html.parser')

    # Recherche des tableaux contenant les données
    tables = soup.find_all('table')
    all_data = []

    for table in tables:
        # Extraction des en-têtes
        headers_row = table.find('thead')
        if not headers_row:
            headers_row = table.find('tr')

        if headers_row:
            headers = [th.get_text(strip=True) for th in headers_row.find_all(['th', 'td'])]

            # Vérifier si c'est le bon tableau (contient les colonnes attendues)
            expected_cols = ['nom', 'haut', 'bas', 'dernier', 'volume', 'variation']
            header_text = ' '.join(headers).lower()

            if any(col in header_text for col in expected_cols):
                # Recherche des lignes de données
                tbody = table.find('tbody')
                rows = tbody.find_all('tr') if tbody else table.find_all('tr')[1:]  # Skip header row

                for row in rows:
                    cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
                    if len(cells) >= 6 and cells[0]:  # Au moins 6 colonnes avec un nom
                        all_data.append(cells[:6])  # Prendre exactement 6 colonnes

    # Si pas de données trouvées dans les tableaux, essayer une approche différente
    if not all_data:
        # Recherche par classes CSS ou divs spécifiques
        stock_elements = soup.find_all(['tr', 'div'], class_=re.compile(r'(stock|action|valeur|quote)', re.I))

        for element in stock_elements:
            # Extraire tous les nombres et textes
            text_content = element.get_text(separator=' ', strip=True)

            # Pattern pour extraire: Nom + 5 valeurs numériques
            pattern = r'([A-Z][A-Z0-9\\s]+?)\\s+([\\d,.-]+)\\s+([\\d,.-]+)\\s+([\\d,.-]+)\\s+([\\d,.-]+)\\s*([-+]?[\\d,.-]+%?)'
            match = re.search(pattern, text_content)

            if match:
                groups = list(match.groups())
                if len(groups) >= 6:
                    all_data.append(groups)

    # Création du DataFrame si des données ont été trouvées
    if all_data:
        # Définir les colonnes exactes selon la structure Sika Finance
        columns = ['Nom', 'Haut', 'Bas', 'Dernier', 'Volume', 'Variation_jour']

        # Filtrer les lignes qui ont exactement 6 colonnes
        filtered_data = [row for row in all_data if len(row) == 6]

        if filtered_data:
            df = pd.DataFrame(filtered_data, columns=columns)

            # Nettoyage des données numériques
            numeric_columns = ['Haut', 'Bas', 'Dernier', 'Volume', 'Variation_jour']

            for col in numeric_columns:
                if col in df.columns:
                    # Nettoyage spécifique selon le type de colonne
                    df[col] = df[col].astype(str)

                    if col == 'Volume':
                        # Pour le volume, garder les grands nombres
                        df[col] = df[col].str.replace(r'[^\\d,.]', '', regex=True)
                        df[col] = df[col].str.replace(',', '')
                    elif col == 'Variation_jour':
                        # Pour la variation, traiter les pourcentages
                        df[col] = df[col].str.replace(r'[^\\d.,%-]', '', regex=True)
                        df[col] = df[col].str.replace('%', '')
                        df[col] = df[col].str.replace(',', '.')
                    else:
                        # Pour les prix (Haut, Bas, Dernier)
                        df[col] = df[col].str.replace(r'[^\\d,.]', '', regex=True)
                        df[col] = df[col].str.replace(',', '.')

                    # Conversion en numérique
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Nettoyage des noms
            df['Nom'] = df['Nom'].str.strip()

            # Supprimer les lignes avec trop de valeurs manquantes
            df = df.dropna(subset=['Nom', 'Dernier'])

            return df

    # Si aucune donnée structurée n'est trouvée, retourner des données de démonstration
    demo_data = {
        'Nom': ['ECOBANK CI', 'SONATEL', 'CFAO MOTORS', 'BICICI', 'ORANGE CI', 'BOA MALI', 'SAFCA CI', 'CROWN SIEM'],
        'Haut': [7500, 15200, 990, 8900, 12500, 1850, 700, 520],
        'Bas': [7350, 14800, 950, 8750, 12200, 1820, 680, 500],
        'Dernier': [7450, 15100, 975, 8850, 12350, 1840, 695, 515],
        'Volume': [1250, 890, 2100, 780, 1650, 920, 450, 320],
        'Variation_jour': [1.35, -0.66, 2.41, 0.57, -1.20, 1.10, 2.21, 3.00]
    }

    st.warning("⚠️ Données de démonstration affichées. Vérifiez la connexion au site Sika Finance.")
    return pd.DataFrame(demo_data)

except requests.RequestException as e:
    st.error(f"❌ Erreur de connexion à Sika Finance: {e}")
    return None
except Exception as e:
    st.error(f"❌ Erreur lors du traitement des données: {e}")
    return None

```

def get_brvm_index():
"""
Fonction de compatibilité - utilise get_brvm_data()
"""
return get_brvm_data()

def main():
st.set_page_config(
page_title="BRVM Indices Dashboard",
page_icon="📈",
layout="wide",
initial_sidebar_state="expanded"
)

```
st.title("📈 Tableau de Bord BRVM - Sika Finance")
st.markdown("*Données en temps réel depuis Sika Finance*")
st.markdown("---")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox(
    "Choisir une page",
    ["Palmarès BRVM", "Analyse Détaillée", "À propos"]
)

if page == "Palmarès BRVM":
    show_overview()
elif page == "Analyse Détaillée":
    show_detailed_analysis()
else:
    show_about()

```

def show_overview():
st.header("📊 Palmarès BRVM - Sika Finance")

```
# Bouton de rafraîchissement
if st.button("🔄 Actualiser les données"):
    st.cache_data.clear()

# Chargement des données
with st.spinner("Chargement des données depuis Sika Finance..."):
    data = get_brvm_data()

if data is not None and not data.empty:
    st.success(f"✅ Données récupérées avec succès! ({len(data)} valeurs)")

    # Affichage de la date de mise à jour
    st.info(f"🕒 Dernière mise à jour: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Métriques principales basées sur les vraies colonnes
    col1, col2, col3, col4 = st.columns(4)

    # Analyse des variations jour
    if 'Variation_jour' in data.columns and data['Variation_jour'].notna().any():
        positive_changes = len(data[data['Variation_jour'] > 0])
        negative_changes = len(data[data['Variation_jour'] < 0])
        avg_change = data['Variation_jour'].mean()
        max_change = data['Variation_jour'].max()

        with col1:
            st.metric("📈 Valeurs en hausse", positive_changes)
        with col2:
            st.metric("📉 Valeurs en baisse", negative_changes)
        with col3:
            st.metric("📊 Variation moy.", f"{avg_change:.2f}%")
        with col4:
            st.metric("🚀 Plus forte hausse", f"{max_change:.2f}%")

    # Métriques de volume et prix
    if 'Volume' in data.columns and data['Volume'].notna().any():
        total_volume = data['Volume'].sum()
        avg_volume = data['Volume'].mean()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("💰 Volume total", f"{total_volume:,.0f}")
        with col2:
            st.metric("📊 Volume moyen", f"{avg_volume:,.0f}")

    # Tableau des données avec filtres
    st.subheader("📋 Données du Palmarès BRVM")

    # Filtre de recherche
    search_term = st.text_input("🔍 Rechercher une valeur:", "").upper()
    filtered_data = data.copy()

    if search_term:
        filtered_data = filtered_data[filtered_data['Nom'].str.upper().str.contains(search_term, na=False)]

    # Filtre par performance
    perf_filter = st.selectbox(
        "Filtrer par performance:",
        ["Toutes", "En hausse (+)", "En baisse (-)", "Stables (0)"]
    )

    if perf_filter != "Toutes" and 'Variation_jour' in filtered_data.columns:
        if perf_filter == "En hausse (+)":
            filtered_data = filtered_data[filtered_data['Variation_jour'] > 0]
        elif perf_filter == "En baisse (-)":
            filtered_data = filtered_data[filtered_data['Variation_jour'] < 0]
        else:  # Stables
            filtered_data = filtered_data[filtered_data['Variation_jour'] == 0]

    # Style du tableau avec formatage des colonnes
    def format_currency(val):
        if pd.isna(val):
            return ''
        return f"{val:,.0f}"

    def format_percentage(val):
        if pd.isna(val):
            return ''
        color = 'green' if val > 0 else 'red' if val < 0 else 'black'
        sign = '+' if val > 0 else ''
        return f'<span style="color: {color}; font-weight: bold">{sign}{val:.2f}%</span>'

    # Affichage du tableau formaté
    display_data = filtered_data.copy()

    # Format des colonnes numériques
    for col in ['Haut', 'Bas', 'Dernier']:
        if col in display_data.columns:
            display_data[col] = display_data[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")

    if 'Volume' in display_data.columns:
        display_data['Volume'] = display_data['Volume'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")

    st.dataframe(display_data, use_container_width=True)

    # Graphiques des performances
    if 'Variation_jour' in data.columns and 'Nom' in data.columns and data['Variation_jour'].notna().any():
        st.subheader("📈 Visualisations des Performances")

        col1, col2 = st.columns(2)

        with col1:
            # Top 10 des hausses
            top_gains = data.nlargest(10, 'Variation_jour')
            if not top_gains.empty:
                fig_gains = px.bar(
                    top_gains,
                    x='Nom',
                    y='Variation_jour',
                    color='Variation_jour',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    title="🔥 Top 10 des plus fortes hausses",
                    labels={'Variation_jour': 'Variation (%)', 'Nom': 'Valeurs'}
                )
                fig_gains.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_gains, use_container_width=True)

        with col2:
            # Top 10 des baisses
            top_losses = data.nsmallest(10, 'Variation_jour')
            if not top_losses.empty:
                fig_losses = px.bar(
                    top_losses,
                    x='Nom',
                    y='Variation_jour',
                    color='Variation_jour',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    title="📉 Top 10 des plus fortes baisses",
                    labels={'Variation_jour': 'Variation (%)', 'Nom': 'Valeurs'}
                )
                fig_losses.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_losses, use_container_width=True)

        # Distribution des variations
        if len(data) > 5:
            fig_hist = px.histogram(
                data,
                x='Variation_jour',
                nbins=20,
                title="📊 Distribution des variations journalières",
                labels={'Variation_jour': "Variation (%)", 'count': 'Nombre de valeurs'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    # Analyse des volumes si disponible
    if 'Volume' in data.columns and 'Nom' in data.columns and data['Volume'].notna().any():
        st.subheader("💰 Analyse des Volumes")

        # Top volumes
        top_volumes = data.nlargest(10, 'Volume')
        fig_vol = px.bar(
            top_volumes,
            x='Nom',
            y='Volume',
            title="🏆 Top 10 des volumes d'échanges",
            labels={'Volume': 'Volume', 'Nom': 'Valeurs'}
        )
        fig_vol.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_vol, use_container_width=True)

    # Analyse Prix vs Volume
    if all(col in data.columns for col in ['Dernier', 'Volume', 'Variation_jour']):
        st.subheader("🎯 Analyse Prix vs Volume vs Performance")

        fig_scatter = px.scatter(
            data,
            x='Volume',
            y='Dernier',
            color='Variation_jour',
            size='Volume',
            hover_name='Nom',
            color_continuous_scale=['red', 'yellow', 'green'],
            title="📈 Relation Prix / Volume / Performance",
            labels={
                'Volume': 'Volume d\\'échanges',
                'Dernier': 'Prix (Dernier)',
                'Variation_jour': 'Variation (%)'
            }
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

else:
    st.error("❌ Impossible de récupérer les données depuis Sika Finance.")
    st.info("💡 Vérifiez votre connexion internet ou réessayez plus tard.")

```

def show_detailed_analysis():
st.header("🔍 Analyse Détaillée")

```
data = get_brvm_data()

if data is not None and not data.empty:
    # Sélection d'une valeur
    selected_stock = st.selectbox("Choisir une valeur:", data['Nom'].tolist())

    # Données de la valeur sélectionnée
    stock_data = data[data['Nom'] == selected_stock].iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"📊 {selected_stock}")

        # Affichage des métriques selon les vraies colonnes
        if pd.notna(stock_data['Dernier']):
            st.metric("💰 Prix Dernier", f"{stock_data['Dernier']:,.0f}")

        if pd.notna(stock_data['Haut']):
            st.metric("📈 Plus Haut", f"{stock_data['Haut']:,.0f}")

        if pd.notna(stock_data['Bas']):
            st.metric("📉 Plus Bas", f"{stock_data['Bas']:,.0f}")

        if pd.notna(stock_data['Volume']):
            st.metric("💱 Volume", f"{stock_data['Volume']:,.0f}")

        if pd.notna(stock_data['Variation_jour']):
            delta = stock_data['Variation_jour']
            st.metric("📊 Variation Jour", f"{delta:+.2f}%", delta=f"{delta:.2f}%")

    with col2:
        # Graphique OHLC simplifié
        if all(pd.notna(stock_data[col]) for col in ['Haut', 'Bas', 'Dernier']):
            # Créer un graphique en chandelier simplifié
            fig = go.Figure()

            # Barre représentant l'amplitude Haut-Bas
            fig.add_trace(go.Scatter(
                x=[selected_stock],
                y=[stock_data['Haut']],
                mode='markers',
                marker=dict(color='green', size=15, symbol='triangle-up'),
                name='Plus Haut'
            ))

            fig.add_trace(go.Scatter(
                x=[selected_stock],
                y=[stock_data['Bas']],
                mode='markers',
                marker=dict(color='red', size=15, symbol='triangle-down'),
                name='Plus Bas'
            ))

            fig.add_trace(go.Scatter(
                x=[selected_stock],
                y=[stock_data['Dernier']],
                mode='markers',
                marker=dict(color='blue', size=20, symbol='diamond'),
                name='Prix Dernier'
            ))

            fig.update_layout(
                title=f"📊 Résumé de {selected_stock}",
                yaxis_title="Prix",
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

    # Comparaison avec le marché
    st.subheader("📊 Positionnement vs Marché")

    if 'Variation_jour' in data.columns and data['Variation_jour'].notna().any():
        market_avg = data['Variation_jour'].mean()
        stock_var = stock_data['Variation_jour'] if pd.notna(stock_data['Variation_jour']) else 0

        comparison_data = pd.DataFrame({
            'Métrique': ['Marché (moyenne)', selected_stock],
            'Variation': [market_avg, stock_var]
        })

        fig = px.bar(
            comparison_data,
            x='Métrique',
            y='Variation',
            color='Variation',
            color_continuous_scale=['red', 'yellow', 'green'],
            title=f"🎯 {selected_stock} vs Marché"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Analyse du marché global
    st.subheader("🌍 Analyse du Marché Global")

    if 'Variation_jour' in data.columns and data['Variation_jour'].notna().any():
        col1, col2, col3 = st.columns(3)

        with col1:
            market_trend = "Haussière 📈" if data['Variation_jour'].mean() > 0 else "Baissière 📉"
            st.metric("🎯 Tendance générale", market_trend)

        with col2:
            volatility = data['Variation_jour'].std()
            st.metric("📊 Volatilité", f"{volatility:.2f}%")

        with col3:
            range_var = data['Variation_jour'].max() - data['Variation_jour'].min()
            st.metric("📏 Amplitude", f"{range_var:.2f}%")

        # Box plot des variations
        fig_box = px.box(
            data,
            y='Variation_jour',
            title="📊 Distribution des variations du marché BRVM"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Statistiques descriptives
    st.subheader("📈 Statistiques Descriptives du Marché")
    numeric_columns = ['Haut', 'Bas', 'Dernier', 'Volume', 'Variation_jour']
    available_cols = [col for col in numeric_columns if col in data.columns]

    if available_cols:
        stats_df = data[available_cols].describe()
        st.dataframe(stats_df, use_container_width=True)

else:
    st.error("❌ Aucune donnée disponible pour l'analyse.")

```

def show_about():
st.header("ℹ️ À propos")

```
st.markdown("""
## 📊 BRVM Indices Dashboard

Cette application web permet de visualiser en temps réel les données de la **Bourse Régionale des Valeurs Mobilières (BRVM)**
depuis le site **Sika Finance**.

### 📋 Structure des données:
- **Nom** - Nom de la valeur mobilière
- **Haut** - Plus haut de la séance
- **Bas** - Plus bas de la séance
- **Dernier** - Dernier prix de transaction
- **Volume** - Volume des échanges
- **Variation jour** - Variation en % sur la journée

### 🚀 Fonctionnalités:
- **Données en temps réel** depuis Sika Finance
- **Visualisations interactives** des performances
- **Analyse comparative** des valeurs
- **Métriques de marché** (volatilité, tendance, amplitude)
- **Filtres et recherche** avancés
- **Interface responsive** adaptée à tous les écrans

### 📈 Analyses disponibles:
- Palmarès complet des valeurs BRVM
- Top des hausses et baisses
- Distribution des variations
- Analyse des volumes d'échanges
- Corrélation Prix/Volume/Performance
- Statistiques descriptives du marché

### 🔧 Technologies utilisées:
- **Streamlit** - Framework d'application web Python
- **BeautifulSoup** - Web scraping des données Sika Finance
- **Plotly** - Visualisations interactives avancées
- **Pandas** - Manipulation et analyse de données

### 📝 Source des données:
Les données sont extraites directement depuis:
[<https://www.sikafinance.com/marches/palmares>](<https://www.sikafinance.com/marches/palmares>)

### ⚠️ Disclaimer:
Cette application est développée à des fins **informatives et éducatives** uniquement.

- Les données peuvent avoir un léger délai par rapport au marché réel
- Aucune garantie n'est donnée sur l'exactitude des informations
- Ne constitue pas un conseil en investissement
- Vérifiez toujours les données auprès de sources officielles pour vos décisions d'investissement

### 🏛️ À propos de la BRVM:
La Bourse Régionale des Valeurs Mobilières est la bourse des valeurs de l'Union Économique et Monétaire Ouest Africaine (UEMOA).
Elle regroupe les marchés financiers de 8 pays: Bénin, Burkina Faso, Côte d'Ivoire, Guinée-Bissau, Mali, Niger, Sénégal et Togo.
""")

st.markdown("---")
st.markdown("💡 **Développé avec ❤️ en utilisant Python et Streamlit**")
st.markdown("🔗 **Données fournies par Sika Finance**")

```

if **name** == "**main**":
main()
