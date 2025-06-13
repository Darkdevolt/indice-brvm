import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

def get_brvm_index():
    """
    Récupère les données des indices BRVM depuis le site officiel
    Adaptation Python du code R fourni
    """
    try:
        # Récupération de la page web
        url = "https://www.brvm.org/en/indices/status/200"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse du HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        tables = soup.find_all('table')
        
        if len(tables) >= 4:
            # Récupération du 4ème tableau (index 3)
            table = tables[3]
            
            # Extraction des données du tableau
            headers = []
            rows = []
            
            # Récupération des en-têtes
            header_row = table.find('thead')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all('th')]
            
            # Récupération des données
            tbody = table.find('tbody')
            if tbody:
                for row in tbody.find_all('tr'):
                    cells = [td.get_text(strip=True) for td in row.find_all('td')]
                    if cells:  # Éviter les lignes vides
                        rows.append(cells)
            
            # Création du DataFrame
            if rows and headers:
                df = pd.DataFrame(rows, columns=headers[:len(rows[0])])
                
                # Nettoyage des données selon le code R original
                if 'Previous closing' in df.columns:
                    df['Previous closing'] = df['Previous closing'].str.replace(' ', '').str.replace(',', '.')
                    df['Previous closing'] = pd.to_numeric(df['Previous closing'], errors='coerce')
                
                if 'Closing' in df.columns:
                    df['Closing'] = df['Closing'].str.replace(' ', '').str.replace(',', '.')
                    df['Closing'] = pd.to_numeric(df['Closing'], errors='coerce')
                
                if 'Change (%)' in df.columns:
                    df['Change (%)'] = df['Change (%)'].str.replace(',', '.')
                    df['Change (%)'] = pd.to_numeric(df['Change (%)'], errors='coerce')
                
                if 'Year to Date Change' in df.columns:
                    df['Year to Date Change'] = df['Year to Date Change'].str.replace(',', '.')
                    df['Year to Date Change'] = pd.to_numeric(df['Year to Date Change'], errors='coerce')
                
                # Renommage des colonnes selon le code R
                new_columns = {
                    'Indexes': 'Indexes',
                    'Previous closing': 'Previous closing',
                    'Closing': 'Closing',
                    'Change (%)': 'Change (%)',
                    'Year to Date Change': 'Year to Date Change'
                }
                
                # Garder seulement les colonnes nécessaires
                available_columns = [col for col in new_columns.keys() if col in df.columns]
                df = df[available_columns]
                
                # Renommage des secteurs selon le code R
                if 'Indexes' in df.columns:
                    df['Indexes'] = df['Indexes'].str.replace('BRVM - INDUSTRIE', 'BRVM - INDUSTRY')
                    df['Indexes'] = df['Indexes'].str.replace('BRVM - AUTRES SECTEURS', 'BRVM - OTHER SECTOR')
                    df['Indexes'] = df['Indexes'].str.replace('SERVICES PUBLICS', 'PUBLIC SERVICES')
                
                return df
            
        return None
        
    except requests.RequestException as e:
        st.error(f"Erreur de connexion: {e}")
        return None
    except Exception as e:
        st.error(f"Erreur lors du traitement des données: {e}")
        return None

def main():
    st.set_page_config(
        page_title="BRVM Indices Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📈 Tableau de Bord des Indices BRVM")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox(
        "Choisir une page",
        ["Aperçu des Indices", "Analyse Détaillée", "À propos"]
    )
    
    if page == "Aperçu des Indices":
        show_overview()
    elif page == "Analyse Détaillée":
        show_detailed_analysis()
    else:
        show_about()

def show_overview():
    st.header("📊 Aperçu des Indices BRVM")
    
    # Bouton de rafraîchissement
    if st.button("🔄 Actualiser les données"):
        st.cache_data.clear()
    
    # Chargement des données
    with st.spinner("Chargement des données BRVM..."):
        data = get_brvm_index()
    
    if data is not None and not data.empty:
        st.success(f"✅ Données récupérées avec succès! ({len(data)} indices)")
        
        # Affichage de la date de mise à jour
        st.info(f"🕒 Dernière mise à jour: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        if 'Change (%)' in data.columns:
            positive_changes = len(data[data['Change (%)'] > 0])
            negative_changes = len(data[data['Change (%)'] < 0])
            avg_change = data['Change (%)'].mean()
            max_change = data['Change (%)'].max()
            
            with col1:
                st.metric("📈 Indices en hausse", positive_changes)
            with col2:
                st.metric("📉 Indices en baisse", negative_changes)
            with col3:
                st.metric("📊 Variation moyenne", f"{avg_change:.2f}%")
            with col4:
                st.metric("🚀 Plus forte hausse", f"{max_change:.2f}%")
        
        # Tableau des données
        st.subheader("📋 Données des Indices")
        
        # Style du tableau
        styled_data = data.copy()
        if 'Change (%)' in styled_data.columns:
            def color_change(val):
                if pd.isna(val):
                    return ''
                elif val > 0:
                    return 'color: green'
                elif val < 0:
                    return 'color: red'
                else:
                    return ''
            
            styled_data = styled_data.style.applymap(color_change, subset=['Change (%)'])
        
        st.dataframe(styled_data, use_container_width=True)
        
        # Graphique des variations
        if 'Change (%)' in data.columns and 'Indexes' in data.columns:
            st.subheader("📈 Variations des Indices")
            
            fig = px.bar(
                data, 
                x='Indexes', 
                y='Change (%)',
                color='Change (%)',
                color_continuous_scale=['red', 'white', 'green'],
                title="Variations journalières des indices BRVM"
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("❌ Impossible de récupérer les données. Vérifiez votre connexion internet.")

def show_detailed_analysis():
    st.header("🔍 Analyse Détaillée")
    
    data = get_brvm_index()
    
    if data is not None and not data.empty:
        # Sélection d'un indice
        if 'Indexes' in data.columns:
            selected_index = st.selectbox("Choisir un indice:", data['Indexes'].tolist())
            
            # Données de l'indice sélectionné
            index_data = data[data['Indexes'] == selected_index].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"📊 {selected_index}")
                if 'Closing' in index_data:
                    st.metric("Clôture", f"{index_data['Closing']:.2f}")
                if 'Previous closing' in index_data:
                    st.metric("Clôture précédente", f"{index_data['Previous closing']:.2f}")
                if 'Change (%)' in index_data:
                    st.metric("Variation (%)", f"{index_data['Change (%)']:.2f}%")
                if 'Year to Date Change' in index_data:
                    st.metric("Variation YTD", f"{index_data['Year to Date Change']:.2f}%")
            
            with col2:
                # Graphique circulaire des variations
                if 'Change (%)' in data.columns:
                    positive_count = len(data[data['Change (%)'] > 0])
                    negative_count = len(data[data['Change (%)'] < 0])
                    neutral_count = len(data[data['Change (%)'] == 0])
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=['Positives', 'Négatives', 'Neutres'],
                        values=[positive_count, negative_count, neutral_count],
                        hole=.3
                    )])
                    fig.update_traces(hoverinfo='label+percent', textinfo='value+percent')
                    fig.update_layout(title="Répartition des variations")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Tableau de corrélation (si pertinent)
        st.subheader("📈 Statistiques Générales")
        numeric_columns = data.select_dtypes(include=[float, int]).columns
        if len(numeric_columns) > 0:
            st.dataframe(data[numeric_columns].describe(), use_container_width=True)
    
    else:
        st.error("❌ Aucune donnée disponible pour l'analyse.")

def show_about():
    st.header("ℹ️ À propos")
    
    st.markdown("""
    ## 📊 BRVM Indices Dashboard
    
    Cette application web permet de visualiser en temps réel les indices de la **Bourse Régionale des Valeurs Mobilières (BRVM)**.
    
    ### 🚀 Fonctionnalités:
    - **Données en temps réel** des indices BRVM
    - **Visualisations interactives** avec Plotly
    - **Métriques clés** et indicateurs de performance
    - **Interface responsive** adaptée à tous les écrans
    
    ### 📈 Indices suivis:
    - BRVM Composite
    - BRVM 30
    - BRVM Industry
    - BRVM Other Sector
    - Public Services
    - Et plus encore...
    
    ### 🔧 Technologies utilisées:
    - **Streamlit** - Framework d'application web
    - **Python** - Langage de programmation
    - **BeautifulSoup** - Web scraping
    - **Plotly** - Visualisations interactives
    - **Pandas** - Manipulation de données
    
    ### 📝 Source des données:
    Les données sont récupérées directement depuis le site officiel de la BRVM: 
    [https://www.brvm.org/](https://www.brvm.org/)
    
    ### ⚠️ Disclaimer:
    Cette application est à des fins informatives uniquement. 
    Les données peuvent avoir un léger délai et ne doivent pas être utilisées 
    pour des décisions d'investissement sans vérification supplémentaire.
    """)
    
    st.markdown("---")
    st.markdown("💡 **Développé avec ❤️ en utilisant Python et Streamlit**")

if __name__ == "__main__":
    main()
