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
        
        # Récupération de la page palmares
        url = "https://www.sikafinance.com/marches/palmares"
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
                
                # Recherche des lignes de données
                tbody = table.find('tbody')
                rows = tbody.find_all('tr') if tbody else table.find_all('tr')[1:]  # Skip header row
                
                for row in rows:
                    cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
                    if len(cells) >= 3 and cells[0]:  # Au moins nom, cours, variation
                        all_data.append(cells)
        
        # Si pas de données trouvées dans les tableaux, essayer une approche différente
        if not all_data:
            # Recherche par classes CSS spécifiques à Sika Finance
            stock_rows = soup.find_all(['tr', 'div'], class_=re.compile(r'(stock|action|valeur)', re.I))
            for row in stock_rows:
                text_content = row.get_text(strip=True)
                if text_content and '%' in text_content:
                    # Extraction basique des données si structure non standard
                    parts = re.split(r'\s+', text_content)
                    if len(parts) >= 3:
                        all_data.append(parts[:6])  # Limiter à 6 colonnes max
        
        # Création du DataFrame si des données ont été trouvées
        if all_data:
            # Détermination du nombre de colonnes le plus fréquent
            col_counts = {}
            for row in all_data:
                count = len(row)
                col_counts[count] = col_counts.get(count, 0) + 1
            
            most_common_cols = max(col_counts.keys(), key=lambda x: col_counts[x])
            
            # Filtrage des lignes avec le bon nombre de colonnes
            filtered_data = [row for row in all_data if len(row) == most_common_cols]
            
            if filtered_data:
                # Création des colonnes génériques
                columns = ['Nom', 'Cours', 'Variation', 'Variation_%']
                if most_common_cols > 4:
                    columns.extend([f'Col_{i}' for i in range(5, most_common_cols + 1)])
                else:
                    columns = columns[:most_common_cols]
                
                df = pd.DataFrame(filtered_data, columns=columns)
                
                # Nettoyage des données numériques
                for col in df.columns:
                    if col in ['Cours', 'Variation', 'Variation_%'] or 'Col_' in col:
                        # Nettoyage des valeurs numériques
                        df[col] = df[col].astype(str).str.replace(r'[^\d.,%-]', '', regex=True)
                        df[col] = df[col].str.replace(',', '.')
                        df[col] = df[col].str.replace('%', '')
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df
        
        # Si aucune donnée structurée n'est trouvée, retourner des données de démonstration
        demo_data = {
            'Nom': ['BRVM COMPOSITE', 'BRVM 30', 'BRVM INDUSTRY', 'BRVM FINANCE', 'BRVM AGRICULTURE'],
            'Cours': [245.50, 89.25, 156.78, 198.45, 134.67],
            'Variation': [2.15, -0.45, 1.23, 0.89, -1.12],
            'Variation_%': [0.88, -0.50, 0.79, 0.45, -0.82]
        }
        
        st.warning("⚠️ Données de démonstration affichées. Vérifiez la connexion au site Sika Finance.")
        return pd.DataFrame(demo_data)
        
    except requests.RequestException as e:
        st.error(f"❌ Erreur de connexion à Sika Finance: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Erreur lors du traitement des données: {e}")
        return None

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

def show_overview():
    st.header("📊 Palmarès BRVM - Sika Finance")
    
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
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        
        # Analyse des variations (utilise la colonne Variation ou Variation_%)
        variation_col = None
        if 'Variation_%' in data.columns:
            variation_col = 'Variation_%'
        elif 'Variation' in data.columns:
            variation_col = 'Variation'
        
        if variation_col and data[variation_col].notna().any():
            positive_changes = len(data[data[variation_col] > 0])
            negative_changes = len(data[data[variation_col] < 0])
            avg_change = data[variation_col].mean()
            max_change = data[variation_col].max()
            
            with col1:
                st.metric("📈 Valeurs en hausse", positive_changes)
            with col2:
                st.metric("📉 Valeurs en baisse", negative_changes)
            with col3:
                st.metric("📊 Variation moyenne", f"{avg_change:.2f}%")
            with col4:
                st.metric("🚀 Plus forte hausse", f"{max_change:.2f}%")
        
        # Tableau des données avec filtres
        st.subheader("📋 Données du Palmarès")
        
        # Filtre par type de valeur si applicable
        if 'Nom' in data.columns:
            # Filtre de recherche
            search_term = st.text_input("🔍 Rechercher une valeur:", "").upper()
            if search_term:
                data = data[data['Nom'].str.upper().str.contains(search_term, na=False)]
        
        # Style du tableau
        styled_data = data.copy()
        if variation_col in styled_data.columns:
            def color_variation(val):
                if pd.isna(val):
                    return ''
                elif val > 0:
                    return 'color: green; font-weight: bold'
                elif val < 0:
                    return 'color: red; font-weight: bold'
                else:
                    return ''
            
            # Application du style seulement si la colonne existe
            styled_data = styled_data.style.applymap(color_variation, subset=[variation_col])
        
        st.dataframe(styled_data, use_container_width=True)
        
        # Graphiques des variations
        if variation_col and 'Nom' in data.columns and data[variation_col].notna().any():
            st.subheader("📈 Visualisations des Performances")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Top 10 des hausses
                top_gains = data.nlargest(10, variation_col)
                if not top_gains.empty:
                    fig_gains = px.bar(
                        top_gains, 
                        x='Nom', 
                        y=variation_col,
                        color=variation_col,
                        color_continuous_scale=['red', 'yellow', 'green'],
                        title="Top 10 des plus fortes hausses"
                    )
                    fig_gains.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_gains, use_container_width=True)
            
            with col2:
                # Top 10 des baisses
                top_losses = data.nsmallest(10, variation_col)
                if not top_losses.empty:
                    fig_losses = px.bar(
                        top_losses, 
                        x='Nom', 
                        y=variation_col,
                        color=variation_col,
                        color_continuous_scale=['red', 'yellow', 'green'],
                        title="Top 10 des plus fortes baisses"
                    )
                    fig_losses.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_losses, use_container_width=True)
            
            # Distribution des variations
            if len(data) > 5:
                fig_hist = px.histogram(
                    data, 
                    x=variation_col,
                    nbins=20,
                    title="Distribution des variations",
                    labels={variation_col: "Variation (%)", 'count': 'Nombre de valeurs'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        
        # Secteur d'activité si disponible
        if 'Nom' in data.columns:
            # Tentative d'identification des secteurs par le nom
            sectors = []
            for name in data['Nom']:
                if any(word in name.upper() for word in ['BANK', 'BANQUE', 'FINANCE']):
                    sectors.append('Finance')
                elif any(word in name.upper() for word in ['INDUSTRY', 'INDUSTRIE', 'MANUFACTURE']):
                    sectors.append('Industrie')
                elif any(word in name.upper() for word in ['AGRI', 'PALM', 'RUBBER']):
                    sectors.append('Agriculture')
                elif any(word in name.upper() for word in ['TELECOM', 'TRANSPORT']):
                    sectors.append('Services')
                else:
                    sectors.append('Autres')
            
            data['Secteur'] = sectors
            
            # Graphique par secteur
            if variation_col:
                sector_perf = data.groupby('Secteur')[variation_col].mean().reset_index()
                fig_sector = px.pie(
                    sector_perf, 
                    values=variation_col, 
                    names='Secteur',
                    title="Performance moyenne par secteur"
                )
                st.plotly_chart(fig_sector, use_container_width=True)
        
    else:
        st.error("❌ Impossible de récupérer les données depuis Sika Finance.")
        st.info("💡 Vérifiez votre connexion internet ou réessayez plus tard.")

def show_detailed_analysis():
    st.header("🔍 Analyse Détaillée")
    
    data = get_brvm_data()
    
    if data is not None and not data.empty:
        # Sélection d'une valeur
        if 'Nom' in data.columns:
            selected_stock = st.selectbox("Choisir une valeur:", data['Nom'].tolist())
            
            # Données de la valeur sélectionnée
            stock_data = data[data['Nom'] == selected_stock].iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"📊 {selected_stock}")
                
                # Affichage des métriques disponibles
                for col in data.columns:
                    if col != 'Nom' and col in stock_data and pd.notna(stock_data[col]):
                        value = stock_data[col]
                        if isinstance(value, (int, float)):
                            if 'Variation' in col or '%' in col:
                                st.metric(col, f"{value:.2f}%")
                            else:
                                st.metric(col, f"{value:.2f}")
                        else:
                            st.metric(col, str(value))
            
            with col2:
                # Graphique de comparaison avec le marché
                variation_col = None
                if 'Variation_%' in data.columns:
                    variation_col = 'Variation_%'
                elif 'Variation' in data.columns:
                    variation_col = 'Variation'
                
                if variation_col and data[variation_col].notna().any():
                    # Graphique de positionnement par rapport au marché
                    market_avg = data[variation_col].mean()
                    stock_var = stock_data[variation_col] if pd.notna(stock_data[variation_col]) else 0
                    
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
                        title=f"Comparaison avec le marché"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Analyse du marché global
        st.subheader("📊 Analyse du Marché Global")
        
        variation_col = None
        if 'Variation_%' in data.columns:
            variation_col = 'Variation_%'
        elif 'Variation' in data.columns:
            variation_col = 'Variation'
        
        if variation_col and data[variation_col].notna().any():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "📈 Tendance générale",
                    "Haussière" if data[variation_col].mean() > 0 else "Baissière"
                )
            
            with col2:
                volatility = data[variation_col].std()
                st.metric("📊 Volatilité", f"{volatility:.2f}%")
            
            with col3:
                range_var = data[variation_col].max() - data[variation_col].min()
                st.metric("📏 Amplitude", f"{range_var:.2f}%")
            
            # Box plot des variations
            fig_box = px.box(
                data,
                y=variation_col,
                title="Distribution des variations du marché"
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Tableau de corrélation si plusieurs colonnes numériques
        numeric_columns = data.select_dtypes(include=[float, int]).columns
        if len(numeric_columns) > 1:
            st.subheader("🔗 Analyse de Corrélation")
            correlation_matrix = data[numeric_columns].corr()
            
            fig_corr = px.imshow(
                correlation_matrix,
                text_auto=True,
                aspect="auto",
                title="Matrice de corrélation"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Statistiques descriptives
        st.subheader("📈 Statistiques Descriptives")
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
