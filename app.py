# -*- coding: utf-8 -*-
"""
Application Streamlit pour Tester l'Ajustement de Distributions Statistiques

Version améliorée pour gérer les données de prix historiques (ex: financiers).
Cette application permet aux utilisateurs de téléverser un fichier de données (CSV)
et de tester l'ajustement de plusieurs distributions de probabilité sur les prix bruts
ou sur les rendements journaliers.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import warnings

# --- Configuration de la Page ---
st.set_page_config(
    page_title="Testeur d'Ajustement de Distributions",
    page_icon="📊",
    layout="wide",
)

# --- Fonctions Utilitaires ---

def fit_and_test_distribution(dist_name, data):
    """
    Ajuste une distribution spécifiée aux données et effectue un test K-S.

    Args:
        dist_name (str): Le nom de la distribution de scipy.stats.
        data (pd.Series): La série de données à analyser.

    Returns:
        tuple: Une tuple contenant (paramètres, stat K-S, p-value K-S).
               Retourne (None, None, None) si l'ajustement échoue.
    """
    try:
        dist = getattr(stats, dist_name)
        
        # Gérer les lois qui ne peuvent pas prendre de valeurs négatives (communes dans les rendements)
        if dist_name in ['lognorm', 'expon', 'weibull_min', 'gamma', 'pareto']:
            if data.min() <= 0:
                return None, None, None # On ne peut pas ajuster ces lois si des valeurs sont négatives ou nulles.
        
        # Pour la loi Bêta, les données doivent être mises à l'échelle entre 0 et 1
        if dist_name == 'beta':
            if data.min() < 0 or data.max() > 1:
                data_scaled = (data - data.min()) / (data.max() - data.min())
                params = dist.fit(data_scaled, floc=0, fscale=1) # floc/fscale pour fixer l'intervalle à [0,1]
            else:
                params = dist.fit(data, floc=0, fscale=1)
        else:
            params = dist.fit(data)

        # Effectuer le test de Kolmogorov-Smirnov
        D, p_value = stats.kstest(data, dist_name, args=params)

        return params, D, p_value
    except Exception:
        # Gérer silencieusement les erreurs d'ajustement
        return None, None, None

def plot_distribution_fit(data, dist_name, params, analysis_type):
    """
    Génère un graphique comparant l'histogramme des données avec la PDF de la distribution ajustée.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(data, bins=50, density=True, alpha=0.7, label=f'Histogramme ({analysis_type})')

    dist = getattr(stats, dist_name)
    x = np.linspace(data.min(), data.max(), 1000)
    
    # Gestion spéciale pour la loi Bêta si les données ont été mises à l'échelle
    if dist_name == 'beta' and (data.min() < 0 or data.max() > 1):
        x_scaled = (x - data.min()) / (data.max() - data.min())
        pdf = dist.pdf(x_scaled, *params)
        # Il faut ajuster la densité à la nouvelle échelle
        pdf = pdf / (data.max() - data.min())
        ax.plot(x, pdf, 'r-', lw=2, label=f'PDF de la Loi {dist_name.capitalize()}')
    else:
        pdf = dist.pdf(x, *params)
        ax.plot(x, pdf, 'r-', lw=2, label=f'PDF de la Loi {dist_name.capitalize()}')


    ax.set_title(f'Ajustement de la Loi {dist_name.capitalize()} sur les {analysis_type}', fontsize=16)
    ax.set_xlabel('Valeur')
    ax.set_ylabel('Densité')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

# --- Interface Utilisateur Streamlit ---

st.title("📊 Testeur d'Ajustement de Lois Statistiques")
st.markdown("""
Cette application vous aide à déterminer quelle loi statistique correspond le mieux à vos données.
Spécialement adaptée pour les **données financières historiques**.
""")

# --- Barre Latérale pour les Entrées ---
with st.sidebar:
    st.header("1. Paramètres des Données")
    uploaded_file = st.file_uploader(
        "Téléversez votre fichier CSV (ex: HistoricalPrices.csv)",
        type="csv",
        help="Le fichier doit contenir une colonne 'Date' et des colonnes numériques (Open, High, Low, Close...)."
    )

    analysis_type = "Prix Bruts" # Valeur par défaut

    if uploaded_file is not None:
        try:
            # Essayer de parser la colonne 'Date' directement
            df = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
            df.sort_index(inplace=True)
            st.success("Fichier téléversé et colonne 'Date' reconnue !")

            numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_columns:
                st.error("Aucune colonne numérique trouvée dans le fichier CSV.")
                st.stop()

            selected_column = st.selectbox(
                "Choisissez la colonne de données à analyser",
                options=numeric_columns,
                index=numeric_columns.index('Close') if 'Close' in numeric_columns else 0
            )
            
            st.header("2. Type d'Analyse")
            analysis_type = st.radio(
                "Que souhaitez-vous analyser ?",
                ('Prix Bruts', 'Rendements Journaliers'),
                help="Les rendements sont souvent plus pertinents pour l'analyse statistique en finance."
            )

        except Exception as e:
            st.error(f"Erreur à la lecture du fichier : {e}. Assurez-vous qu'il y a bien une colonne 'Date'.")
            st.stop()

    st.header("3. Lois à Tester")
    available_distributions = {
        "Normale": "norm",
        "Log-Normale": "lognorm",
        "Exponentielle": "expon",
        "Weibull Min": "weibull_min",
        "Gamma": "gamma",
        "Bêta": "beta",
        "Uniforme": "uniform",
        "Pareto": "pareto"
    }
    
    selected_distributions_names = st.multiselect(
        "Sélectionnez les lois à tester",
        options=list(available_distributions.keys()),
        default=list(available_distributions.keys())
    )
    
    distributions_to_test = {name: code for name, code in available_distributions.items() if name in selected_distributions_names}


# --- Zone Principale pour les Résultats ---
if 'uploaded_file' in locals() and uploaded_file is not None and 'selected_column' in locals():
    
    data_raw = df[selected_column].dropna()

    if analysis_type == 'Rendements Journaliers':
        data_to_analyze = data_raw.pct_change().dropna()
        st.header(f"Analyse des Rendements Journaliers de : `{selected_column}`")
    else:
        data_to_analyze = data_raw
        st.header(f"Analyse des Prix Bruts de : `{selected_column}`")

    if data_to_analyze.empty or len(data_to_analyze) < 10:
        st.warning(f"La série de données ({analysis_type}) contient moins de 10 points. Analyse impossible.")
        st.stop()
        
    with st.expander("Aperçu des données et statistiques descriptives"):
        st.dataframe(data_to_analyze.head())
        st.write(data_to_analyze.describe())

    st.header("Résultats de l'Ajustement des Distributions")

    results = []
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        for pretty_name, dist_code in distributions_to_test.items():
            
            with st.spinner(f"Ajustement en cours pour la loi {pretty_name}..."):
                params, D_stat, p_value = fit_and_test_distribution(dist_code, data_to_analyze)
            
            if params is not None:
                st.subheader(f"Test de la Loi {pretty_name}")
                results.append({
                    "Loi": pretty_name,
                    "Statistique K-S (D)": D_stat,
                    "P-value": p_value,
                    "Paramètres ajustés": params
                })
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric(label="Statistique K-S (D)", value=f"{D_stat:.4f}")
                    st.metric(label="P-value du test K-S", value=f"{p_value:.4f}")
                    if p_value < 0.05:
                        st.warning("""**Interprétation :** La p-value est faible (< 0.05). L'hypothèse que les données suivent cette loi est rejetée.""")
                    else:
                        st.success("""**Interprétation :** La p-value est élevée (>= 0.05). L'ajustement est plausible.""")
                
                with col2:
                    plot_distribution_fit(data_to_analyze, dist_code, params, analysis_type)
            else:
                 st.subheader(f"Test de la Loi {pretty_name}")
                 st.info(f"Impossible d'ajuster la loi {pretty_name}. Les données (ex: négatives) ne sont peut-être pas compatibles avec cette loi.")
            
            st.divider()

    if results:
        st.header("Tableau de Synthèse")
        st.info("""
        **Comment lire ce tableau ?**
        - **P-value :** Une p-value élevée (proche de 1) suggère un bon ajustement. C'est souvent le critère le plus important.
        - **Statistique K-S (D) :** Une valeur faible (proche de 0) suggère un bon ajustement.
        **La meilleure loi est généralement celle avec la p-value la plus élevée et la statistique K-S la plus faible.**
        """)
        
        results_df = pd.DataFrame(results).sort_values(by=["P-value", "Statistique K-S (D)"], ascending=[False, True]).set_index("Loi")
        
        st.dataframe(
            results_df.style.background_gradient(cmap='Greens', subset=['P-value'])
                             .background_gradient(cmap='Reds_r', subset=['Statistique K-S (D)'])
        )
    else:
        st.warning("Aucun résultat à afficher. Cela peut arriver si aucune des lois sélectionnées n'est compatible avec vos données (par exemple, des données avec des valeurs négatives pour des lois qui ne les supportent pas).")

else:
    st.info("Veuillez téléverser un fichier CSV et sélectionner une colonne pour commencer l'analyse.")
