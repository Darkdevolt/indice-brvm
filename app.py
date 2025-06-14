# -*- coding: utf-8 -*-
"""
Application Streamlit pour Tester l'Ajustement de Distributions Statistiques

Cette application permet aux utilisateurs de téléverser un fichier de données (CSV)
et de tester l'ajustement de plusieurs distributions de probabilité à leurs données.
Elle calcule les paramètres pour chaque distribution, effectue un test de Kolmogorov-Smirnov (K-S)
pour évaluer la qualité de l'ajustement, et visualise l'histogramme des données par rapport
à la fonction de densité de probabilité (PDF) de chaque distribution ajustée.
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
        # Récupérer l'objet de distribution de scipy.stats
        dist = getattr(stats, dist_name)

        # Ajuster la distribution aux données pour obtenir les paramètres
        # Pour la loi Bêta, les données doivent être mises à l'échelle entre 0 et 1
        if dist_name == 'beta':
            if data.min() < 0 or data.max() > 1:
                # Mise à l'échelle Min-Max simple
                data_scaled = (data - data.min()) / (data.max() - data.min())
                params = dist.fit(data_scaled)
            else:
                params = dist.fit(data)
        else:
            params = dist.fit(data)

        # Effectuer le test de Kolmogorov-Smirnov
        D, p_value = stats.kstest(data, dist_name, args=params)

        return params, D, p_value
    except Exception as e:
        # Gérer les erreurs potentielles lors de l'ajustement (par exemple, données non valides pour une loi)
        st.warning(f"Impossible d'ajuster la loi {dist_name}. Erreur : {e}")
        return None, None, None

def plot_distribution_fit(data, dist_name, params):
    """
    Génère un graphique comparant l'histogramme des données avec la PDF de la distribution ajustée.

    Args:
        data (pd.Series): La série de données.
        dist_name (str): Le nom de la distribution.
        params (tuple): Les paramètres de la distribution ajustée.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Tracer l'histogramme des données (densité)
    ax.hist(data, bins=30, density=True, alpha=0.6, label='Histogramme des Données')

    # Générer la PDF de la distribution ajustée
    dist = getattr(stats, dist_name)
    x = np.linspace(data.min(), data.max(), 1000)

    # Si la loi est Bêta, les données d'origine ont peut-être été mises à l'échelle
    if dist_name == 'beta' and (data.min() < 0 or data.max() > 1):
        # La PDF est calculée sur [0, 1] puis l'axe des x est remis à l'échelle
        x_scaled = np.linspace(0, 1, 1000)
        pdf = dist.pdf(x_scaled, *params)
        ax.plot(x, pdf, 'r-', lw=2, label=f'PDF de la Loi {dist_name.capitalize()}')
    else:
        pdf = dist.pdf(x, *params)
        ax.plot(x, pdf, 'r-', lw=2, label=f'PDF de la Loi {dist_name.capitalize()}')

    # Mise en forme du graphique
    ax.set_title(f'Ajustement de la Loi {dist_name.capitalize()}', fontsize=16)
    ax.set_xlabel('Valeur')
    ax.set_ylabel('Densité')
    ax.legend()
    st.pyplot(fig)
    plt.close(fig) # Fermer la figure pour libérer la mémoire

# --- Interface Utilisateur Streamlit ---

st.title("📊 Testeur d'Ajustement de Lois Statistiques")
st.markdown("""
Cette application vous aide à déterminer quelle loi statistique correspond le mieux à votre série de données historiques.
Téléversez un fichier CSV, sélectionnez la colonne à analyser, et l'application testera 8 lois différentes.
""")

# --- Barre Latérale pour les Entrées ---
with st.sidebar:
    st.header("1. Paramètres des Données")
    uploaded_file = st.file_uploader(
        "Téléversez votre fichier CSV",
        type="csv",
        help="Le fichier doit contenir au moins une colonne de données numériques."
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Fichier téléversé avec succès !")

            numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_columns:
                st.error("Aucune colonne numérique trouvée dans le fichier CSV.")
                st.stop()

            selected_column = st.selectbox(
                "Choisissez la colonne de données à analyser",
                options=numeric_columns
            )
        except Exception as e:
            st.error(f"Erreur à la lecture du fichier : {e}")
            st.stop()

    st.header("2. Lois à Tester")
    # Liste des distributions à tester
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
    st.header(f"Analyse de la colonne : `{selected_column}`")
    data = df[selected_column].dropna()

    if data.empty or len(data) < 5:
        st.warning("La colonne sélectionnée contient moins de 5 points de données après suppression des valeurs manquantes. Analyse impossible.")
        st.stop()
        
    # Afficher un aperçu des données
    with st.expander("Aperçu des données et statistiques descriptives"):
        st.dataframe(data.head())
        st.write(data.describe())

    st.header("Résultats de l'Ajustement des Distributions")

    results = []
    # Ignorer les avertissements de runtime de SciPy qui peuvent survenir lors de l'ajustement
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        for pretty_name, dist_code in distributions_to_test.items():
            st.subheader(f"Test de la Loi {pretty_name}")
            
            with st.spinner(f"Ajustement en cours pour la loi {pretty_name}..."):
                params, D_stat, p_value = fit_and_test_distribution(dist_code, data)
            
            if params is not None:
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
                    st.info(f"**Paramètres :** {[f'{p:.2f}' for p in params]}")
                    if p_value < 0.05:
                        st.warning("""**Interprétation :** La p-value est faible (< 0.05). L'hypothèse que les données suivent cette loi est rejetée. L'ajustement n'est pas bon.""")
                    else:
                        st.success("""**Interprétation :** La p-value est élevée (>= 0.05). On ne peut pas rejeter l'hypothèse que les données suivent cette loi. L'ajustement est plausible.""")
                
                with col2:
                    plot_distribution_fit(data, dist_code, params)
            
            st.divider()

    # --- Synthèse des Résultats ---
    if results:
        st.header("Tableau de Synthèse")
        st.info("""
        **Comment lire ce tableau ?**
        - **P-value :** Une p-value élevée (proche de 1) suggère un bon ajustement. C'est souvent le critère le plus important.
        - **Statistique K-S (D) :** Une valeur faible (proche de 0) suggère un bon ajustement. Elle mesure la distance maximale entre la distribution de vos données et la distribution testée.
        
        **La meilleure loi est généralement celle avec la p-value la plus élevée et la statistique K-S la plus faible.**
        """)
        
        results_df = pd.DataFrame(results)
        # Trier par p-value (décroissant) puis par statistique K-S (croissant)
        results_df = results_df.sort_values(by=["P-value", "Statistique K-S (D)"], ascending=[False, True])
        results_df = results_df.set_index("Loi")
        
        st.dataframe(
            results_df.style.background_gradient(cmap='Greens', subset=['P-value'])
                             .background_gradient(cmap='Reds_r', subset=['Statistique K-S (D)'])
        )
    else:
        st.warning("Aucun résultat à afficher. Vérifiez vos données ou les lois sélectionnées.")

else:
    st.info("Veuillez téléverser un fichier CSV et sélectionner une colonne pour commencer l'analyse.")
