import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import t, jarque_bera, shapiro, kstest
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Analyse Distribution Student-t et Rendements",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Analyse de la Distribution Student-t et Prévision des Rendements")
st.markdown("---")

# Fonction pour charger et préparer les données
@st.cache_data
def load_and_prepare_data(uploaded_file=None):
    """Charge et prépare les données de prix"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Données par défaut (extrait de votre CSV)
        data = """Date, Open, High, Low, Close, Volume
06/13/25, 25000, 24900, 24900, 24900, 21005
06/12/25, 25000, 25000, 25000, 25000, 19553
06/11/25, 25000, 25000, 25000, 25000, 35662
06/10/25, 25000, 25000, 25000, 25000, 19045
06/05/25, 25000, 25000, 25000, 25000, 5889
06/04/25, 25000, 25000, 25000, 25000, 73169"""
        
        from io import StringIO
        df = pd.read_csv(StringIO(data))
    
    # Nettoyer les noms de colonnes
    df.columns = df.columns.str.strip()
    
    # Convertir la date
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
    
    # Trier par date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Calculer les rendements
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Supprimer les valeurs NaN
    df = df.dropna()
    
    return df

# Fonction pour tester la distribution Student-t
def test_student_t_distribution(returns):
    """Teste si les rendements suivent une distribution Student-t"""
    
    # Éliminer les outliers extrêmes
    q1, q3 = np.percentile(returns, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    clean_returns = returns[(returns >= lower_bound) & (returns <= upper_bound)]
    
    # Ajuster une distribution Student-t
    params_t = stats.t.fit(clean_returns)
    df_t, loc_t, scale_t = params_t
    
    # Ajuster une distribution normale pour comparaison
    params_norm = stats.norm.fit(clean_returns)
    loc_norm, scale_norm = params_norm
    
    # Tests statistiques
    # Test de Kolmogorov-Smirnov pour Student-t
    ks_stat_t, ks_p_t = stats.kstest(clean_returns, lambda x: stats.t.cdf(x, df_t, loc_t, scale_t))
    
    # Test de Kolmogorov-Smirnov pour distribution normale
    ks_stat_norm, ks_p_norm = stats.kstest(clean_returns, lambda x: stats.norm.cdf(x, loc_norm, scale_norm))
    
    # Test de Jarque-Bera (normalité)
    jb_stat, jb_p = jarque_bera(clean_returns)
    
    # Calcul de l'AIC pour comparaison
    log_likelihood_t = np.sum(stats.t.logpdf(clean_returns, df_t, loc_t, scale_t))
    log_likelihood_norm = np.sum(stats.norm.logpdf(clean_returns, loc_norm, scale_norm))
    
    aic_t = -2 * log_likelihood_t + 2 * 3  # 3 paramètres pour Student-t
    aic_norm = -2 * log_likelihood_norm + 2 * 2  # 2 paramètres pour normale
    
    return {
        'params_t': params_t,
        'params_norm': params_norm,
        'ks_test_t': (ks_stat_t, ks_p_t),
        'ks_test_norm': (ks_stat_norm, ks_p_norm),
        'jarque_bera': (jb_stat, jb_p),
        'aic_t': aic_t,
        'aic_norm': aic_norm,
        'clean_returns': clean_returns
    }

# Fonction pour calculer les niveaux de rendement
def calculate_return_levels(returns, distribution_params, dist_type='t'):
    """Calcule les niveaux de rendement attendus"""
    
    if dist_type == 't':
        df_param, loc, scale = distribution_params
        distribution = stats.t(df_param, loc, scale)
    else:
        loc, scale = distribution_params
        distribution = stats.norm(loc, scale)
    
    # Niveaux de confiance
    confidence_levels = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    
    quantiles = {}
    for conf in confidence_levels:
        quantiles[f'{conf*100:.0f}%'] = distribution.ppf(conf)
    
    # VaR et CVaR
    var_95 = distribution.ppf(0.05)
    var_99 = distribution.ppf(0.01)
    
    # CVaR (Expected Shortfall)
    x_range = np.linspace(distribution.ppf(0.001), var_95, 1000)
    cvar_95 = np.trapz(x_range * distribution.pdf(x_range), x_range) / distribution.cdf(var_95)
    
    x_range_99 = np.linspace(distribution.ppf(0.001), var_99, 1000)
    cvar_99 = np.trapz(x_range_99 * distribution.pdf(x_range_99), x_range_99) / distribution.cdf(var_99)
    
    return {
        'quantiles': quantiles,
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99,
        'expected_return': distribution.mean(),
        'volatility': distribution.std()
    }

# Interface utilisateur
st.sidebar.header("📁 Chargement des données")
uploaded_file = st.sidebar.file_uploader(
    "Charger un fichier CSV", 
    type=['csv'],
    help="Format attendu: Date, Open, High, Low, Close, Volume"
)

# Charger les données
df = load_and_prepare_data(uploaded_file)

# Affichage des informations de base
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Nombre d'observations", len(df))
with col2:
    st.metric("Période", f"{df['Date'].min().strftime('%m/%Y')} - {df['Date'].max().strftime('%m/%Y')}")
with col3:
    st.metric("Prix moyen", f"{df['Close'].mean():.2f}")
with col4:
    st.metric("Volatilité (std)", f"{df['Returns'].std():.4f}")

# Analyse de la distribution
st.header("🔍 Test de la Distribution Student-t")

returns = df['Log_Returns'].dropna()
test_results = test_student_t_distribution(returns)

# Résultats des tests
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Tests Statistiques")
    
    # Test de Jarque-Bera
    jb_stat, jb_p = test_results['jarque_bera']
    if jb_p < 0.05:
        st.error(f"❌ Test Jarque-Bera: p-value = {jb_p:.4f} (Rejet de la normalité)")
    else:
        st.success(f"✅ Test Jarque-Bera: p-value = {jb_p:.4f} (Non-rejet de la normalité)")
    
    # Test KS pour Student-t
    ks_stat_t, ks_p_t = test_results['ks_test_t']
    if ks_p_t > 0.05:
        st.success(f"✅ KS Test (Student-t): p-value = {ks_p_t:.4f} (Bonne adéquation)")
    else:
        st.warning(f"⚠️ KS Test (Student-t): p-value = {ks_p_t:.4f} (Adéquation questionnable)")
    
    # Test KS pour Normale
    ks_stat_norm, ks_p_norm = test_results['ks_test_norm']
    if ks_p_norm > 0.05:
        st.success(f"✅ KS Test (Normale): p-value = {ks_p_norm:.4f} (Bonne adéquation)")
    else:
        st.warning(f"⚠️ KS Test (Normale): p-value = {ks_p_norm:.4f} (Adéquation questionnable)")

with col2:
    st.subheader("📈 Comparaison des Modèles")
    
    aic_t = test_results['aic_t']
    aic_norm = test_results['aic_norm']
    
    st.write(f"**AIC Student-t:** {aic_t:.2f}")
    st.write(f"**AIC Normale:** {aic_norm:.2f}")
    
    if aic_t < aic_norm:
        st.success("🏆 **Student-t est le meilleur modèle** (AIC plus faible)")
        best_model = 't'
    else:
        st.info("🏆 **Distribution Normale est le meilleur modèle** (AIC plus faible)")
        best_model = 'norm'
    
    # Paramètres de la distribution Student-t
    df_param, loc_t, scale_t = test_results['params_t']
    st.write(f"**Paramètres Student-t:**")
    st.write(f"- Degrés de liberté: {df_param:.2f}")
    st.write(f"- Location: {loc_t:.4f}")
    st.write(f"- Scale: {scale_t:.4f}")

# Graphiques de distribution
st.header("📊 Visualisation des Distributions")

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Distribution des Rendements', 'Q-Q Plot Student-t', 
                   'Q-Q Plot Normal', 'Densité des Distributions'),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

clean_returns = test_results['clean_returns']

# Histogramme
fig.add_trace(
    go.Histogram(x=clean_returns, nbinsx=50, name='Données', opacity=0.7),
    row=1, col=1
)

# Q-Q Plot Student-t
df_param, loc_t, scale_t = test_results['params_t']
theoretical_quantiles_t = stats.t.ppf(np.linspace(0.01, 0.99, len(clean_returns)), df_param, loc_t, scale_t)
sample_quantiles = np.sort(clean_returns)

fig.add_trace(
    go.Scatter(x=theoretical_quantiles_t, y=sample_quantiles, mode='markers', name='Student-t Q-Q'),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(x=[min(theoretical_quantiles_t), max(theoretical_quantiles_t)], 
              y=[min(theoretical_quantiles_t), max(theoretical_quantiles_t)], 
              mode='lines', name='Ligne théorique', line=dict(color='red')),
    row=1, col=2
)

# Q-Q Plot Normal
loc_norm, scale_norm = test_results['params_norm']
theoretical_quantiles_norm = stats.norm.ppf(np.linspace(0.01, 0.99, len(clean_returns)), loc_norm, scale_norm)

fig.add_trace(
    go.Scatter(x=theoretical_quantiles_norm, y=sample_quantiles, mode='markers', name='Normal Q-Q'),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=[min(theoretical_quantiles_norm), max(theoretical_quantiles_norm)], 
              y=[min(theoretical_quantiles_norm), max(theoretical_quantiles_norm)], 
              mode='lines', name='Ligne théorique', line=dict(color='red')),
    row=2, col=1
)

# Comparaison des densités
x_range = np.linspace(clean_returns.min(), clean_returns.max(), 1000)
pdf_t = stats.t.pdf(x_range, df_param, loc_t, scale_t)
pdf_norm = stats.norm.pdf(x_range, loc_norm, scale_norm)

fig.add_trace(
    go.Scatter(x=x_range, y=pdf_t, mode='lines', name='Student-t PDF'),
    row=2, col=2
)
fig.add_trace(
    go.Scatter(x=x_range, y=pdf_norm, mode='lines', name='Normal PDF'),
    row=2, col=2
)

fig.update_layout(height=800, showlegend=True, title_text="Analyse Comparative des Distributions")
st.plotly_chart(fig, use_container_width=True)

# Calcul des niveaux de rendement
st.header("💰 Niveaux de Rendement Attendus")

if best_model == 't':
    return_levels = calculate_return_levels(returns, test_results['params_t'], 't')
    st.success("🎯 Calculs basés sur la distribution **Student-t** (meilleur modèle)")
else:
    return_levels = calculate_return_levels(returns, test_results['params_norm'], 'norm')
    st.info("🎯 Calculs basés sur la distribution **Normale** (meilleur modèle)")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📊 Statistiques Principales")
    st.metric("Rendement Espéré", f"{return_levels['expected_return']:.4f}")
    st.metric("Volatilité", f"{return_levels['volatility']:.4f}")

with col2:
    st.subheader("⚠️ Value at Risk (VaR)")
    st.metric("VaR 95%", f"{return_levels['var_95']:.4f}")
    st.metric("VaR 99%", f"{return_levels['var_99']:.4f}")

with col3:
    st.subheader("💥 Conditional VaR (CVaR)")
    st.metric("CVaR 95%", f"{return_levels['cvar_95']:.4f}")
    st.metric("CVaR 99%", f"{return_levels['cvar_99']:.4f}")

# Tableau des quantiles
st.subheader("📋 Quantiles de Rendement")
quantiles_df = pd.DataFrame(list(return_levels['quantiles'].items()), 
                           columns=['Niveau de Confiance', 'Rendement'])
quantiles_df['Rendement (%)'] = quantiles_df['Rendement'] * 100

st.dataframe(quantiles_df, use_container_width=True)

# Interprétation et recommandations
st.header("🎯 Interprétation et Recommandations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Résumé de l'Analyse")
    
    if best_model == 't':
        st.write("""
        **✅ La distribution Student-t est appropriée pour vos données**
        
        - Les rendements présentent des queues plus épaisses qu'une distribution normale
        - Le modèle Student-t capture mieux les événements extrêmes
        - Les tests statistiques confirment une meilleure adéquation
        """)
    else:
        st.write("""
        **ℹ️ La distribution Normale semble suffisante**
        
        - Les rendements suivent approximativement une distribution normale
        - Pas de queues particulièrement épaisses détectées
        - Le modèle normal est plus simple et approprié
        """)

with col2:
    st.subheader("⚡ Recommandations")
    
    if return_levels['expected_return'] > 0:
        st.success(f"📈 **Tendance positive**: Rendement espéré de {return_levels['expected_return']*100:.2f}%")
    else:
        st.warning(f"📉 **Tendance négative**: Rendement espéré de {return_levels['expected_return']*100:.2f}%")
    
    volatility_level = return_levels['volatility']
    if volatility_level > 0.02:
        st.error("🔥 **Volatilité élevée**: Investissement risqué")
    elif volatility_level > 0.01:
        st.warning("⚠️ **Volatilité modérée**: Risque modéré")
    else:
        st.success("✅ **Faible volatilité**: Investissement relativement stable")

# Footer
st.markdown("---")
st.markdown("""
**Note**: Cette analyse est basée sur les données historiques et ne constitue pas un conseil financier. 
Les performances passées ne préjugent pas des performances futures.
""")
