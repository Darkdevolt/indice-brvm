import streamlit as st
import pandas as pd
import pandas_ta as ta  # Bibliothèque populaire pour les indicateurs techniques

# --- Configuration de la page Streamlit ---
st.set_page_config(
    page_title="Analyseur de Stratégies Forex",
    page_icon="📈",
    layout="wide"
)

# --- Fonctions de Stratégie ---
# Chaque fonction de stratégie prend un DataFrame et retourne des signaux d'achat (buy) et de vente (sell).

def strategie_croisement_mm(df, mm_courte=20, mm_longue=50):
    """ Stratégie basée sur le croisement de deux moyennes mobiles. """
    df['MM_Courte'] = ta.sma(df['Close'], length=mm_courte)
    df['MM_Longue'] = ta.sma(df['Close'], length=mm_longue)
    
    # Le signal d'achat est quand la MM courte croise AU-DESSUS de la MM longue
    df['signal_buy'] = (df['MM_Courte'] > df['MM_Longue']) & (df['MM_Courte'].shift(1) <= df['MM_Longue'].shift(1))
    
    # Le signal de vente est quand la MM courte croise EN-DESSOUS de la MM longue
    df['signal_sell'] = (df['MM_Courte'] < df['MM_Longue']) & (df['MM_Courte'].shift(1) >= df['MM_Longue'].shift(1))
    
    return df

def strategie_rsi(df, rsi_length=14, surachat=70, survente=30):
    """ Stratégie basée sur le Relative Strength Index (RSI). """
    df['RSI'] = ta.rsi(df['Close'], length=rsi_length)
    
    # Le signal d'achat est quand le RSI sort de la zone de survente
    df['signal_buy'] = (df['RSI'] > survente) & (df['RSI'].shift(1) <= survente)
    
    # Le signal de vente est quand le RSI sort de la zone de surachat
    df['signal_sell'] = (df['RSI'] < surachat) & (df['RSI'].shift(1) >= surachat)

    return df

def strategie_macd(df, fast=12, slow=26, signal=9):
    """ Stratégie basée sur la MACD. """
    macd = ta.macd(df['Close'], fast=fast, slow=slow, signal=signal)
    df['MACD'] = macd[f'MACD_{fast}_{slow}_{signal}']
    df['MACD_Signal'] = macd[f'MACDs_{fast}_{slow}_{signal}']
    
    # Le signal d'achat est quand la ligne MACD croise AU-DESSUS de sa ligne de signal
    df['signal_buy'] = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
    
    # Le signal de vente est quand la ligne MACD croise EN-DESSOUS de sa ligne de signal
    df['signal_sell'] = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
    
    return df

def strategie_bollinger(df, length=20, std=2):
    """ Stratégie basée sur les Bandes de Bollinger. """
    bollinger = ta.bbands(df['Close'], length=length, std=std)
    df['BB_Lower'] = bollinger[f'BBL_{length}_{std}']
    df['BB_Upper'] = bollinger[f'BBU_{length}_{std}']
    
    # Achat quand le prix touche ou casse la bande inférieure
    df['signal_buy'] = (df['Low'] < df['BB_Lower']) & (df['Low'].shift(1) >= df['BB_Lower'].shift(1))
    
    # Vente quand le prix touche ou casse la bande supérieure
    df['signal_sell'] = (df['High'] > df['BB_Upper']) & (df['High'].shift(1) <= df['BB_Upper'].shift(1))

    return df


# --- Moteur de Backtesting ---
def backtest_engine(df, stop_loss_pct=0.01, risk_reward_ratio=3.0):
    """
    Exécute le backtest sur un DataFrame qui contient déjà des colonnes 'signal_buy' et 'signal_sell'.
    """
    trades = []
    position_ouverte = None  # Peut être 'long' (achat), 'short' (vente) ou None

    for i in range(1, len(df)):
        # Gérer une position LONG (achat)
        if position_ouverte == 'long':
            # Vérifier Take Profit
            if df['High'].iloc[i] >= take_profit_price:
                trades.append({'type': 'long', 'resultat': 'win', 'sortie': take_profit_price, 'date_sortie': df.index[i]})
                position_ouverte = None
                continue
            # Vérifier Stop Loss
            elif df['Low'].iloc[i] <= stop_loss_price:
                trades.append({'type': 'long', 'resultat': 'loss', 'sortie': stop_loss_price, 'date_sortie': df.index[i]})
                position_ouverte = None
                continue

        # Gérer une position SHORT (vente)
        elif position_ouverte == 'short':
            # Vérifier Take Profit
            if df['Low'].iloc[i] <= take_profit_price:
                trades.append({'type': 'short', 'resultat': 'win', 'sortie': take_profit_price, 'date_sortie': df.index[i]})
                position_ouverte = None
                continue
            # Vérifier Stop Loss
            elif df['High'].iloc[i] >= stop_loss_price:
                trades.append({'type': 'short', 'resultat': 'loss', 'sortie': stop_loss_price, 'date_sortie': df.index[i]})
                position_ouverte = None
                continue

        # Ouvrir une nouvelle position si aucune n'est ouverte
        if position_ouverte is None:
            # Signal d'achat
            if df['signal_buy'].iloc[i]:
                position_ouverte = 'long'
                entry_price = df['Open'].iloc[i]
                stop_loss_price = entry_price * (1 - stop_loss_pct)
                take_profit_price = entry_price * (1 + (stop_loss_pct * risk_reward_ratio))
                trades.append({'type': 'long', 'resultat': 'pending', 'entree': entry_price, 'date_entree': df.index[i], 'sl': stop_loss_price, 'tp': take_profit_price})
            
            # Signal de vente
            elif df['signal_sell'].iloc[i]:
                position_ouverte = 'short'
                entry_price = df['Open'].iloc[i]
                stop_loss_price = entry_price * (1 + stop_loss_pct)
                take_profit_price = entry_price * (1 - (stop_loss_pct * risk_reward_ratio))
                trades.append({'type': 'short', 'resultat': 'pending', 'entree': entry_price, 'date_entree': df.index[i], 'sl': stop_loss_price, 'tp': take_profit_price})

    # Analyser les résultats
    if not trades:
        return None

    results_df = pd.DataFrame(trades)
    # On ne garde que les trades fermés
    closed_trades = results_df[results_df['resultat'] != 'pending'].copy()
    
    if closed_trades.empty:
        return {
            'methode': 'N/A',
            'nb_trades': 0,
            'win_rate': 0,
            'gains': 0,
            'pertes': 0,
            'repond_critere': False
        }

    wins = closed_trades[closed_trades['resultat'] == 'win']
    losses = closed_trades[closed_trades['resultat'] == 'loss']
    
    win_rate = len(wins) / len(closed_trades) if len(closed_trades) > 0 else 0
    
    return {
        'nb_trades': len(closed_trades),
        'win_rate': win_rate * 100,
        'gains': len(wins),
        'pertes': len(losses),
        'repond_critere': win_rate >= 0.5
    }

# --- Interface Utilisateur Streamlit ---
st.title("🔎 Testeur de Stratégies de Trading Forex")

st.markdown("""
Cette application teste plusieurs stratégies de trading sur des données historiques que vous fournissez.
Elle applique une gestion de risque stricte : **Stop-Loss de 1%** et **Ratio Risque/Récompense de 1:3**.
Le but est d'identifier les combinaisons d'indicateurs qui auraient atteint un **Win Rate de 50% ou plus** sur la période testée.

**⚠️ AVERTISSEMENT :** Les performances passées ne préjugent pas des performances futures. Cet outil est à but éducatif uniquement.
""")

uploaded_file = st.file_uploader("Choisissez un fichier CSV de données historiques", type="csv")

if uploaded_file is not None:
    try:
        # Lecture et préparation des données
        data = pd.read_csv(uploaded_file)
        # S'assurer que les noms de colonnes sont standardisés (minuscules)
        data.columns = [col.lower() for col in data.columns]
        
        # Vérification des colonnes nécessaires
        required_cols = ['date', 'open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            st.error(f"Le fichier CSV doit contenir les colonnes suivantes : {', '.join(required_cols)}")
        else:
            data['date'] = pd.to_datetime(data['date'])
            data = data.set_index('date')
            data = data.sort_index()
            # Renommer pour la compatibilité avec la bibliothèque `ta`
            data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)

            st.success(f"Fichier chargé avec succès ! {len(data)} lignes de données trouvées.")

            # Définition des stratégies à tester
            strategies = {
                "Croisement Moyennes Mobiles (20/50)": strategie_croisement_mm,
                "RSI (14, 70/30)": strategie_rsi,
                "MACD (12/26/9)": strategie_macd,
                "Bandes de Bollinger (20, 2)": strategie_bollinger
            }
            
            all_results = []
            progress_bar = st.progress(0)
            
            for i, (name, func) in enumerate(strategies.items()):
                st.write(f"--- \n**Test de la stratégie : {name}**")
                
                # Copier les données pour ne pas les altérer
                df_strategy = data.copy()
                
                # Appliquer la stratégie pour obtenir les signaux
                df_with_signals = func(df_strategy)
                
                # Exécuter le backtest
                resultats = backtest_engine(df_with_signals, stop_loss_pct=0.01, risk_reward_ratio=3.0)
                
                if resultats and resultats['nb_trades'] > 0:
                    resultats['methode'] = name
                    all_results.append(resultats)
                    
                    # Affichage des résultats pour cette stratégie
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Nombre de Trades", resultats['nb_trades'])
                    col2.metric("Trades Gagnants", resultats['gains'])
                    col3.metric("Trades Perdants", resultats['pertes'])
                    col4.metric("Win Rate", f"{resultats['win_rate']:.2f}%")

                    if resultats['repond_critere']:
                        st.success(f"✅ Cette méthode a atteint les critères sur les données historiques !")
                    else:
                        st.warning(f"❌ Cette méthode n'a pas atteint le Win Rate de 50%.")

                else:
                    st.write("Aucun trade n'a été exécuté pour cette stratégie sur la période donnée.")
                
                progress_bar.progress((i + 1) / len(strategies))

            # Affichage du tableau récapitulatif
            st.write("---")
            st.header("🏆 Tableau de Bord des Résultats")
            
            if all_results:
                results_df = pd.DataFrame(all_results)
                results_df = results_df[['methode', 'nb_trades', 'win_rate', 'gains', 'pertes', 'repond_critere']]
                results_df['win_rate'] = results_df['win_rate'].apply(lambda x: f"{x:.2f}%")
                
                # Mettre en évidence les "gagnants"
                st.dataframe(results_df.style.apply(
                    lambda row: ['background-color: #28a745' if row.repond_critere else '' for _ in row],
                    axis=1
                ))

                st.subheader("Méthodes qui marchent (sur ces données historiques) :")
                gagnants = results_df[results_df['repond_critere'] == True]
                if not gagnants.empty:
                    st.table(gagnants[['methode', 'win_rate', 'nb_trades']])
                else:
                    st.info("Aucune des stratégies testées n'a rempli les critères sur la période de données fournie.")
            
            else:
                st.warning("Aucun résultat à afficher.")
    
    except Exception as e:
        st.error(f"Une erreur est survenue lors du traitement du fichier : {e}")
