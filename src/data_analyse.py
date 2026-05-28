import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import chi2_contingency

class FonctionsStats:

    def valeur_modale(self, series : pd.Series):
        return series.mode()

    def moyenne(self, series : pd.Series):
        return series.mean().round(2)

    def mediane(self, series : pd.Series):
        return series.median().round(2)

    def maximum(self, series : pd.Series):
        return series.max().round(2)
    
    def minimum(self, series : pd.Series):
        return series.min().round(2)
    
    def etendue(self, series : pd.Series):
        return (series.max() - series.min()).round(2)
    
    def somme(self, series : pd.Series):
        return series.sum().round(2)

    def quantile(self, series : pd.Series, quant : float):
        return series.quantile(q = quant).round(2)

    def variance(self, series : pd.Series, ddof : int = 1):
        return series.var(ddof=ddof).round(2)

    def ecart_type(self, series : pd.Series, ddof : int = 1):
        return series.std(ddof=ddof).round(2)
    
    def unique_vals(self, series : pd.Series):
        print(f"Valeurs uniques de la colonne {series.name} :\n{series.unique()}")
    
    def v_cramer(self, chi2, n, r, c):
        return np.sqrt(chi2 / (n * (min(r, c) - 1)))

    def force_v_cramer(self, cramer_v):
        """Interprétation de la force du lien (rule of thumb)
        0.00 - 0.10 : négligeable / 0.10 - 0.30 : faible / 0.30 - 0.50 : moyen / >= 0.50 : fort"""
        if cramer_v < 0.10:
            return "négligeable"
        elif cramer_v < 0.30:
            return "faible"
        elif cramer_v < 0.50:
            return "moyen"
        else:
            return "fort"

class DataAnalyse(FonctionsStats):

    def analyse_var_numeriques(self, df : pd.DataFrame):
        """Analyse des statistiques des variables numériques"""

        df_stats = []
        # cols_num = [j for j in df.columns if np.issubdtype(df[j].dtype, np.number)]
        cols_num = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]   

        print(f"Analyse des variables numériques :\n{', '.join(cols_num)}")
        for i in cols_num:
            df_stats.append({
                'moyenne' : self.moyenne(df[i]),
                'quantile 25%' : self.quantile(df[i], quant=0.25),
                'mediane (50%)' : self.mediane(df[i]),
                'quantile 75%' : self.quantile(df[i], quant=0.75),
                'maximum' : self.maximum(df[i]),
                'minimum' : self.minimum(df[i]),
                'etendue' : self.etendue(df[i]),
                'somme' : self.somme(df[i]),
                'variance' : self.variance(df[i]),
                'ecart type' : self.ecart_type(df[i])
            })
        df_stats = pd.DataFrame(df_stats)
        df_stats.index = cols_num
        return df_stats
    
    def analyse_var_categorielles(self, df : pd.DataFrame, col_to_ignore : list[str]):
        """Analyse des statistiques des variables catégorielles"""
        
        print("Analyse des variables catégorielles :")
        for j in df.columns:
            if pd.api.types.is_string_dtype(df[j]) and j not in col_to_ignore:
                print(f"Variable {j}")
                df_catg = df.groupby(by = j)\
                    .size()\
                    .reset_index(name = 'Occurence')\
                    .sort_values(by="Occurence", ascending=False)\
                    .reset_index(drop=True)
                df_catg['Pourcentage (%)'] = ((df_catg['Occurence']/sum(df_catg['Occurence']))*100).round(2)
                print(df_catg, '\n')
                


    def analyse_var_temporelles(self, df : pd.DataFrame):
        """Analyse des variables temporelles"""
        
        print("Analyse des variables temporelles :")
        col_date = [x for x in df.columns if pd.api.types.is_datetime64_any_dtype(df[x])]
        
        # Calcul de l'intervalle de temps
        for c in col_date:
            print("\nStatistiques de la variable temporelle : ", c)

            print(df[c].describe())
            intervalle = max(df[c]) - min(df[c])

            # Conversion en années, mois et jours
            annees = intervalle.days // 365
            mois = (intervalle.days % 365) // 30
            jours = (intervalle.days % 365) % 30

            print(f"Intervalle de temps des données de la colonnes {c} : {annees} années, {mois} mois, et {jours} jours")


    def groupby_data(self, df : pd.DataFrame, col_x : list[str], col_y : str, operation : list[str]):
        tmp = df.groupby(by=col_x, as_index=False)[col_y].agg(operation)
        if operation == 'count':
            tmp['prct'] = round((tmp['count'] / tmp['count'].sum())*100,2)
            tmp['prct'] = [f"{x}%" for x in tmp['prct']]
        return pd.DataFrame(tmp)


    def calcul_khi2(self, df : pd.DataFrame, alpha=0.05):
        """
        Calcule pour toutes les paires de colonnes catégorielles :
        - Khi2, p-value, ddl
        - V de Cramer (force du lien)
        - interprétation de la force du lien
        
        Retourne un DataFrame avec ces résultats.
        """
        # Sélectionner les colonnes catégorielles ou discrètes
        df['defaulted'] = df['defaulted'].map({0 : 'Non défaut', 1 : 'Défaut'})
        cat_cols = df.select_dtypes(include=["object", "category", "str"]).columns.tolist()
        
        # Filtrer les colonnes avec trop de catégories (optionnel, à ajuster)
        cat_cols = [c for c in cat_cols if df[c].nunique() <= 15]
        results = []
        
        for col1, col2 in combinations(cat_cols, 2):
            # Tableau de contingence
            table = pd.crosstab(df[col1], df[col2])
            
            # Ignorer si une case a un effectif 0 (optionnel, selon ton besoin)
            if table.min().min() == 0:
                continue
            
            chi2, p_value, dof, expected = chi2_contingency(table)
            n = table.values.sum()
            r, c = table.shape
            
            # V de Cramer
            cramer_v = self.v_cramer(chi2=chi2, n=n, r=r, c=c)
            
            # Interprétation de la force du lien (rule of thumb)
            force = self.force_v_cramer(cramer_v=cramer_v)
            
            results.append({
                "var1": col1,
                "var2": col2,
                "chi2": chi2,
                "p_value": p_value,
                "dof": dof,
                "cramer_v": cramer_v,
                "force_lien": force,
                "significatif": p_value < alpha
            })
        
        return pd.DataFrame(results)