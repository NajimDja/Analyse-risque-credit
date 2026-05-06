# Methodes d'analyse de la qualité de la données
import pandas as pd
import re

class DataQuality:

    def __init__(self):
        pass

    def check_nulls(self, df : pd.DataFrame):
        """Check le nombre, pourcentage de nulls par colonne"""
        df_null = []
        for i in df.columns:
            nb_null = df[i].isna().sum()
            prct_null = nb_null/len(df)
            df_null.append({'Colonne' : i,
                            'Nombre de null' : nb_null,
                            '% de null' : prct_null})
        df_null = pd.DataFrame(df_null)
        print(df_null)


    def check_types(self, df : pd.DataFrame):
        """Affiche le type de chaque colonne"""
        print(df.dtypes)


    def check_unique_values(self, df : pd.DataFrame, cols  : list[str] | None = None):
        """Check si le dataframe a des lignes dupliquées"""
        total_rows = len(df)
        unique_rows = len(df.drop_duplicates(subset=cols))
        print(f"Nombre de lignes totales : {total_rows}")
        print(f"Nombre de lignes uniques : {unique_rows}")
        print(f"Nombre de lignes dupliquées : {total_rows - unique_rows}")


    def check_date_format(self, df: pd.DataFrame, threshold: float = 0.6):
        """Check les colonnes avec des données au format de dates"""
        
        patterns = [
            r"^\d{4}[-/]\d{2}[-/]\d{2}$",          # 2024-12-31, 2024/12/31
            r"^\d{2}[-/]\d{2}[-/]\d{4}$",          # 31-12-2024, 31/12/2024
            r"^\d{4}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}:\d{2}$",  # datetime complet
            r"^\d{4}[-/]\d{2}[-/]\d{2}T\d{2}:\d{2}:\d{2}$",     # ISO 8601 simplifié
        ]

        date_cols = []
        cols = []

        for col in df.columns:
            len_col = len(df[col])
            s = df[col].dropna().astype(str).str.strip()
            if s.empty:
                continue

            total = len(s)
            matches = sum(
                s.str.match(pat, na=False).sum()
                for pat in patterns
            )

            if total > 0 and (matches / total) >= threshold:
                date_cols.append({"Colonne" : col, 
                                  "Cohérence sur lignes valides" : str(matches) + "/" + str(total), 
                                  "Total lignes" : len_col})
                cols.append(col)

        date_cols = pd.DataFrame(date_cols)
        print(f"Colonnes ayant des données au format de date\n")
        print(date_cols)
        return cols