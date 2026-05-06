import pandas as pd

class DataTransformation:

    def col_mapping(self, df : pd.DataFrame, col_to_map : str, new_col_name : str, mapper):
        """Map une colonne existante en fonction d'un dictionnaire (clé:valeur) ou une fonction ('Je suis {}'.format)"""
        df[new_col_name] = df[col_to_map]\
            .map(mapper, na_action='ignore')
        return df
    
    def set_col_type(self, series : pd.Series, type : str):
        """Change le type d'une colonne"""
        return series.astype(type)
    
    def set_datetime_type(self, series : pd.Series, date_format : str = "%Y-%m-%d"):
        """Change le type d'une colonne en datetime"""
        series = pd.to_datetime(series, 
                                format=date_format, 
                                errors='coerce')
        return series