# Methodes d'analyse de la qualité de la données
# Ecriture d'un rapport sur la qualité de la données
import pandas as pd

class DataQuality:

    def __init__(self):
        pass

    def check_nulls(self, df : pd.DataFrame):
        df_null = []
        for i in df.columns:
            nb_null = df[i].isna().sum()
            prct_null = nb_null/len(df)
            df_null.append({'Colonne' : i,
                            'Nombre de null' : nb_null,
                            '% de null' : prct_null})
        df_null = pd.DataFrame(df_null)
        print(df_null)