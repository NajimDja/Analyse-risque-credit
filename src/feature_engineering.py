import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, QuantileTransformer, OneHotEncoder


class FeatureEncoding:
    """
    Class regroupant les méthodes d'encodage des variables à des fins de modélisation
    
    Méthodes:
     - min_max_scaler: Ramène les données entre un intervalle de 0 et 1 ou un autre spécifié en entrée.
     - robust_scaler: Centre les données autour de la médiane et met à l'échelle en fonction de l'intervalle interquartile (IQR)
     - quantile_transformer: Mappe les données à une distribution uniforme entre 0 et 1, tout en étant robuste aux valeurs aberrantes
     - feature_scaler: scale les variables avec un des standardiseur ci-dessus, permet de récupérer le scaler fit sur les données de train.
    """
    
    def one_hot_encoding_y(self, df : pd.DataFrame, col : str, prefix : str) -> pd.DataFrame:
        """
        One hot encoding de la target y.
        """
        df = pd.get_dummies(df, columns=[col], prefix=[prefix], drop_first=False, dtype=int)
        return df

    def min_max_scaler(self, X, feature_range : tuple = (0,1)):
        """
        Ramène les données entre un intervalle de 0 et 1 ou un autre spécifié en entrée.\n
        Utilisation pour un df -> df['X_unif'], _ = FeatureEncoding().quantile_transformer(X = df[['X']])
        """
        scaler = MinMaxScaler(feature_range = feature_range)
        scaled_data = scaler.fit_transform(X)
        X = scaled_data
        return X, scaler
    
    def robust_scaler(self, X, quantile_range : tuple = (0.25, 0.75)):
        """
        Centre les données autour de la médiane et met à l'échelle en fonction de l'intervalle interquartile (IQR)\n
        Utilisation pour un df -> df['X_unif'], _ = FeatureEncoding().quantile_transformer(X = df[['X']])\n
         Attention :
         - Il ne mets pas les valeurs entre 0 et 1
        """
        scaler = RobustScaler(quantile_range = quantile_range)
        scaled_data = scaler.fit_transform(X)
        X = scaled_data
        return X, scaler

    def quantile_transformer(self, X, output_distribution : str = 'uniform'):
        """
        Mappe les données à une distribution uniforme entre 0 et 1, tout en étant robuste aux valeurs aberrantes.\n
        Utilisation pour un df -> df['X_unif'], _ = FeatureEncoding().quantile_transformer(X = df[['X']])\n
        Options de output_distribution :
         - uniform
         - normal
        """
        scaler = QuantileTransformer(output_distribution = output_distribution, random_state=0)
        scaled_data = scaler.fit_transform(X)
        X = scaled_data
        return X, scaler
    
    def feature_scaler(self, X, scaler, feature_indices):
        """
        Scale selected features using the provided scaler
        """
        X[:, feature_indices] = scaler.transform(X[:, feature_indices])
        return X