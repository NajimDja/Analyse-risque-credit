import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, QuantileTransformer, StandardScaler
from sklearn.model_selection import train_test_split

class SplitData:

    def split_in_train_test_stratify(self, df : pd.DataFrame, y : str, test_size : int = 0.2):
        """Diviser la données en dataset d'entrainement et de test"""

        X = df.drop(columns=[y])
        y = df[y]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=test_size, 
            random_state=100, 
            stratify=y)
        
        return X_train, X_test, y_train, y_test


class FeatureEncoding:
    """
    Class regroupant les méthodes d'encodage des variables à des fins de modélisation
    
    Méthodes:
     - one_hot_encoding
     - min_max_scaler: Ramène les données entre un intervalle de 0 et 1 ou un autre spécifié en entrée.
     - robust_scaler: Centre les données autour de la médiane et met à l'échelle en fonction de l'intervalle interquartile (IQR)
     - quantile_transformer: Mappe les données à une distribution uniforme entre 0 et 1, tout en étant robuste aux valeurs aberrantes
     - feature_scaler: scale les variables avec un des standardiseur ci-dessus, permet de récupérer le scaler fit sur les données de train.
    """
    
    def one_hot_encoding(self, df : pd.DataFrame, col : list[str]) -> pd.DataFrame:
        """
        One hot encoding de la target.
        """
        return pd.get_dummies(data = df, columns=col, drop_first=False, dtype=int)
    

    def min_max_scaler(self, X, feature_range : tuple = (0,1)):
        """
        Ramène les données entre un intervalle de 0 et 1 ou un autre spécifié en entrée.\n
        Utilisation pour un df -> df['X_unif'], _ = FeatureEncoding().quantile_transformer(X = df[['X']])
        """
        scaler = MinMaxScaler(feature_range = feature_range)
        X_scaled = scaler.fit_transform(X)
        return X_scaled
    
    def standard_scaler(self, X):
        """
        Standardisation des données
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled
    
    def robust_scaler(self, X, quantile_range : tuple = (0.25, 0.75)):
        """
        Centre les données autour de la médiane et met à l'échelle en fonction de l'intervalle interquartile (IQR)\n
        Utilisation pour un df -> df['X_unif'], _ = FeatureEncoding().quantile_transformer(X = df[['X']])\n
         Attention :
         - Il ne mets pas les valeurs entre 0 et 1
        """
        scaler = RobustScaler(quantile_range = quantile_range)
        X_scaled = scaler.fit_transform(X)
        return X_scaled

    def quantile_transformer(self, X, output_distribution : str = 'uniform'):
        """
        Mappe les données à une distribution uniforme entre 0 et 1, tout en étant robuste aux valeurs aberrantes.\n
        Utilisation pour un df -> df['X_unif'], _ = FeatureEncoding().quantile_transformer(X = df[['X']])\n
        Options de output_distribution :
         - uniform
         - normal
        """
        scaler = QuantileTransformer(output_distribution = output_distribution, random_state=0)
        X_scaled = scaler.fit_transform(X)
        return X_scaled
    
    # def feature_scaler(self, X, scaler, feature_indices):
    #     """
    #     Scale selected features using the provided scaler
    #     """
    #     X[:, feature_indices] = scaler.transform(X[:, feature_indices])
    #     return X