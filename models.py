class MinMaxScaler:
    def __init__(self, features=None, feature_range=(-1, 1)):
        self.features = features
        self.feature_range = feature_range

    def fit(self, df, exclude=None):
        if self.features is None:
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            self.features_ = numeric_cols
        else:
            self.features_ = self.features.copy()

        if exclude:
            self.features_ = [f for f in self.features_ if f not in exclude]

        self.min_ = df[self.features_].min()
        self.max_ = df[self.features_].max()

        zero_range = (self.max_ - self.min_) == 0
        if zero_range.any():
            zero_cols = zero_range[zero_range].index.tolist()
            raise ValueError(f"Features have zero range (constant values): {zero_cols}")

        return self

    def transform(self, df, round_decimals=4):
        if not hasattr(self, 'min_'):
            raise ValueError("Scaler not fitted. Call fit() first.")

        df_scaled = df.copy()

        target_min, target_max = self.feature_range
        target_range = target_max - target_min

        for col in self.features_:
            if col not in df.columns:
                raise ValueError(f"Feature '{col}' not found in dataframe")

            normalized = (df[col] - self.min_[col]) / (self.max_[col] - self.min_[col])
            df_scaled[col] = (normalized * target_range + target_min).round(round_decimals)

        return df_scaled

    def fit_transform(self, df, exclude=None, round_decimals=4):
        return self.fit(df, exclude=exclude).transform(df, round_decimals=round_decimals)

    def inverse_transform(self, df_scaled, columns=None):
        if not hasattr(self, 'min_'):
            raise ValueError("Scaler not fitted. Call fit() first.")

        target_min, target_max = self.feature_range
        target_range = target_max - target_min

        df_original = df_scaled.copy()
        cols_to_unscale = columns or self.features_

        for col in cols_to_unscale:
            if col in df_original.columns:
                normalized = (df_scaled[col] - target_min) / target_range
                df_original[col] = normalized * (self.max_[col] - self.min_[col]) + self.min_[col]

        return df_original
