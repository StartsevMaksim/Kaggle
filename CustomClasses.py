import numpy as np
import pandas as pd
from collections import deque
from sklearn.model_selection import train_test_split, GridSearchCV

class TreeBoostingTrainer:
    def __init__(self, model, model_params, grid_search_gap=5, min_data_part=0.5):
        self.model = model
        self.model_params = model_params
        self.grid_search_gap = grid_search_gap
        self.min_data_part = min_data_part

    def _getParamStartVal(self, value):
        return 2 * value // (self.grid_search_gap + 1)
    
    def _getParamsBiasArray(self, epochs):
        params_bias = {}
        for param, value in self.model_params.items():
            bias_arr = [self._getParamStartVal(value)]
            if epochs > 1:
                delim = np.power(bias_arr[-1], 1/(epochs-1))
                for _ in range(epochs-1):
                    bias_arr.append(bias_arr[-1]/delim)
            bias_arr = deque(map(lambda x: int(np.round(x)), bias_arr))
            params_bias[param] = bias_arr
        
        result = []
        for _ in range(epochs):
            bias_dict = {}
            for param, bias_arr in params_bias.items():
                bias_dict[param] = bias_arr.popleft()
            result.append(bias_dict)
        return result
    
    def _getGridParams(self, params_bias):
        result = {}
        for param, bias in params_bias.items():
            values = [max(1, self.model_params[param]-bias*(self.grid_search_gap//2))]
            for _ in range(self.grid_search_gap-1):
                values.append(values[-1]+bias)
            result[param] = values 
        return result

    @staticmethod
    def train_model(model, df_train, target_column):
        X = df_train.drop(target_column, axis=1)
        y = df_train[target_column] 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        print(f'Параметры: {model.best_params_}')
        print(f'Ошибка модели = {root_mean_squared_error(y_test, y_predict)}')
    
    def train(self, df_train, epochs, target_column, scoring, verbose=0):
        data_part_perc = np.linspace(self.min_data_part, 1, epochs)
        params_bias_array = self._getParamsBiasArray(epochs)
        for part_perc, params_bias in zip(data_part_perc, params_bias_array):
            df_train_part = df_train.iloc[np.random.choice(list(range(df_train.shape[0])),
                                                           size=int(df_train.shape[0]*part_perc),
                                                           replace=False)]
            grid_params = self._getGridParams(params_bias)
            print(f'Параметры сетки\n{grid_params}')
            print(f'Размер данных ({np.round(part_perc*100, 2)}%)\n{df_train_part.shape}')
            grid_model = GridSearchCV(self.model, grid_params, scoring=scoring, cv=5, verbose=verbose)
            self.train_model(grid_model, df_train_part, target_column)
            self.model_params = grid_model.best_params_