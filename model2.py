import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier

# Data preprocessing
class DataPreprocessor:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def load_and_preprocess_data(self):
        # Handle missing values
        self.df = self.df.dropna(subset=['MFCCs', 'Pitch Mean', 'Pitch Contour', 'Energy', 'Formant F1', 'Formant F2', 'Formant F3', 'Emotion Name'])
        print(f"Length after dropping NA: {len(self.df)}")
        # Convert MFCCs column to arrays of floats
        def parse_mfccs(mfcc_str):
            if isinstance(mfcc_str, str):
                mfcc_str = mfcc_str.replace('[', '').replace(']', '').replace(',', '').strip()
                mfcc_array = np.array(mfcc_str.split(), dtype=float)
                return mfcc_array
            return np.array([])

        # Parse MFCCs
        mfcc_arrays = self.df['MFCCs'].apply(parse_mfccs)
        max_length = max(len(arr) for arr in mfcc_arrays)
        self.X = np.array([np.pad(arr, (0, max_length - len(arr)), mode='constant') for arr in mfcc_arrays])
        print(f"Length of X: {len(self.X)}")
        # Additional features
        self.pitch_mean = self.df['Pitch Mean'].values
        self.pitch_contour = self.df['Pitch Contour'].apply(parse_mfccs).apply(lambda x: np.mean(x)).values
        self.energy = self.df['Energy'].values
        self.formant_f1 = self.df['Formant F1'].values
        self.formant_f2 = self.df['Formant F2'].values
        self.formant_f3 = self.df['Formant F3'].values
        self.y = self.df['Emotion Name'].values
        if len(self.X) != len(self.y):
            raise ValueError(f"Feature and label length mismatch: {len(self.X)} != {len(self.y)}")

        # Scale features
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.pitch_mean_scaled = self.scaler.fit_transform(self.pitch_mean.reshape(-1, 1))
        self.pitch_contour_scaled = self.scaler.fit_transform(self.pitch_contour.reshape(-1, 1))
        self.energy_scaled = self.scaler.fit_transform(self.energy.reshape(-1, 1))
        self.formant_f1_scaled = self.scaler.fit_transform(self.formant_f1.reshape(-1, 1))
        self.formant_f2_scaled = self.scaler.fit_transform(self.formant_f2.reshape(-1, 1))
        self.formant_f3_scaled = self.scaler.fit_transform(self.formant_f3.reshape(-1, 1))

        # Combine features into a single dataset
        self.X_combined = np.concatenate([
            self.X_scaled,
            self.pitch_mean_scaled,
            self.pitch_contour_scaled,
            self.energy_scaled,
            self.formant_f1_scaled,
            self.formant_f2_scaled,
            self.formant_f3_scaled
        ], axis=1)
        print(f"Length of X_combined: {self.X_combined.shape}")
        # Encode labels
        self.y = self.label_encoder.fit_transform(self.df['Emotion Name'].values)
        print(f"Length of y: {len(self.y)}")
        self.y_one_hot = to_categorical(self.y)
        if len(self.X_combined) != len(self.y_one_hot):
            raise ValueError(f"Feature and label length mismatch: {len(self.X_combined)} != {len(self.y_one_hot)}")

        return self.X_combined, self.y, self.y_one_hot

# Build LSTM model
def build_lstm(units=64, optimizer='adam'):
    model = Sequential()
    model.add(LSTM(units=units, input_shape=(None, 1)))
    model.add(Dense(8, activation='softmax'))  # Assuming 8 emotion categories
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Build RNN model
def build_rnn(units=64, optimizer='adam'):
    model = Sequential()
    model.add(SimpleRNN(units=units, input_shape=(None, 1)))
    model.add(Dense(8, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# GridSearchCV for SVM
def svm_model_tuning(X_train, y_train):
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

# LSTM Tuning
def lstm_model_tuning(X_train, y_train):
    model = KerasClassifier(build_fn=build_lstm, verbose=0)
    param_grid = {
        'batch_size': [32, 64],
        'epochs': [50, 100],
        'units': [64, 128],
        'optimizer': ['adam', 'rmsprop']
    }
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X_train, y_train)
    return grid_result.best_estimator_

# RNN Tuning
def rnn_model_tuning(X_train, y_train):
    model = KerasClassifier(build_fn=build_rnn, verbose=0)
    param_grid = {
        'batch_size': [32, 64],
        'epochs': [50, 100],
        'units': [64, 128],
        'optimizer': ['adam', 'rmsprop']
    }
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X_train, y_train)
    return grid_result.best_estimator_

# Main program to train and compare models
if __name__ == "__main__":
    # Load and preprocess data
    preprocessor = DataPreprocessor('Merged_CSV.csv')
    X, y, y_one_hot = preprocessor.load_and_preprocess_data()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")


    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train SVM
    best_svm_model = svm_model_tuning(X_train, y_train)
    print(f"Best SVM Model: {best_svm_model}")

    # Reshape X_train for LSTM/RNN
    X_train_reshaped = X_train.reshape(X_train.shape[0],1,X_train.shape[1])  # Reshape for LSTM input
    X_test_reshaped = X_test.reshape(X_test.shape[0],1,X_test.shape[1]) 
    #X_train_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    #X_test_reshaped = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    print(f"Length of X_train_reshaped: {len(X_train_reshaped)}")
    print(f"Shape of X_train_reshaped: {X_train_reshaped.shape}")
    print(f"Shape of X_test_reshaped: {X_test_reshaped.shape}")

    # Train LSTM
    best_lstm_model = lstm_model_tuning(X_train_reshaped, y_one_hot)
    print(f"Best LSTM Model: {best_lstm_model}")

    # Train RNN
    best_rnn_model = rnn_model_tuning(X_train_reshaped, y_one_hot)
    print(f"Best RNN Model: {best_rnn_model}")

    # Evaluate models on test data
    print("Evaluating models on test data:")
    print(f"SVM Accuracy: {best_svm_model.score(X_test, y_test)}")
    print(f"LSTM Accuracy: {best_lstm_model.score(X_test_reshaped, to_categorical(LabelEncoder().fit_transform(y_test)))}")
    print(f"RNN Accuracy: {best_rnn_model.score(X_test_reshaped, to_categorical(LabelEncoder().fit_transform(y_test)))}")
