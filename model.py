import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

class EmotionRecognitionModel:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        print(self.df.describe(include=["O"]))
        self.features = ['MFCCs', 'Pitch Mean', 'Pitch Contour', 'Energy', 'Formant F1', 'Formant F2', 'Formant F3', 'Speech Rate', 'Zero Crossing Rate']
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()


    def load_data(self):
        # Handle missing values
        self.df = self.df.dropna(subset=self.features + ['Emotion Name'])

        # Convert MFCCs column to arrays of floats
        def parse_mfccs(mfcc_str):
            if isinstance(mfcc_str, str):
                # Remove brackets, commas, and extra spaces
                mfcc_str = mfcc_str.replace('[', '').replace(']', '').replace(',', '').strip()
                # Convert the cleaned string to a float array
                mfcc_array = np.array(mfcc_str.split(), dtype=float)
                return mfcc_array
            return np.array([])  # Handle any non-string cases gracefully

        # Parse MFCCs and check lengths
        mfcc_arrays = self.df['MFCCs'].apply(parse_mfccs)
        max_length = max(len(arr) for arr in mfcc_arrays)
        self.X_mfcc = np.array([np.pad(arr, (0, max_length - len(arr)), mode='constant') for arr in mfcc_arrays])

        # Extract other features
        self.pitch_mean = self.df['Pitch Mean'].values
        self.pitch_contour = self.df['Pitch Contour'].apply(parse_mfccs).apply(lambda x: np.mean(x)).values  # Example handling pitch contour
        self.energy = self.df['Energy'].values
        self.formant_f1 = self.df['Formant F1'].values
        self.formant_f2 = self.df['Formant F2'].values
        self.formant_f3 = self.df['Formant F3'].values
        self.speech_rate = self.df['Speech Rate'].values
        self.zero_crossing_rate = self.df['Zero Crossing Rate'].values

        # Extract labels
        self.y = self.df['Emotion Name'].values


    def preprocess_data(self):
        # Scale features
        self.X_mfcc_scaled = self.scaler.fit_transform(self.X_mfcc)
        self.pitch_mean_scaled = self.scaler.fit_transform(self.pitch_mean.reshape(-1, 1))
        self.pitch_contour_scaled = self.scaler.fit_transform(self.pitch_contour.reshape(-1, 1))
        self.energy_scaled = self.scaler.fit_transform(self.energy.reshape(-1, 1))
        self.formant_f1_scaled = self.scaler.fit_transform(self.formant_f1.reshape(-1, 1))
        self.formant_f2_scaled = self.scaler.fit_transform(self.formant_f2.reshape(-1, 1))
        self.formant_f3_scaled = self.scaler.fit_transform(self.formant_f3.reshape(-1, 1))
        self.speech_rate_scaled = self.scaler.fit_transform(self.speech_rate.reshape(-1, 1))
        self.zero_crossing_rate_scaled = self.scaler.fit_transform(self.zero_crossing_rate.reshape(-1, 1))

        # Combine features into a single dataset
        self.X_combined = np.concatenate([
            self.X_mfcc_scaled,
            self.pitch_mean_scaled,
            self.pitch_contour_scaled,
            self.energy_scaled,
            self.formant_f1_scaled,
            self.formant_f2_scaled,
            self.formant_f3_scaled,
            self.speech_rate_scaled,
            self.zero_crossing_rate_scaled
        ], axis=1)


        # Encode labels
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        self.y_one_hot = to_categorical(self.y_encoded)
        print(self.y_one_hot.shape)
       
    def create_lstm_model(self):
        # Create LSTM model
        model = Sequential()
        model.add(LSTM(units=128, return_sequences=True, input_shape=(1, self.X_combined.shape[1])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(7, activation='softmax', kernel_regularizer=l2(0.001)))  # Assuming 8 emotion categories
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model

    def train_model(self, epochs=50, batch_size=64):
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(self.X_combined, self.y_one_hot, test_size=0.2, random_state=42)
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])  # Reshape for LSTM input (timesteps = 1)
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        # Train model
        history=self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),callbacks=[early_stopping],shuffle=True)
        return history

    def evaluate_model(self):
        # Split data for testing
        X_train, X_test, y_train, y_test = train_test_split(self.X_combined, self.y_one_hot, test_size=0.2, random_state=42)
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        # Evaluate the model
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f'Test accuracy: {accuracy:.3f}')

    def evaluate_model_with_confusion_matrix(self):
        # Split the data again
        X_train, X_test, y_train, y_test = train_test_split(self.X_combined, self.y_one_hot, test_size=0.2, random_state=42)
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        # Get predictions (as probabilities)
        y_pred_probs = self.model.predict(X_test)

        # Convert probabilities to class labels
        y_pred_classes = np.argmax(y_pred_probs, axis=1)
        y_true_classes = np.argmax(y_test, axis=1)
        report = classification_report(y_true_classes, y_pred_classes, target_names=self.label_encoder.classes_)
        print("Classification Report:\n", report)

        # Compute confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)

        # Plot confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()


    def save_model(self, file_path):
        """Save the trained model to the specified file path."""
        self.model.save(file_path)
        # After training, save the scaler
        joblib.dump(self.scaler, 'scaler.pkl')
        np.save('label_encoder_classes.npy', self.label_encoder.classes_)  # Save LabelEncoder classes
        print(f"Model and LabelEncoder saved to {file_path} and label_encoder_classes.npy")

def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Usage example
model = EmotionRecognitionModel('Merged_CSV.csv')
model.load_data()
model.preprocess_data()
model.create_lstm_model()
history = model.train_model()
model.evaluate_model()
model.evaluate_model_with_confusion_matrix()
model.save_model('emotion_recognition_model.h5')
plot_training_history(history)
