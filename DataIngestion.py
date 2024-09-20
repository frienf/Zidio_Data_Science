import zipfile
import os
import pandas as pd
import librosa
import numpy as np
import parselmouth  # For formant frequency extraction from Praat
from parselmouth.praat import call
import scipy.signal as signal  # Required for noise removal in AudioPreprocessing class

class DataIngestion:
    def __init__(self, zip_file_path, extract_to_dir, output_dir):
        self.zip_file_path = zip_file_path
        self.extract_to_dir = extract_to_dir
        self.output_dir = output_dir
        self.audio_file_info_df = None
        self.audio_features_df = None
        self.preprocessor = AudioPreprocessing(sample_rate=16000)

    @staticmethod
    def is_audio_file(file_name):
        """Check if a file is an audio file based on its extension."""
        audio_extensions = ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a', '.wma']
        return any(file_name.lower().endswith(ext) for ext in audio_extensions)

    def extract_audio_files_from_zip(self):
        """Extract audio files from a ZIP archive and save their paths in a DataFrame."""
        # Create the directory if it doesn't exist
        if not os.path.exists(self.extract_to_dir):
            os.makedirs(self.extract_to_dir)

        # Extract files from the ZIP archive
        with zipfile.ZipFile(self.zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.extract_to_dir)

        # Prepare a list to store audio file information
        audio_file_info_list = []

        # Walk through the extracted files
        for root, dirs, files in os.walk(self.extract_to_dir):
            for file in files:
                if self.is_audio_file(file):
                    file_path = os.path.join(root, file)
                    folder_name = os.path.basename(root)  # Extract folder name
                    audio_file_info_list.append({'Folder Name': folder_name, 'File Name': file, 'File Path': file_path})

        # Create a DataFrame to store audio file names and paths
        self.audio_file_info_df = pd.DataFrame(audio_file_info_list)
        return self.audio_file_info_df

    @staticmethod
    def extract_formants(file_path):
        """Extract formant frequencies from an audio file using Praat-parselmouth."""
        snd = parselmouth.Sound(file_path)
        formants = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50.0)
        formant_freqs = [call(formants, "Get value at time", i, 0.5, 'Hertz', 'Linear') for i in range(1, 4)]  # F1, F2, F3
        return formant_freqs

    def extract_audio_features(self):
        """Extract audio features from audio files and store them in a DataFrame."""
        # Prepare a list to store features
        feature_list = []

        for index, row in self.audio_file_info_df.iterrows():
            file_path = row['File Path']
            try:
                # Preprocess the audio file using AudioPreprocessing class
                preprocessed_audio = self.preprocessor.preprocess(file_path)  # Apply the full preprocessing pipeline

                # Preprocessed audio comes as frames after segmentation. Flatten frames for feature extraction.
                preprocessed_audio_flat = preprocessed_audio.flatten()

                # Extract MFCC features from the preprocessed audio
                mfccs = librosa.feature.mfcc(y=preprocessed_audio_flat, sr=self.preprocessor.sample_rate, n_mfcc=40)
                mfccs_mean = np.mean(mfccs.T, axis=0)  # Mean of MFCCs

                # Extract Pitch and Pitch Contour
                pitches, magnitudes = librosa.core.piptrack(y=preprocessed_audio_flat, sr=self.preprocessor.sample_rate)
                pitch_mean = np.mean(pitches[pitches > 0])  # Mean pitch
                pitch_contour = pitches[pitches > 0]  # Pitch contour

                # Extract Energy
                energy = np.sum(preprocessed_audio_flat ** 2) / np.float64(len(preprocessed_audio_flat))  # Simple energy calculation

                # Extract Formant Frequencies using raw file (optional, formants usually work on raw audio)
                formant_freqs = self.extract_formants(file_path)

                # Extract Speech Rate (number of syllables per second)
                speech_rate = len(librosa.effects.split(preprocessed_audio_flat)) / librosa.get_duration(y=preprocessed_audio_flat, sr=self.preprocessor.sample_rate)

                # Extract Zero Crossing Rate
                zcr = np.mean(librosa.feature.zero_crossing_rate(y=preprocessed_audio_flat))

                # Append features to the list
                feature_list.append({
                    'Folder Name': row['Folder Name'],
                    'File Name': row['File Name'],
                    'MFCCs': mfccs_mean,
                    'Pitch Mean': pitch_mean,
                    'Pitch Contour': pitch_contour.tolist(),
                    'Energy': energy,
                    'Formant F1': formant_freqs[0],
                    'Formant F2': formant_freqs[1],
                    'Formant F3': formant_freqs[2],
                    'Speech Rate': speech_rate,
                    'Zero Crossing Rate': zcr
                })
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        # Create a DataFrame to store audio features
        self.audio_features_df = pd.DataFrame(feature_list)
        return self.audio_features_df

    def save_audio_data_to_csv(self):
        """Save audio data to separate CSV files based on folder type."""
        # Create the directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Group by folder name and save to separate CSV files
        grouped = self.audio_features_df.groupby('Folder Name')
        for folder_name, group_df in grouped:
            # Define the CSV file path
            csv_file_path = os.path.join(self.output_dir, f'{folder_name}_audio_data.csv')
            
            # Save the DataFrame to CSV
            group_df.to_csv(csv_file_path, index=False)
            print(f"Saved {csv_file_path}")

# Audio Preprocessing Class
class AudioPreprocessing:
    def __init__(self, sample_rate=16000, frame_size=0.025, frame_stride=0.01):
        """
        Initialize the preprocessing parameters.
        :param sample_rate: Target sampling rate for audio (default: 16000 Hz).
        :param frame_size: Length of each frame in seconds (default: 25 ms).
        :param frame_stride: Stride between frames in seconds (default: 10 ms).
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.frame_stride = frame_stride

    def load_audio(self, file_path):
        """
        Load an audio file and return the audio signal and its sampling rate.
        :param file_path: Path to the audio file.
        :return: audio signal, sampling rate
        """
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio, sr

    def noise_removal(self, audio):
        """
        Apply noise reduction using spectral subtraction or filtering.
        :param audio: Input audio signal.
        :return: Denoised audio signal.
        """
        # Basic noise reduction using a high-pass filter
        b, a = signal.butter(1, 100 / (0.5 * self.sample_rate), btype='high')
        audio_denoised = signal.lfilter(b, a, audio)
        return audio_denoised

    def silence_removal(self, audio, top_db=20):
        """
        Remove silence from the audio.
        :param audio: Input audio signal.
        :param top_db: The threshold below reference to consider as silence.
        :return: Audio without silence.
        """
        non_silent_intervals = librosa.effects.split(audio, top_db=top_db)
        audio_nonsilent = np.concatenate([audio[start:end] for start, end in non_silent_intervals])
        return audio_nonsilent

    def normalize(self, audio):
        """
        Normalize the audio signal to have consistent amplitude.
        :param audio: Input audio signal.
        :return: Normalized audio signal.
        """
        return librosa.util.normalize(audio)

    def resample(self, audio, orig_sr, target_sr):
        """
        Resample the audio to a target sampling rate.
        :param audio: Input audio signal.
        :param orig_sr: Original sampling rate of the audio.
        :param target_sr: Target sampling rate.
        :return: Resampled audio signal.
        """
        if orig_sr != target_sr:
            audio_resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
            return audio_resampled
        return audio

    def frame_segmentation(self, audio):
        """
        Segment audio into frames for short-term analysis.
        :param audio: Input audio signal.
        :return: Framed audio signal.
        """
        frame_length = int(self.frame_size * self.sample_rate)
        frame_step = int(self.frame_stride * self.sample_rate)
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=frame_step)
        return frames.T

    def apply_window(self, frames):
        """
        Apply a Hamming window to each frame to reduce spectral leakage.
        :param frames: Input frames of audio signal.
        :return: Windowed frames.
        """
        window = np.hamming(frames.shape[1])
        windowed_frames = frames * window
        return windowed_frames

    def preprocess(self, file_path):
        """
        Full preprocessing pipeline for audio.
        :param file_path: Path to the audio file.
        :return: Preprocessed audio ready for feature extraction.
        """
        # Load audio
        audio, sr = self.load_audio(file_path)
        
        # Resample audio if necessary
        audio = self.resample(audio, sr, self.sample_rate)
        
        # Remove noise
        audio = self.noise_removal(audio)
        
        # Remove silence
        audio = self.silence_removal(audio)
        
        # Normalize the audio
        audio = self.normalize(audio)
        
        # Segment the audio into frames
        frames = self.frame_segmentation(audio)
        
        # Apply windowing to the frames
        windowed_frames = self.apply_window(frames)
        
        return windowed_frames

# Example usage
if __name__ == "__main__":
    # Replace with your ZIP file path, extraction directory, and output directory for CSV files
    zip_file_path = r'C:\Users\Admin\Desktop\Zidio\Speech Emotion Recognition\archive.zip'  # Update to the correct ZIP file path
    extract_to_dir = r'extracted_audio_files'    # Update to your desired extraction directory
    output_dir = r'output_csv_files'             # Update to your desired output directory for CSV files

    # Check if the ZIP file exists
    if not os.path.isfile(zip_file_path):
        print(f"Error: The ZIP file at {zip_file_path} does not exist. Please check the file path.")
    else:
        # Instantiate the DataIngestion class
        data_ingestion = DataIngestion(zip_file_path, extract_to_dir, output_dir)

        # Step 1: Extract audio files and get their paths
        try:
            df_audio_files = data_ingestion.extract_audio_files_from_zip()
            print(df_audio_files)
        except Exception as e:
            print(f"An error occurred while extracting audio files: {e}")

        # Step 2: Extract features from audio files and load them into a DataFrame
        try:
            df_audio_features = data_ingestion.extract_audio_features()
            print(df_audio_features)
        except Exception as e:
            print(f"An error occurred while extracting audio features: {e}")

        # Step 3: Save the audio data to separate CSV files based on folder type
        try:
            data_ingestion.save_audio_data_to_csv()
        except Exception as e:
            print(f"An error occurred while saving audio data to CSV files: {e}")
