import pandas as pd
import os

class BasicDataInspection:
    def __init__(self, csv_folder_path):
        """
        Initialize the BasicDataInspection class with the folder path containing CSV files.
        
        :param csv_folder_path: Folder path containing CSV files to be read.
        """
        self.csv_folder_path = csv_folder_path
        self.data_frame = None

    def merge_csv_files(self):
        """
        Merge all CSV files in the specified folder into a single DataFrame.
        
        :return: Merged DataFrame containing data from all CSV files.
        """
        # List to hold data from each CSV file
        data_frames = []
        
        # Read each CSV file in the folder
        for file in os.listdir(self.csv_folder_path):
            if file.endswith('.csv'):
                file_path = os.path.join(self.csv_folder_path, file)
                try:
                    # Read the CSV file into a DataFrame
                    df = pd.read_csv(file_path)
                    data_frames.append(df)
                    print(f"Successfully read {file}")
                    print(f"Loaded {file} with shape {df.shape}")

                except Exception as e:
                    print(f"Error reading {file}: {e}")
        
        # Concatenate all DataFrames into a single DataFrame
        if data_frames:
            self.data_frame = pd.concat(data_frames, ignore_index=True)
            print(f"Merged DataFrame Shape: {self.data_frame.shape}")
        else:
            print("No CSV files found in the specified directory.")
        
        return self.data_frame
    
    def save_merged_data(self, output_file='Merged_CSV.csv'):
        """
        Save the merged DataFrame to a CSV file.
        
        :param output_file: Name of the output CSV file where the merged data will be saved.
        :return: None
        """
        # Merge the CSV files into a single DataFrame
        merged_df = self.merge_csv_files()
        
        if merged_df is not None:
            # Save the merged DataFrame to a CSV file
            merged_df.to_csv(output_file, index=False)
            print(f"Merged data has been saved to {output_file}")
        else:
            print("No data to save as the merged DataFrame is empty.")

    def perform_basic_inspection(self):
        """
        Perform basic data inspection on the merged DataFrame.
        
        :return: None
        """
        if self.data_frame is not None:
            # Display the first few rows of the DataFrame
            print("\nDataFrame Head:")
            print(self.data_frame.head())

            # Display information about the DataFrame (data types, non-null counts, etc.)
            print("\nDataFrame Info:")
            print(self.data_frame.info())

            # Check for missing values
            print("\nMissing Values:")
            print(self.data_frame.isnull().sum())

            # Display descriptive statistics
            print("\nDescriptive Statistics:")
            print(self.data_frame.describe())
            print("\nDescriptive Statistics:(Categorical)")
            print(self.data_frame.describe(include=["O"]))
        else:
            print("Data frame is empty. Please ensure CSV files were successfully merged.")

# Example usage
if __name__ == "__main__":
    # Replace with your folder path containing CSV files
    csv_folder_path = r'C:\Users\Admin\Desktop\Zidio\Speech Emotion Recognition\output_csv_files'  # Update this path
    
    # Instantiate the BasicDataInspection class
    basic_data_inspection = BasicDataInspection(csv_folder_path)

    # Merge CSV files into one DataFrame
    merged_df = basic_data_inspection.merge_csv_files()
    
    # Perform basic data inspection
    basic_data_inspection.perform_basic_inspection()
    basic_data_inspection.save_merged_data(output_file='Merged_CSV.csv')
