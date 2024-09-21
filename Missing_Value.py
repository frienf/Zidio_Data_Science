import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class BasicDataInspection:
    def __init__(self, csv_folder_path):
        self.csv_folder_path = csv_folder_path
        self.data_frame = None

    def merge_csv_files(self):
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
    
class MissingValueAnalysis:
    def __init__(self, data_frame):
        self.data_frame = data_frame

    def calculate_missing_values(self):
        missing_values = self.data_frame.isnull().sum()
        print("\nMissing Values in Each Column:")
        print(missing_values)
        return missing_values

    def visualize_missing_values(self):
        """
        Create a bar chart to visualize the missing values in the DataFrame.
        
        :return: None
        """
        missing_values = self.calculate_missing_values()
        
        # Filter out columns with no missing values
        missing_values = missing_values[missing_values > 0]
        
        if not missing_values.empty:
            # Plot the missing values
            plt.figure(figsize=(12, 6))
            sns.barplot(x=missing_values.index, y=missing_values.values, palette='viridis')
            sns.heatmap(self.data_frame.isnull(), cbar=False, cmap='viridis', yticklabels=False)
            plt.title('Missing Values per Column')
            plt.xlabel('Columns')
            plt.ylabel('Number of Missing Values')
            plt.xticks(rotation=45)
            plt.show()
        else:
            print("No missing values to visualize.")

# Example usage
if __name__ == "__main__":
    # Example to demonstrate how to use the MissingValueAnalysis class
    # Replace this with the actual DataFrame loading process
    df_example = BasicDataInspection(r'C:\Users\Admin\Desktop\Zidio\Speech Emotion Recognition\output_csv_files')
    merged_df = df_example.merge_csv_files() 
         # Instantiate the MissingValueAnalysis class
    missing_value_analysis = MissingValueAnalysis(merged_df)

    # Calculate missing values
    missing_value_analysis.calculate_missing_values()

    # Visualize missing values
    missing_value_analysis.visualize_missing_values()
