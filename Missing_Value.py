import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Basic_Data_Inspection import BasicDataInspection

class MissingValueAnalysis:
    def __init__(self, data_frame):
        """
        Initialize the MissingValueAnalysis class with the DataFrame to analyze.
        
        :param data_frame: pandas DataFrame for which missing values need to be calculated and visualized.
        """
        self.data_frame = data_frame

    def calculate_missing_values(self):
        """
        Calculate missing values for each column in the DataFrame.
        
        :return: pandas Series containing the count of missing values for each column.
        """
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
