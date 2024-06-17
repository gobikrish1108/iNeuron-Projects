import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def eda(data_path):
    df = pd.read_csv("C:\ineuron projects\flight_fare_predict_Project\arline_data.xlsx")

    # Display basic information
    print(df.head())
    print(df.shape)
    print(df.info())
    print(df.describe())

    # Visualization examples
    plt.figure(figsize=(10,5))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()

    plt.figure(figsize=(10,5))
    sns.countplot(y='Airline', data=df, order=df['Airline'].value_counts().index)
    plt.title('Airline Count Plot')
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        #print('Usage: python eda.py <data_file_path>')
        sys.exit(1)

    data_file_path = sys.argv[1]
    eda(data_file_path)
