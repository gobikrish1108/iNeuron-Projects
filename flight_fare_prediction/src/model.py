import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
import matplotlib.pyplot as plt

def train_model(data_path):
    df = pd.read_csv("C:\ineuron projects\flight_fare_predict_Project\arline_data.xlsx")

    # Assuming 'Price' is the target variable
    X = df.drop(columns=['Price'])
    y = df['Price']

    # Train the model
    model = ExtraTreesRegressor()
    model.fit(X, y)

    # Feature importance
    feature_importances = model.feature_importances_
    features = X.columns
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importances
    plt.figure(figsize=(10,5))
    plt.title('Feature Importances')
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.show()

    return model

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        #print('Usage: python model.py <data_file_path>')
        sys.exit(1)

    data_file_path = sys.argv[1]
    model = train_model(data_file_path)
