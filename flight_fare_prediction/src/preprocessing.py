import pandas as pd
import os

def preprocess_data(file_path, output_path):
    # Load the dataset
    df = pd.read_excel("C:\ineuron projects\flight_fare_predict_Project\arline_data.xlsx")

    # Drop rows with missing values
    df = df.dropna(axis=0)

    # Extract day and month from Date_of_Journey
    df["journey_day"] = pd.to_datetime(df['Date_of_Journey'], format="%d/%m/%Y").dt.day
    df["journey_month"] = pd.to_datetime(df['Date_of_Journey'], format="%d/%m/%Y").dt.month
    df.drop(["Date_of_Journey"], axis=1, inplace=True)

    # Extract hour and minute from Dep_Time
    df["dep_hour"] = pd.to_datetime(df['Dep_Time']).dt.hour
    df["dep_min"] = pd.to_datetime(df['Dep_Time']).dt.minute
    df.drop(["Dep_Time"], axis=1, inplace=True)

    # Extract hour and minute from Arrival_Time
    df["arrival_hour"] = pd.to_datetime(df["Arrival_Time"]).dt.hour
    df["arrival_minute"] = pd.to_datetime(df["Arrival_Time"]).dt.minute
    df.drop(["Arrival_Time"], axis=1, inplace=True)

    # Process Duration column
    duration = list(df["Duration"])
    for i in range(len(duration)):
        if len(duration[i].split()) != 2:
            if "h" in duration[i]:
                duration[i] = duration[i].strip() + " 0m"
            else:
                duration[i] = "0h " + duration[i]

    duration_hours = []
    duration_mins = []
    for i in range(len(duration)):
        duration_hours.append(int(duration[i].split(sep="h")[0]))
        duration_mins.append(int(duration[i].split(sep="m")[0].split()[-1]))

    df["Duration_hours"] = duration_hours
    df["Duration_mins"] = duration_mins
    df.drop(["Duration"], axis=1, inplace=True)

    # OneHotEncoding for Airline, Source, and Destination
    Airline = df[["Airline"]]
    Airline['Airline'] = Airline['Airline'].apply(lambda x: x if x in ['Jet Airways', 'IndiGo', 'Air India', 'SpiceJet', 'Multiple carriers', 'GoAir', 'Vistara', 'Air Asia'] else 'Other')
    Airline = pd.get_dummies(Airline, drop_first=True)

    Source = pd.get_dummies(df[["Source"]], drop_first=True)

    df["Destination"] = df["Destination"].replace("New Delhi", "Delhi")
    Destination = pd.get_dummies(df[["Destination"]], drop_first=True)

    # Drop Route and Additional_Info
    df.drop(["Route", "Additional_Info"], axis=1, inplace=True)

    # Label Encoding for Total_Stops
    df.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace=True)

    # Concatenate dataframes
    data_train = pd.concat([df, Airline, Source, Destination], axis=1)
    data_train.drop(["Airline", "Source", "Destination"], axis=1, inplace=True)

    # Save the processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data_train.to_csv("C:\ineuron projects\flight_fare_prediction\data\processed", index=False)
    #print(f"Processed data saved to: {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        #print('Usage: python preprocessing.py <input_file_path> <output_file_path>')
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    preprocess_data(input_file_path, output_file_path)
