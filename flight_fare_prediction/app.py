from flask import Flask, request, render_template
from flask_cors import cross_origin
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        try:
            # Extracting data from form
            journey_date = request.form["Dep_Time"]
            dep_time = request.form["Dep_Time"]
            arrival_time = request.form["Arrival_Time"]
            total_stops = int(request.form["stops"])
            airline = request.form["airline"]
            source = request.form["Source"]
            destination = request.form["Destination"]

            # Feature Engineering
            journey_day = int(pd.to_datetime(journey_date).day)
            journey_month = int(pd.to_datetime(journey_date).month)
            dep_hour = int(pd.to_datetime(dep_time).hour)
            dep_minute = int(pd.to_datetime(dep_time).minute)
            arrival_hour = int(pd.to_datetime(arrival_time).hour)
            arrival_minute = int(pd.to_datetime(arrival_time).minute)
            total_duration_minutes = (arrival_hour * 60 + arrival_minute) - (dep_hour * 60 + dep_minute)
            if total_duration_minutes < 0:
                total_duration_minutes += 24 * 60

            # One-hot encoding dictionaries
            airline_dict = {'Jet Airways': 0, 'IndiGo': 1, 'Air India': 2, 'Multiple carriers': 3,
                            'SpiceJet': 4, 'Vistara': 5, 'GoAir': 6, 'others': 7}
            source_dict = {'Delhi': 0, 'Kolkata': 1, 'Mumbai': 2, 'Chennai': 3,
                           'Bangalore': 4, 'Hyderabad': 5, 'Cochin': 6}
            destination_dict = {'Cochin': 0, 'Delhi': 1, 'Hyderabad': 2, 'Kolkata': 3,
                                'Bangalore': 4, 'Mumbai': 5, 'Chennai': 6}

            # Creating input array
            input_data = {
                'Total_Stops': total_stops,
                'Journey_Day': journey_day,
                'Journey_Month': journey_month,
                'Dep_Hour': dep_hour,
                'Dep_Minute': dep_minute,
                'Arrival_Hour': arrival_hour,
                'Arrival_Minute': arrival_minute,
                'Total_Duration_minutes': total_duration_minutes,
                'Airline': airline_dict[airline],
                'Source': source_dict[source],
                'Destination': destination_dict[destination]
            }

            columns = [
                'Total_Stops', 'Journey_Day', 'Journey_Month', 'Dep_Hour', 'Dep_Minute',
                'Arrival_Hour', 'Arrival_Minute', 'Total_Duration_minutes',
                'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
                'Airline_Jet Airways', 'Airline_Multiple carriers', 'Airline_SpiceJet',
                'Airline_Vistara', 'Airline_others', 'Source_Chennai', 'Source_Delhi',
                'Source_Kolkata', 'Source_Mumbai', 'Destination_Cochin',
                'Destination_Delhi', 'Destination_Hyderabad', 'Destination_Kolkata',
                'Outlier_Yes', 'Journey_Day_of_Week_Monday',
                'Journey_Day_of_Week_Saturday', 'Journey_Day_of_Week_Sunday',
                'Journey_Day_of_Week_Thursday', 'Journey_Day_of_Week_Tuesday',
                'Journey_Day_of_Week_Wednesday'
            ]

            input_array = np.zeros(len(columns))
            for i, col in enumerate(columns):
                if col.startswith('Airline_'):
                    airline_col = col.split('_')[1]
                    if airline_col == airline:
                        input_array[i] = 1
                elif col.startswith('Source_'):
                    source_col = col.split('_')[1]
                    if source_col == source:
                        input_array[i] = 1
                elif col.startswith('Destination_'):
                    destination_col = col.split('_')[1]
                    if destination_col == destination:
                        input_array[i] = 1
                else:
                    if col in input_data:
                        input_array[i] = input_data[col]

            # Predict the price
            prediction = model.predict([input_array])
            output = round(prediction[0], 2)

            return render_template('home.html', prediction_text=f"Your Flight price is Rs. {output}")

        except Exception as e:
            return render_template('home.html', prediction_text=f"Error occurred: {e}")

    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
