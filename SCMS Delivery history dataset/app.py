from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the model
model = pickle.load(open("model_01.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input values from the form
        country = request.form['country']
        shipment_mode = request.form['shipment_mode']
        product_group = request.form['product_group']
        sub_classification = request.form['sub_classification']
        brand = request.form['brand']
        unit_of_measure = int(request.form['unit_of_measure'])
        line_item_quantity = int(request.form['line_item_quantity'])
        line_item_value = float(request.form['line_item_value'])

        # Create input array with one-hot encoding
        input_array = []

        # One-hot encode country
        country_dict = {'South Africa': 0, 'Nigeria': 1, 'Côte d\'Ivoire': 2, 'Uganda': 3, 'Vietnam': 4, 'Zambia': 5,
                        'Haiti': 6, 'Mozambique': 7, 'Zimbabwe': 8, 'Tanzania': 9, 'Others': 10}
        country_encoded = [0] * len(country_dict)
        if country in country_dict:
            country_encoded[country_dict[country]] = 1
        input_array.extend(country_encoded)

        # One-hot encode shipment mode
        shipment_mode_dict = {'Air': 0, 'Truck': 1, 'Air Charter': 2, 'Ocean': 3}
        shipment_mode_encoded = [0] * len(shipment_mode_dict)
        if shipment_mode in shipment_mode_dict:
            shipment_mode_encoded[shipment_mode_dict[shipment_mode]] = 1
        input_array.extend(shipment_mode_encoded)

        # One-hot encode product group
        product_group_dict = {'ARV': 0, 'HRDT': 1, 'ANTM': 2, 'ACT': 3, 'MRDT': 4}
        product_group_encoded = [0] * len(product_group_dict)
        if product_group in product_group_dict:
            product_group_encoded[product_group_dict[product_group]] = 1
        input_array.extend(product_group_encoded)

        # One-hot encode sub classification
        sub_classification_dict = {'Adult': 0, 'Pediatric': 1, 'HIV test': 2, 'HIV test - Ancillary': 3, 'Malaria': 4, 'ACT': 5}
        sub_classification_encoded = [0] * len(sub_classification_dict)
        if sub_classification in sub_classification_dict:
            sub_classification_encoded[sub_classification_dict[sub_classification]] = 1
        input_array.extend(sub_classification_encoded)

        # One-hot encode brand
        brand_dict = {'Generic': 0, 'Others': 1, 'Determine': 2}
        brand_encoded = [0] * len(brand_dict)
        if brand in brand_dict:
            brand_encoded[brand_dict[brand]] = 1
        input_array.extend(brand_encoded)

        # Add numerical features
        input_array.extend([unit_of_measure, line_item_quantity, line_item_value])

        # Predict the pack price
        prediction = model.predict([input_array])
        output = round(prediction[0], 2)

        # Display the result
        result = f"The predicted pack price is: {output}"
        return render_template('index.html', result=result)

    except Exception as e:
        return render_template('index.html', result=f"Error occurred: {e}")

if __name__ == "__main__":
    app.run(debug=True)
