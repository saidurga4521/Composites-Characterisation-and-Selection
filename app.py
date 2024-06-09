from flask import Flask, render_template, request
import joblib
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# Define the FrequencyEncoder class
class FrequencyEncoder:
    def __init__(self):
        self.frequency_map = {}

    def fit(self, X, y=None):
        for column in X.columns:
            self.frequency_map[column] = X[column].value_counts(normalize=True).to_dict()
        return self

    def transform(self, X):
        X_encoded = X.copy()
        for column in X.columns:
            X_encoded[column] = X_encoded[column].map(self.frequency_map[column])
            # Replace NaNs from mapping with 0 or another default value
            X_encoded[column].fillna(0, inplace=True)
        return X_encoded

# Load the pre-trained model, scaler, and encoder
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

composite_name = {
    5: 'Aluminum-Silicon Carbide', 1: 'Aluminum-Boron Carbide',
    0: 'Aluminum-Alumina', 2: 'Aluminum-Graphite', 7: 'Aluminum-Zirconium',
    4: 'Aluminum-Silicon', 6: 'Aluminum-Titanium Carbide',
    3: 'Aluminum-Magnesium ', 16: 'Nickel-Aluminum', 22: 'Nickel-Silicon Carbide',
    20: 'Nickel-Graphite', 17: 'Nickel-Boron', 24: 'Nickel-Tungsten',
    19: 'Nickel-Copper', 23: 'Nickel-Titanium', 21: 'Nickel-Molybdenum',
    25: 'Nickel-Zirconium', 18: 'Nickel-Cobalt', 14: 'Magnesium-Silicon Carbide',
    8: 'Magnesium-Alumina', 10: 'Magnesium-Boron ', 9: 'Magnesium-Aluminum',
    15: 'Magnesium-Zirconium', 12: 'Magnesium-Graphene',
    13: 'Magnesium-Nano Particles', 11: 'Magnesium-Carbon Nanotubes',
    34: 'Titanium-Carbon', 44: 'Titanium-Silicon', 31: 'Titanium-Alumina',
    38: 'Titanium-Graphite', 33: 'Titanium-Boron', 32: 'Titanium-Aluminum',
    48: 'Titanium-Zirconium', 42: 'Titanium-Nickel', 37: 'Titanium-Copper',
    43: 'Titanium-Niobium', 40: 'Titanium-Molybdenum', 47: 'Titanium-Vanadium',
    36: 'Titanium-Chromium', 46: 'Titanium-Tantalum', 39: 'Titanium-Magnesium',
    49: 'Titanium-Zirconium-Nickel', 45: 'Titanium-Silicon ',
    35: 'Titanium-Carbon Nanotubes ', 41: 'Titanium-Nano Particles',
    26: 'Palladium-Aluminum', 29: 'Palladium-Silicon Carbide',
    27: 'Palladium-Carbon', 30: 'Palladium-Titanium', 28: 'Palladium-Nickel'
}

corrosive_environment = {0: 'High', 1: 'Medium'}
cost_effectiveness = {0: 'High', 1: 'Low', 2: 'Medium'}
strength_to_weight_ratio = {0: 'High', 1: 'Medium'}
corrosion_resistance = {0: 'High', 1: 'Medium'}
fatigue_resistance = {0: 'High', 1: 'Medium'}


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        # Retrieve form data
        data = {
            'density': float(request.form['density']),
            'hardness': float(request.form['hardness']),
            'primary_composition': float(request.form['primary_composition']),
            'secondary_composition': float(request.form['secondary_composition']),
            'tertiary_composition': float(request.form['tertiary_composition']),
            'youngs_modulus': float(request.form['youngs_modulus']),
            'yield_strength': float(request.form['yield_strength']),
            'tensile_strength': float(request.form['tensile_strength']),
            'elasticity_modulus': float(request.form['elasticity_modulus']),
            'thermal_conductivity': float(request.form['thermal_conductivity']),
            'thermal_expansion': float(request.form['thermal_expansion']),
            'specific_heat_capacity': float(request.form['specific_heat_capacity']),
            'operating_temperature': float(request.form['operating_temperature']),
            'primary_component': request.form['primary_component'],
            'formability': request.form['formability'],
            'durability': request.form['durability'],
            'recyclability': request.form['recyclability']
        }

        # Convert the form data to a DataFrame
        data_df = pd.DataFrame([data])

        # Debugging: Print the input data
        print("Input data:")
        print(data_df)

        # Preprocess input data
        processed_data = preprocess_input(data_df)

        # Debugging: Check for NaNs in the processed data
        if processed_data.isnull().values.any():
            print("Processed data contains NaNs:")
            print(processed_data)
            return "Error: Processed data contains NaNs. Please check input values."

        # Make predictions using the model
        prediction = model.predict(processed_data)
        print("Prediction:", prediction)

        # Map the prediction to the descriptive values
        composite_prediction = composite_name[prediction[0][0]]
        corrosive_env_prediction = corrosive_environment[prediction[0][1]]
        cost_effectiveness_prediction = cost_effectiveness[prediction[0][2]]
        strength_to_weight_prediction = strength_to_weight_ratio[prediction[0][3]]
        corrosion_resistance_prediction = corrosion_resistance[prediction[0][4]]
        fatigue_resistance_prediction = fatigue_resistance[prediction[0][5]]

        # Render the composite prediction HTML page
        composite_html = render_template('composite.html', composite=composite_prediction)
        # Render the properties prediction HTML page
        properties_html = render_template('properties.html', corrosive_env=corrosive_env_prediction,
                                           cost_effectiveness=cost_effectiveness_prediction,
                                           strength_to_weight=strength_to_weight_prediction,
                                           corrosion_resistance=corrosion_resistance_prediction,
                                           fatigue_resistance=fatigue_resistance_prediction)
        # Combine both HTML pages
        full_html = composite_html + properties_html

        return full_html



# Function to preprocess input data
def preprocess_input(data):
    # Define original feature names used during fitting
    original_feature_names = [
        'DENSITY', 'HARDNESS', 'PRIMARY CHEMICAL COMPOSITION',
        'SECONDARY CHEMICAL COMPOSITION', 'TERTIARY  CHEMICALCOMPOSITION',
        'YOUNGS MODULUS', 'YIELD STRENGTH', 'TENSILE STRENGTH',
        'MODULUS OF ELASTISITY', 'THERMAL CONDUCTIVITY',
        'THERMAL EXPANSION', 'SPECIFIC HEAT CAPACITY',
        'OPERATING TEMPERATURE', 'PRIMARY CHEMICAL COMPONENT',
        'FORMABILITY', 'DURABILITY', 'RECYCLABILITY'
    ]

    # Create a mapping from form field names to original feature names
    feature_name_mapping = {
        'density': 'DENSITY',
        'hardness': 'HARDNESS',
        'primary_composition': 'PRIMARY CHEMICAL COMPOSITION',
        'secondary_composition': 'SECONDARY CHEMICAL COMPOSITION',
        'tertiary_composition': 'TERTIARY  CHEMICALCOMPOSITION',
        'youngs_modulus': 'YOUNGS MODULUS',
        'yield_strength': 'YIELD STRENGTH',
        'tensile_strength': 'TENSILE STRENGTH',
        'elasticity_modulus': 'MODULUS OF ELASTISITY',
        'thermal_conductivity': 'THERMAL CONDUCTIVITY',
        'thermal_expansion': 'THERMAL EXPANSION',
        'specific_heat_capacity': 'SPECIFIC HEAT CAPACITY',
        'operating_temperature': 'OPERATING TEMPERATURE',
        'primary_component': 'PRIMARY CHEMICAL COMPONENT',
        'formability': 'FORMABILITY',
        'durability': 'DURABILITY',
        'recyclability': 'RECYCLABILITY'
    }

    # Rename columns to match the original feature names
    data = data.rename(columns=feature_name_mapping)

    # Ensure the order of columns is the same as original feature names
    data = data[original_feature_names]

    # Define numerical and categorical features with their original names
    numerical_features = [
        'DENSITY', 'HARDNESS', 'PRIMARY CHEMICAL COMPOSITION', 'SECONDARY CHEMICAL COMPOSITION',
        'TERTIARY  CHEMICALCOMPOSITION', 'YOUNGS MODULUS', 'YIELD STRENGTH',
        'TENSILE STRENGTH', 'MODULUS OF ELASTISITY', 'THERMAL CONDUCTIVITY',
        'THERMAL EXPANSION', 'SPECIFIC HEAT CAPACITY', 'OPERATING TEMPERATURE'
    ]
    categorical_features = [
        'PRIMARY CHEMICAL COMPONENT', 'FORMABILITY', 'DURABILITY', 'RECYCLABILITY'
    ]

    # Debugging: Print feature names
    print("Expected numerical features:", numerical_features)
    print("Data numerical features:", data[numerical_features].columns.tolist())

    # Debugging: Print the data before imputation
    print("Data before imputation:")
    print(data)

    # Impute missing values for numerical features
    numerical_imputer = SimpleImputer(strategy='mean')
    data[numerical_features] = numerical_imputer.fit_transform(data[numerical_features])

    # Impute missing values for categorical features
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    data[categorical_features] = categorical_imputer.fit_transform(data[categorical_features])

    # Debugging: Print the data after imputation
    print("Data after imputation:")
    print(data)

    # Check for NaNs after imputation
    if pd.isnull(data).any().any():
        print("Data contains NaNs after imputation:")
        print(data)
        return pd.DataFrame()  # Return empty DataFrame to signify error

    # Scale numerical features
    data[numerical_features] = scaler.transform(data[numerical_features])

    # Encode categorical features
    data[categorical_features] = encoder.transform(data[categorical_features])

    # Debugging: Print the data after scaling and encoding
    print("Data after scaling and encoding:")
    print(data)

    return pd.DataFrame(data, columns=original_feature_names)

if __name__ == '__main__':
    app.run(debug=True)


