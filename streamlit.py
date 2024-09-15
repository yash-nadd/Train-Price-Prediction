import pickle
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Flight Price Predictor",
)

st.title("Flight Price Predictor")
st.write("\n")

st.markdown(
    """
Predicting flight prices can help you plan your trips better by finding the best deals. This tool uses machine learning to predict the price of a flight based on several factors such as airline, source city, departure time, number of stops, etc.
    """
)

# Load the model and label encoders
with open('model1.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('label_encoders.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)

col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox(label="Airline", options=['AirAsia', 'Air_India', 'GO_FIRST', 'Indigo', 'SpiceJet', 'Vistara'])
    source_city = st.selectbox(label="Source City", options=['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai'])
    departure_time = st.selectbox(label="Departure Time", options=['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night'])
    arrival_time = st.selectbox(label="Arrival Time", options=['Morning', 'Night'])

with col2:
    destination_city = st.selectbox(label="Destination City", options=['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai'])
    flight_class = st.selectbox(label="Class", options=['Business', 'Economy'])
    duration = st.number_input(label="Duration (in hours)", min_value=0.0, max_value=24.0, step=0.1)


# Function to make prediction
def prediction(airline, source_city, departure_time,  arrival_time, destination_city, flight_class, duration):
    # Encode input data
    input_data = {
        'airline': label_encoders['airline'].transform([airline])[0],
        'source_city': label_encoders['source_city'].transform([source_city])[0],
        'departure_time': label_encoders['departure_time'].transform([departure_time])[0],
        'arrival_time': label_encoders['arrival_time'].transform([arrival_time])[0],
        'destination_city': label_encoders['destination_city'].transform([destination_city])[0],
        'class': label_encoders['class'].transform([flight_class])[0],
        'duration': duration,
    }

    df_input = pd.DataFrame([input_data])
    prediction = model.predict(df_input)
    return prediction

if st.button('Predict'):
    predict = prediction(airline, source_city, departure_time,arrival_time, destination_city, flight_class, duration)
    st.success(f"Predicted Flight Price: ₹{predict[0]:.2f}")
