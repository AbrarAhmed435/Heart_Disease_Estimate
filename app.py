import pickle
import numpy as np 
from sklearn.preprocessing import StandardScaler
from flask import Flask,request,render_template

app = Flask(__name__)

with open('scale.pkl', 'rb') as f:
    scale = pickle.load(f)
dtc=pickle.load(open('dtc.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form (assuming 8 features for the diabetes model)
        features = [float(request.form.get(f'feature{i}')) for i in range(1, 14)]
        input_data = np.array(features).reshape(1, -1)  # Reshape for a single sample

        # Apply the same scaling transformation that was used during training
        input_data_scaled = scale.transform(input_data)
        
        predicted_value=dtc.predict(input_data_scaled).reshape(1,-1)
        
        

        # Extract the predicted class (for binary classification)
        
        if predicted_value==1:
            prediction_text="Positive"
        else:
            prediction_text="Negative"
        # Return the prediction result
    except Exception as e:
        prediction_text = f"Error: {str(e)}"  # Handle error and display it

    # Return the prediction result
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)

