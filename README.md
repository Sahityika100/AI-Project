# AI Disease Predictor Project
This is a machine learning web app built with Streamlit that predicts whether a person is diabetic based on input medical data. It uses a Random Forest Classifier and displays results with visual insights using matplotlib.

Features
1]Predicts diabetes using a trained ML model
2]User-friendly Streamlit interface
3]visualizations with matplotlib
4]Model serialized using pickle

Installation

1. Clone the repository:
git clone https://github.com/Sahityika100/AI-Project.git
2. Install dependencies:
pip install -r requirements.txt
3. Run the app:
streamlit run Disease_Predict.py

Input Fields
Pregnancies
Glucose
Blood Pressure
Skin Thickness
Insulin
BMI
Diabetes Pedigree Function
Age

Output
Prediction: Diabetic or Not Diabetic
Probability Score: Confidence level of prediction
Visualization: Feature importance or data distribution (if implemented)

Technologies Used
Python
Streamlit
Pandas, NumPy
Scikit-learn
Matplotlib
Pickle
