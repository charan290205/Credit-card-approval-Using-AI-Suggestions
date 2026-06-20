# Credit Card Approval Using AI Suggestions

A machine learning and AI integration project that predicts credit card approval based on applicant details, previous credit history, and CIBIL score. The system also provides AI-powered suggestions to improve approval chances.

## Features
- Credit approval prediction using Extra Trees Classifier
- AI-powered suggestions
- Streamlit web application
- Data preprocessing with label encoders

## Technologies Used
- Python
- Streamlit
- Pandas
- Scikit-learn
- Joblib

## Dataset
- German Credit Data

## How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/charan290205/Credit-card-approval-Using-AI-Suggestions.git
cd Credit-card-approval-Using-AI-Suggestions
```

### 2. Install Dependencies
```bash
pip install streamlit pandas joblib scikit-learn
pip install -r requriments.txt
```

### 3. Run the Application
```bash
streamlit run App.py
```

### 4. Open in Browser
```
http://localhost:8501
```

## Project Structure
```
Credit-card-approval-Using-AI-Suggestions/
│
├── App.py
├── Analysis_model.ipynb
├── Extra_trees_credit_model.pkl
├── target_encoder.pkl
├── Sex_encoder.pkl
├── Housing_encoder.pkl
├── Saving accounts_encoder.pkl
├── Checking account_encoder.pkl
├── german_credit_data.csv
├── requriments.txt
├── static/
└── README.md
```

## Author
**Sai Charan Bhonagiri**
