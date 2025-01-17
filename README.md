# Depression Prediction

This repository contains a machine learning project to predict depression tendencies based on personal and lifestyle data. The project is divided into two main components: a Streamlit application and a Jupyter Notebook for analysis.

## Files

- **app.py**: Main Streamlit application for collecting user input and providing real-time predictions based on the trained model.
- **relatorio.ipynb**: Jupyter Notebook containing data exploration, preprocessing, and the steps taken to develop and validate the machine learning model.

## Usage

1. **Run the Streamlit Application**:
   - Open a terminal and navigate to the project directory.
   - Run the following command to launch the app:
     ```bash
     streamlit run app.py
     ```
   - Open the provided URL in your browser to interact with the app.

2. **Explore the Analysis**:
   - Open `relatorio.ipynb` in Jupyter Notebook or a compatible editor to review the exploratory data analysis and model development process.

## Requirements

To install the required dependencies, ensure you have Python 3.7 or higher and run:

```bash
pip install -r requirements.txt
```

### Key Libraries Used

- **pandas**: Data manipulation and analysis.
- **numpy**: Numerical operations.
- **seaborn** and **matplotlib**: Data visualization.
- **scikit-learn**: Machine learning tools and preprocessing.
- **catboost** and **xgboost**: Advanced machine learning models.
- **imblearn**: Handling imbalanced datasets with SMOTE.
- **streamlit**: Interactive application framework.
- **plotly**: Interactive visualizations.

## About the Project

This project aims to leverage machine learning to predict the likelihood of depression based on personal attributes such as age, dietary habits, and stress levels. It uses a dataset of approximately 28,000 individuals and incorporates techniques such as feature engineering, SMOTE for handling class imbalance, and scaling for model optimization.
