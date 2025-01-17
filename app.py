import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import plotly.express as px

#Data Cleaning and Preprocessing
df = pd.read_csv("Student Depression Dataset.csv")
df.loc[df["Financial Stress"].isnull()] = int(df["Financial Stress"].mean())
df.drop(columns=["id"], axis=1, inplace=True)
group = df.groupby("City")["Depression"].count().reset_index()
valid_cities = group[group.Depression >=5]["City"]
df = df[df["City"].isin(valid_cities)]
df = df.drop(columns=["Job Satisfaction", "Work Pressure", "Profession"], axis=1)

#Data Analiysis with Steamlit
st.sidebar.title("Page Navigation")
option = st.sidebar.selectbox("Choose an option", ["Initial Page", "Main Causes", "Prediction"])

if option == "Initial Page":
    st.title('Depression Analysis and Prediction')
    st.write("""
    ### What is Depression?
    > Depression is more than just feeling sad or having a bad day. It is a complex condition characterized by persistent feelings of sadness, loss of interest in activities, changes in sleep and appetite, and difficulty concentrating. It can range from mild to severe and, if untreated, can lead to serious consequences.

    ### Key Symptoms
    - Persistent sadness or empty mood
    - Loss of interest in previously enjoyed activities
    - Fatigue or lack of energy
    - Difficulty concentrating or making decisions
    - Sleep disturbances (either insomnia or excessive sleep)
    - Changes in appetite or weight

    ### Importance of the Study
    > Depression is a mental health condition that affects millions of people worldwide, impacting their daily lives, relationships, and overall well-being. Understanding depression is crucial for improving diagnosis, providing effective treatment, and reducing the stigma associated with mental health issues.

    ### Global Impact
    - Over 264 million people worldwide suffer from depression (World Health Organization).
    - Depression is the leading cause of disability and the largest contributor to global mental health disorders.
    - It affects people of all ages, backgrounds, and cultures, though it is particularly prevalent among young adults and women.

    ### Objectives of This Project
    **In this application, we aim to:**
    - Analyze data to identify key factors associated with depression.
    - Explore the main causes and trends that contribute to its prevalence.
    - Use machine learning models to predict the likelihood of depression based on individual data.
    """)
    st.markdown("""
        <style>
            h1, h3 {
                text-align: center;
            }
        </style>
    """, unsafe_allow_html=True)


elif option == "Main Causes":
    st.write("""
    # Data Analysis
    > To better understand the factors associated with depression, we analyzed a dataset of almost 28,000 individuals and their responses to various questions related to mental health, lifestyle, and personal experiences.

    Here are some key insights from our analysis:
    """)
    st.divider()

    #Age and Depression
    col1, = st.columns(1)
    counts_age = df.groupby(['Age', 'Depression']).size().reset_index(name='Count')
    
    fig_age = px.histogram(counts_age, x='Age', y='Count', color='Depression',barmode="group",
                     title='Age and Depression', labels={"Age": "Age", "Count": "Count"})
    
    fig_age.update_layout(title={ 'font': {'size': 25}},
                          yaxis=dict(title="Count"))
    
    col1.plotly_chart(fig_age, use_container_width=True)
    
    st.write("""
    - **Age**: We observed a trend indicating that younger individuals are more likely to experience depression.  
    """)
    st.divider()

    #Study Satisfaction
    col2, = st.columns(1)
    group_study = df.groupby(["Study Satisfaction", "Depression"]).size().reset_index(name="Count")
    
    fig_study = px.scatter(
        group_study,x="Study Satisfaction", y="Count", color="Depression", size="Count",
        title="Study Satisfaction and Depression", labels={"Study Satisfaction": "Study Satisfaction", "Count": "Count"})
    
    fig_study.update_layout(title={ 'font': {'size': 25}})
    
    col2.plotly_chart(fig_study, use_container_width=True)
    
    st.write("""
             - Study Satisfaction: **There is a correlation between satisfaction with studies and depression, suggesting that the lower the satisfaction, the greater the likelihood of developing depression.**
                """)
    st.divider()
    
    #Academic Pressure and Depression
    col3, = st.columns(1)
    counts_academic = df.groupby(["Academic Pressure", "Depression"]).size().reset_index(name="Count")
    
    fig_academic = px.line(counts_academic, x="Academic Pressure", y="Count", color="Depression",
                             title= "Academic Pressure and Depression", labels={"Academic Pressure": "Academic Pressure", "Count": "Count"})  
      
    fig_academic.update_layout(title={'font': {'size': 25}})
    
    col3.plotly_chart(fig_academic, use_container_width=True)
    
    st.write("""
             - Academic Pressure: **We noticed a trend that the more academic pressure a person faces, the higher the chance of experiencing depression.**
                """)
    st.divider()
             
    #Dietary Habits
    col4, = st.columns(1)
    counts_habits = df.groupby(["Dietary Habits", "Depression"]).size().reset_index(name="Count")
    
    fig_habits = px.histogram(counts_habits, x="Dietary Habits", y="Count", color="Depression",barmode="group",
                         title="Dietary Habits and Depression",
                         labels={"Dietary Habits": "Dietary Habits", "Count": "Count"})
    
    fig_habits.update_layout(title={ "font": {"size":25}},
                             yaxis= dict(title="Count"))
    
    col4.plotly_chart(fig_habits, use_container_width=True)
    
    st.write("""
             - Dietary Habits: **We noticed that individuals with eating habits classified as "unhealthy" are much more likely to develop depression than those with eating habits considered healthier.**
             """)
    st.divider()
    
    #Financial Stress
    col5, = st.columns(1)
    counts_stress = df.groupby(["Financial Stress", "Depression"]).size().reset_index(name="Count")
    
    fig_stress = px.line(counts_stress, x="Financial Stress", y="Count", color="Depression",
                         title="Financial Stress and Depression", labels={"Financial Stress": "Financial Stress", "Count": "Count"})
    
    fig_stress.update_layout(title={ "font": {"size":25}})
    
    col5.plotly_chart(fig_stress, use_container_width=True)
    
    st.write("""
             - Financial Stress: **People who face high financial stress are much more likely to develop depression, while those with low financial stress are not prone to illness.**
             """)
    st.divider()
    
    # Suicidal Thoughts
    col6, = st.columns(1)
    group_thoughts = df.groupby(["Have you ever had suicidal thoughts ?", "Depression"]).size().reset_index(name="Count")

    fig_thoughts_combined = px.sunburst(group_thoughts, path=["Have you ever had suicidal thoughts ?", "Depression"], 
        values="Count", title="Suicidal Thoughts and Depression", color="Have you ever had suicidal thoughts ?")

    fig_thoughts_combined.update_layout(
        title={ 'font': {'size': 25}},
    )
    
    col6.plotly_chart(fig_thoughts_combined, use_container_width=True)
    
    st.write("""
        - Suicidal Thoughts: **As expected, people who have suicidal thoughts are much more likely to develop depression than those who do not.**
             """)
    
    st.write("""
    <style>
        h1, h2 {
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)
    
else:
    st.write("""# Depression Prediction""")
    st.write("""
        ### Machine Learning Model
        > We developed a machine learning model to predict the likelihood of depression based on individual data. The model uses a dataset of almost 28,000 individuals and their responses to various questions related to mental health, lifestyle, and personal experiences.
        - Answer the following questions to get the prediction.
        """)
    st.divider()
    
    #Feature Engineering
    #Label Encoding
    le = LabelEncoder()
    label_encoder = df.select_dtypes(include='object').columns
    for i in label_encoder:
        df[i] = le.fit_transform(df[i])
    # Smote
    smote = SMOTE()
    x = df.drop(columns=["Depression"],axis=1)
    y = df["Depression"]
    X_resampled, y_resampled = smote.fit_resample(x,y)
    df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

    # Splitting the data
    x = df_resampled.drop(columns=["Depression", "Family History of Mental Illness", "Sleep Duration", "Gender",
                                "City", "Degree"],axis=1)
    y = df_resampled["Depression"]
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)

    #Standard Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(x_train)
    X_test = scaler.transform(x_test)

    #Model
    catboost_model = CatBoostClassifier(task_type='CPU', eval_metric='Accuracy', iterations= 200, learning_rate =  0.1, 
                                        max_depth= 3, random_state= 1, verbose=0).fit(X_train, y_train)
        
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col5, col6 = st.columns(2)
    col7, col8 = st.columns(2)
    
    with col1:
        age = st.slider("Enter your Age", min_value=0, max_value=100, value=0, step=1)
        
    with col2:
        dietary = st.selectbox("Select your Dietary Habits", 
                            ["Healthy", "Moderate", "Unhealthy", "Others"])
            
    with col3:
        study = st.selectbox("Select your Study Satisfaction", range(0,6))
    
    with col4:
        cgpa = st.slider("Enter your CGPA (Cumulative Grade Point Average)", 
                        min_value=1.0, max_value=10.0, value=1.0, step=0.1)
    
    with col5:
        work_study = st.slider("How many hours do you work or study per day?", min_value=0.0, 
        max_value=24.0, value=8.0, step=0.5)
        
    with col6:
        stress = st.selectbox("Report your level of financial stress", range(1,6))
        
    with col7:
        academic = st.selectbox("Rate your academic pressure", range(0,6))
    
    with col8:
        thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["No", "Yes"])

    st.markdown("""
    <style>
        h1, h3 {
            text-align: center;
        }
    </style>
        """, unsafe_allow_html=True)
    st.divider()
    
    patient = np.array([
        [age,
         0 if dietary == "Healthy" else (1 if dietary == "Moderate" else (3 if dietary == "Unhealthy" else 4)),
         study,
         cgpa,
         work_study,
         stress,
         academic,
         1 if thoughts == "Yes" else 0]
    ])
    
    if st.button('Predict Depression Tendency'):

        patient_scaled = scaler.transform(patient)
        result = catboost_model.predict(patient_scaled)
        
        if result == 1:
            st.markdown(
                "<h3 style='color: red;'>The individual HAS tendencies towards Depression</h3>", 
                unsafe_allow_html=True
        )
        else:
            st.markdown(
                "<h3 style='color: green;'>The individual DOES NOT HAVE tendencies towards Depression</h3>", 
                unsafe_allow_html=True
        )