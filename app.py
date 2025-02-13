import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained models, label encoder, and scaler
model_score = joblib.load("student_score_model.pkl")
model_grade = joblib.load("student_grade_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Generate random data for student grades (same as in model_training.ipynb)
np.random.seed(42)
num_students = 100
courses = [f"500{i}" for i in range(8)]
student_ids = [f"S{i+1:03d}" for i in range(num_students)]

# Random scores (0-100)
scores = np.random.randint(50, 100, size=(num_students, len(courses)))

# Map scores to grades
def map_score_to_grade(score):
    if score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 50:
        return "D"
    else:
        return "F"

# Create a DataFrame
data = pd.DataFrame(scores, columns=courses, index=student_ids)

# Generate grades for each student
data["Grades"] = [list(map(map_score_to_grade, row)) for row in data[courses].values]

# Add Final Score and Final Grade
data["Final Score"] = np.mean(data[courses].values, axis=1).astype(int)  # Average of all course scores
data["Final Grade"] = data["Final Score"].apply(map_score_to_grade)

# Streamlit UI
st.title("Student Performance Prediction and Assessment")
st.write("Select a student ID to view and predict performance.")

# Dropdown for student ID
student_id = st.selectbox("Select Student ID", student_ids)

# Display current grades, scores, final score, and final grade in one table
if st.button("Get Current Performance"):
    st.write(f"Student ID: {student_id}")
    
    # Create a table for grades, scores, final score, and final grade
    performance_table = pd.DataFrame({
        "Course": courses + [" Next 5008"],
        "Score": list(data.loc[student_id, courses]) + [data.loc[student_id, "Final Score"]],
        "Grade": list(data.loc[student_id, "Grades"]) + [data.loc[student_id, "Final Grade"]]
    })
    
    st.write("Performance Table:")
    st.table(performance_table)

# Predict final score and grade
if st.button("Predict Next Score and Grade"):
    # Prepare student data for prediction
    student_data = data.loc[student_id, courses].values.reshape(1, -1)
    
    # Scale the student data
    student_data_scaled = scaler.transform(student_data)
    
    # Predict final score
    predicted_final_score = model_score.predict(student_data_scaled)[0]
    
    # Predict final grade
    predicted_final_grade_encoded = model_grade.predict(student_data_scaled)
    predicted_final_grade = label_encoder.inverse_transform(predicted_final_grade_encoded)[0]

    st.write(f"Predicted Next Score: {predicted_final_score}")
    st.write(f"Predicted Next Grade: {predicted_final_grade}")



# Display the dataset
if st.checkbox("Show Dataset"):
    st.write(data)
