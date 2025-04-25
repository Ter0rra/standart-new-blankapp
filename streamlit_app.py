# Notebook version sans Tkinter, avec widgets interactifs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from ipywidgets import interact, widgets
from IPython.display import display

csv_url = "https://raw.githubusercontent.com/Ter0rra/blank-app/6813973fbe231ac40da9129ab94dca649ee09702/student_habits_performance.csv"
df_student = pd.read_csv(csv_url)
df_student = df_student.drop(['student_id'], axis=1)
df_student['media_hours'] = df_student['netflix_hours'] + df_student['social_media_hours']
df_student = df_student.reindex(['age', 'gender', 'study_hours_per_day', 'social_media_hours','netflix_hours', 'media_hours','part_time_job','attendance_percentage','sleep_hours','diet_quality','exercise_frequency','parental_education_level','internet_quality','mental_health_rating','extracurricular_participation','exam_score'], axis=1)
df_num_value = df_student.select_dtypes(include=['number'])
df_student_ml = df_num_value.drop(['age', 'social_media_hours', 'netflix_hours'], axis=1)

target = 'exam_score'
X = df_student_ml.drop(columns=[target])
y = df_student_ml[target]

numeric_features = X.select_dtypes(include=np.number).columns.to_list()
categorical_features = X.select_dtypes(include='object').columns.to_list()

# Préprocessing
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

X_processed = preprocessor.fit_transform(X)

model = LinearRegression()
model.fit(X_processed, y)

def get_grade_color(score):
    if score >= 90: return 'A', 'green'
    elif score >= 80: return 'B', 'lightgreen'
    elif score >= 70: return 'C', 'yellow'
    elif score >= 60: return 'D', 'orange'
    elif score >= 50: return 'E', 'orangered'
    else: return 'F', 'red'

def get_dynamic_suggestions(df_input):
    current_score = model.predict(preprocessor.transform(df_input))[0]
    improvements = []
    for col in numeric_features:
        better_value = df_input[col].values[0]
        test_up = df_input.copy()
        test_up[col] = better_value + 1
        score_up = model.predict(preprocessor.transform(test_up))[0]

        test_down = df_input.copy()
        test_down[col] = better_value - 1
        score_down = model.predict(preprocessor.transform(test_down))[0]

        score_up = min(score_up, 100)
        score_down = min(score_down, 100)

        delta_up = score_up - current_score
        delta_down = score_down - current_score

        best_delta = max(delta_up, delta_down)
        if best_delta > 0:
            direction = "augmenter" if delta_up > delta_down else "réduire"
            improvements.append((col, direction, best_delta))

    improvements.sort(key=lambda x: x[2], reverse=True)
    return improvements[:3]

def predict_score(**kwargs):
    input_df = pd.DataFrame([kwargs])
    X_input = preprocessor.transform(input_df)
    score = model.predict(X_input)[0]
    score = min(score, 100)
    grade, color = get_grade_color(score)

    plt.figure(figsize=(6, 1.5))
    plt.barh(["Score"], [score], color=color)
    plt.xlim(0, 100)
    plt.title(f"Score: {score:.2f} - Note: {grade}")
    plt.show()

    if score >= 90:
        print("🎉 Félicitations ! Continue comme ça, c'est parfait ! 🎉")
    else:
        suggestions = get_dynamic_suggestions(input_df)
        print("Suggestions pour améliorer votre score :")
        for var, direction, gain in suggestions:
            print(f"- {direction} {var} → gain estimé de {gain:.2f} points")

# Génération dynamique des widgets
widget_dict = {}
for col in X.columns:
    if col in numeric_features:
        min_val = float(df_student_ml[col].min())
        max_val = float(df_student_ml[col].max())
        mean_val = float(df_student_ml[col].mean())
        widget_dict[col] = widgets.FloatSlider(value=mean_val, min=min_val, max=max_val, step=0.1, description=col)
    else:
        options = df_student_ml[col].dropna().unique().tolist()
        widget_dict[col] = widgets.Dropdown(options=options, description=col)

interact(predict_score, **widget_dict)
