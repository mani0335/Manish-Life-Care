from flask import Flask, request, render_template, send_from_directory
import numpy as np
import pandas as pd
import pickle
import os
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Models
brain_model = load_model('models/model.h5')
svc = pickle.load(open('models/.ipynb_checkpoints/svc.pkl', 'rb'))

# Class labels for brain tumors
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Load datasets
sym_des = pd.read_csv("datasets/.ipynb_checkpoints/symtoms_df.csv")
precautions = pd.read_csv("datasets/.ipynb_checkpoints/precautions_df.csv")
workout = pd.read_csv("datasets/.ipynb_checkpoints/workout_df.csv")
description = pd.read_csv("datasets/.ipynb_checkpoints/description.csv")
medications = pd.read_csv("datasets/.ipynb_checkpoints/medications.csv")
diets = pd.read_csv("datasets/.ipynb_checkpoints/diets.csv")

# Symptom Dictionary (132 total)
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3,
    'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8,
    'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12,
    'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16,
    'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20,
    'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24,
    'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28,
    'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
    'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36,
    'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40,
    'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44,
    'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47,
    'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51,
    'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55,
    'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59,
    'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63,
    'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68,
    'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71,
    'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74,
    'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77,
    'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81,
    'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84,
    'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87,
    'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90,
    'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93,
    'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96,
    'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99,
    'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic_patches': 102,
    'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105,
    'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108,
    'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111,
    'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114,
    'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116,
    'blood_in_sputum': 117, 'prominent_veins_on_calf': 118, 'palpitations': 119,
    'painful_walking': 120, 'pus_filled_pimples': 121, 'blackheads': 122, 'scurring': 123,
    'skin_peeling': 124, 'silver_like_dusting': 125, 'small_dents_in_nails': 126,
    'inflammatory_nails': 127, 'blister': 128, 'red_sore_around_nose': 129,
    'yellow_crust_ooze': 130, 'prognosis': 131
}

# Disease Index Mapping
diseases_list = {
    0: "Fungal infection", 1: "Allergy", 2: "GERD", 3: "Chronic cholestasis",
    4: "Drug Reaction", 5: "Peptic ulcer diseae", 6: "AIDS", 7: "Diabetes",
    8: "Gastroenteritis", 9: "Bronchial Asthma", 10: "Hypertension", 11: "Migraine",
    12: "Cervical spondylosis", 13: "Paralysis (brain hemorrhage)", 14: "Jaundice",
    15: "Malaria", 16: "Chicken pox", 17: "Dengue", 18: "Typhoid", 19: "Hepatitis A",
    20: "Hepatitis B", 21: "Hepatitis C", 22: "Hepatitis D", 23: "Hepatitis E",
    24: "Alcoholic hepatitis", 25: "Tuberculosis", 26: "Common Cold", 27: "Pneumonia",
    28: "Dimorphic hemorrhoids(piles)", 29: "Heart attack", 30: "Varicose veins",
    31: "Hypothyroidism", 32: "Hyperthyroidism", 33: "Hypoglycemia",
    34: "Osteoarthristis", 35: "Arthritis", 36: "(vertigo) Paroymsal Positional Vertigo",
    37: "Acne", 38: "Urinary tract infection", 39: "Psoriasis", 40: "Impetigo"
}

# ===================== Helper Functions =====================
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    input_vector = input_vector.reshape(1, -1)
    prediction = svc.predict(input_vector)[0]
    return diseases_list.get(prediction, "Unknown Disease")


def helper(dis):
    desc = " ".join(description[description['Disease'] == dis]['Description'])
    pre = precautions[precautions['Disease'] == dis][
        ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.flatten().tolist()
    med = medications[medications['Disease'] == dis]['Medication'].values.tolist()
    die = diets[diets['Disease'] == dis]['Diet'].values.tolist()

    # Fix workout retrieval
    wrkout_rows = workout[workout['disease'] == dis]['workout'].values
    wrkout = " ".join(wrkout_rows) if len(wrkout_rows) > 0 else "No workout recommendations available."

    return desc, pre, med, die, wrkout


def predict_tumor(image_path):
    IMAGE_SIZE = 128
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = brain_model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]
    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

# ===================== Routes =====================
@app.route('/')
def index():
    return render_template('index..html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        user_symptoms = [s.strip() for s in symptoms.split(',')]
        user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
        predicted_disease = get_predicted_value(user_symptoms)

        desc, pre, med, die, wrkout = helper(predicted_disease)

        # Ensure lists are properly formatted and stripped of unwanted characters
        def clean_list(data):
            if isinstance(data, list) and data:
                return [item.strip("[]' ") for item in data[0].split(",")]
            return []

        my_pre = clean_list(pre)
        my_die = clean_list(die)
        my_med = clean_list(med)
        my_wrkout = clean_list(wrkout)

        return render_template(
            'testinominal.html',
            predicted_disease=predicted_disease,
            dis_des=desc,
            dis_pre=my_pre,
            dis_diet=my_die,
            dis_med=my_med,
            dis_wrkout=wrkout  # Directly passing string, not a list
        )



@app.route('/Brain', methods=['GET', 'POST'])
def brain_page():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            file_location = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_location)
            result, confidence = predict_tumor(file_location)
            return render_template('Brain.html', result=result,
                                   confidence=f"{confidence*100:.2f}%",
                                   file_path=f'/uploads/{file.filename}')
    return render_template('Brain.html', result=None)

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/about')
def about(): return render_template('about.html')

@app.route('/contact')
def contact(): return render_template('contact.html')

@app.route('/developer')
def developer(): return render_template('developer.html')

@app.route('/testinominal')
def testinominal(): return render_template('testinominal.html')

@app.route('/blog')
def blog(): return render_template('blog.html')

@app.route('/services')
def services(): return render_template('services.html')

if __name__ == "__main__":
    app.run(debug=True)
