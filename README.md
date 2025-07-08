🏥  Healthcare Web Application  ( Manish Life Care )
This is an integrated healthcare web application built using Flask that provides users with intelligent health predictions and assistance through multiple features. The system combines machine learning, deep learning, and NLP technologies to deliver a modern health support platform.

🌟 Features
🤒 1. Disease Prediction

Users can input symptoms through a form.
The system uses a Support Vector Classifier (SVC) model to predict the most likely disease.

Provides:
📄 1.Description of the disease

💊 Recommended medications
🍲 Diet suggestions
🧘 Workout routines
⚠️ Precautionary measures

🧠 2. Brain Tumor Detection
Upload an MRI scan image.

A Convolutional Neural Network (CNN) model analyzes the image and classifies it as:

No Tumor
Glioma
Meningioma
Pituitary Tumor
Displays prediction with a confidence score.

🗓️ 3. Doctor Appointment Booking

Users can book appointments with doctors.
After booking, an automated confirmation email is sent with appointment details.

💬 4. AI Chatbot

Built-in chatbot for answering user queries.
Provides instant responses to health-related and diet-related questions.
Enhances user experience with real-time interaction.

🛠️ Technologies Used
Layer	Tools / Libraries
Frontend	HTML, CSS, Javascript Templates
Backend	Flask (Python)
ML Model	Xgboost (eXtreme Gradient Boosting)
DL Model	Keras CNN
NLP Bot	(Dialogflow / Custom Chatbot with NLP logic)
Data	CSV files for symptoms, descriptions, medications, diets, workouts
Other	Pandas, NumPy, Pickle, Email (SMTP), OS, Keras Preprocessing
