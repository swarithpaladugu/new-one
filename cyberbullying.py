import pandas as pd
import re
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Global variables
model = None
vectorizer = None
dataset_path = None

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

# Upload dataset
def upload_dataset():
    global dataset_path
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        dataset_path = file_path
        status_label.config(text="Dataset uploaded successfully!", fg="green")

# Train model
def train_model():
    global model, vectorizer, dataset_path

    if not dataset_path:
        status_label.config(text="Please upload a dataset first.", fg="red")
        return

    try:
        status_label.config(text="Training in progress... Please wait ", fg="blue")
        upload_window.update_idletasks()

        data = pd.read_csv(dataset_path, encoding='windows-1252')
        data['Text'] = data['Text'].apply(clean_text)
        X = data['Text']
        y = data['CB_Label']

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=5)
        X_train_vectors = vectorizer.fit_transform(X_train)

        model = SVC(kernel='linear', class_weight='balanced')

        time.sleep(10)  # Simulate training delay
        model.fit(X_train_vectors, y_train)

        status_label.config(text="Model trained successfully. Opening detection window...", fg="green")
        upload_window.update_idletasks()
        time.sleep(2)
        upload_window.destroy()
        open_detection_window()

    except Exception as e:
        status_label.config(text=f"Training failed: {str(e)}", fg="red")

# Predict cyberbullying
def predict_text():
    if model is None or vectorizer is None:
        messagebox.showwarning("Warning", "Train the model first!")
        return

    user_input = text_box.get("1.0", tk.END).strip()
    if not user_input:
        messagebox.showwarning("Warning", "Enter some text!")
        return

    cleaned = clean_text(user_input)
    input_vector = vectorizer.transform([cleaned])
    prediction = model.predict(input_vector)

    if prediction[0] == 1:
        result_label.config(text="Cyberbullying Detected!", fg="red")
    else:
        result_label.config(text="No Cyberbullying Detected.", fg="green")

# GUI 1: Upload + Train
upload_window = tk.Tk()
upload_window.title("Cyberbullying Detection - Upload Dataset")
upload_window.geometry("600x300")
upload_window.configure(bg="#f0f9ff")

tk.Label(upload_window, text="Cyberbullying Detection", font=("Helvetica", 20, "bold"), bg="#f0f9ff", fg="blue").pack(pady=20)
tk.Button(upload_window, text="Upload Dataset", font=("Arial", 14), width=20, bg="#4CAF50", fg="white", command=upload_dataset).pack(pady=10)
tk.Button(upload_window, text="Train Model", font=("Arial", 14), width=20, bg="#2196F3", fg="white", command=train_model).pack(pady=10)

status_label = tk.Label(upload_window, text="", font=("Arial", 12), bg="#f0f9ff", fg="black")
status_label.pack(pady=20)

tk.Label(upload_window, text="Upload a CSV file ", font=("Arial", 11), bg="#f0f9ff").pack()

# GUI 2: Detection
def open_detection_window():
    detect_window = tk.Tk()
    detect_window.title("Cyberbullying Detection - Analyze Comment")
    detect_window.geometry("700x400")
    detect_window.configure(bg="#eef9f1")

    tk.Label(detect_window, text="Cyberbullying Detection Using SVC", font=("Helvetica", 22, "bold"), fg="green", bg="#eef9f1").pack(pady=20)

    global text_box, result_label
    text_box = tk.Text(detect_window, height=6, width=60, font=("Arial", 14))
    text_box.pack(pady=10)

    tk.Button(detect_window, text="Analyze", font=("Arial", 14), bg="#4CAF50", fg="white", padx=30, pady=5, command=predict_text).pack(pady=10)
    result_label = tk.Label(detect_window, text="", font=("Arial", 16), bg="#eef9f1")
    result_label.pack(pady=20)

    detect_window.mainloop()

upload_window.mainloop()