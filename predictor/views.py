from django.shortcuts import render, HttpResponse
import os
from django.http import JsonResponse
from lime.lime_tabular import LimeTabularExplainer
import joblib
from django.conf import settings

MODEL_DIR = os.path.join(settings.BASE_DIR, "predictor", "models")

rf_model = joblib.load(os.path.join(MODEL_DIR, "voting_model.pkl"))
top_20 = joblib.load(os.path.join(MODEL_DIR, "top_20.pkl"))
dtn = joblib.load(os.path.join(MODEL_DIR, "disease_labels.pkl"))
X_train = joblib.load(os.path.join(MODEL_DIR, "X_train_top20.pkl"))

explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=top_20,
    class_names=list(dtn),
    mode="classification"
)

def home(request):
    return render(request, "Home.html")


SYMPTOMS = [
    'Temperature', 'painless lumps', 'Age', 'blisters on hooves', 'blisters on tongue',
    'blisters on gums', 'swelling in limb', 'swelling in muscle', 'blisters on mouth',
    'crackling sound', 'lameness', 'swelling in abdomen', 'swelling in neck',
    'chest discomfort', 'fever', 'shortness of breath', 'swelling in extremities',
    'difficulty walking', 'chills', 'depression'
]

from .forms import SymptomForm



def predict_disease(symptom_dict):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    input_row = {feature: 0 for feature in SYMPTOMS}
    input_row['Age'] = symptom_dict.get('Age', 5)
    input_row['Temperature'] = symptom_dict.get('Temperature', 37)

    for symptom in SYMPTOMS:
        if symptom not in ['Age', 'Temperature']:
            input_row[symptom] = symptom_dict.get(symptom, 0)

    input_df = pd.DataFrame([input_row])

    numeric_features = ['Age', 'Temperature']
    binary_features = [col for col in SYMPTOMS if col not in numeric_features]

    X_top_input = input_df[numeric_features + binary_features]

    scaler = StandardScaler()
    scaler.fit(input_df[numeric_features])
    X_top_input[numeric_features] = scaler.transform(X_top_input[numeric_features])

    X_final = X_top_input[top_20].apply(pd.to_numeric, errors='coerce').fillna(0)

    prediction = rf_model.predict(X_final)
    return dtn[prediction[0]],X_final





def symptom_check(request):
    if request.method == 'POST':
        form = SymptomForm(request.POST)
        if form.is_valid():
            symptoms_selected = form.cleaned_data['symptoms']
            symptom_dict = {symptom: 0 for symptom in SYMPTOMS}

            for symptom in symptoms_selected:
                symptom_dict[symptom] = 1

            try:
                age = float(request.POST.get('age', 5))
            except:
                age = 5

            try:
                temperature = float(request.POST.get('temperature', 37))
            except:
                temperature = 37

            symptom_dict['Age'] = age
            symptom_dict['Temperature'] = temperature

            # ---- prediction + feature row ----
            predicted_disease, X_final = predict_disease(symptom_dict)

            # ---- LIME explanation ----
            exp = explainer.explain_instance(
                data_row=X_final.iloc[0],
                predict_fn=rf_model.predict_proba,
                num_features=10
            )
            lime_html = exp.as_html()
            explanation = exp.as_list()
            dark_css = """
<style>
  body {
    background-color: #000 !important;
    color: #ffffff !important;
  }
  text, tspan {
    fill: #ffffff !important;
  }
  div, span, p, h1, h2, h3, h4, h5 {
    color: #ffffff !important;
  }
</style>
"""

            lime_html = lime_html.replace("</head>", dark_css + "</head>")
            return render(
                request,
                'result.html',
                {
                    'disease': predicted_disease,
                    'explanation': explanation,
                    'lime_html': lime_html
                }
            )
    else:
        form = SymptomForm()

    return render(request, 'form.html', {'form': form})




import PyPDF2
import numpy as np
import re
from django.conf import settings
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI(api_key=settings.OPENAI_API_KEY)


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text


PDF_PATH = os.path.join(settings.BASE_DIR, 'predictor', 'static', 'livestock_info.pdf')
PDF_TEXT = extract_text_from_pdf(PDF_PATH)

CHUNK_SIZE = 500
chunks = [PDF_TEXT[i:i+CHUNK_SIZE] for i in range(0, len(PDF_TEXT), CHUNK_SIZE)]

chunk_embeddings = None


def get_chunk_embeddings():
    global chunk_embeddings
    if chunk_embeddings is not None:
        return chunk_embeddings

    chunk_embeddings = []
    for chunk in chunks:
        emb = client.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        ).data[0].embedding

        chunk_embeddings.append(emb)

    return chunk_embeddings


def rag_chat(request):
    response = None

    if request.method == "POST":
        query = request.POST.get("query")

        try:
            chunk_embs = get_chunk_embeddings()

            query_emb = client.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            ).data[0].embedding

            sims = cosine_similarity([query_emb], chunk_embs)[0]
            top_chunk = chunks[int(np.argmax(sims))]

            prompt = f"""
            Context:
            {top_chunk}

            Question:
            {query}

            Answer as a helpful veterinary assistant:
            """

            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )

            response = completion.choices[0].message.content
            response = re.sub(r'[#*_`]+', '', response)
            response = re.sub(r'\n{2,}', '\n', response.strip())
            response = response.lstrip()

        except Exception as e:
            error_text = str(e)
            if "insufficient_quota" in error_text or "You exceeded your current quota" in error_text:
                response = "OpenAI tokens are extinguished"
            else:
                response = "An unexpected error occurred"

    return render(request, "rag_chat.html", {"response": response})



def about(request):
    return render(request, "about.html")

def publication(request):
    return render(request, "publication.html")
