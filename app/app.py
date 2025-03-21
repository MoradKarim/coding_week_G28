# -*- coding: utf-8 -*-
"""Brayan_G28.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1RaK0uNX32-H4ZFOIwqn4i5JMLOeYWLhw

# MEDICAL DECISION SUPPORT APPLICATION PREDICTING SUCCESS OF PEDIATRIC BONE MARROW TRANSPLANTS WITH EXPLAINABLE ML (SHAP)
"""

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# from joblib import load
# 
# st.set_page_config(page_title="Prédiction de succès de greffe", page_icon="🏥", layout="wide")
# 
# @st.cache_resource
# def load_pipeline_and_model():
#     pipeline = load('pipeline_smote.joblib')
#     model = load('Final_Model.joblib')
#     return pipeline, model
# 
# def main():
#     st.title("Outil de prédiction de succès de greffe")
#     st.write("Entrez les informations sur le patient pour prédire le résultat de la greffe")
# 
#     col1, col2, col3 = st.columns(3)
# 
#     with col1:
#         st.subheader("Renseignements sur le receveur")
#         body_mass = st.number_input("Masse corporelle (Rbodymass)", min_value=0.0, max_value=200.0, value=70.0)
#         recipient_gender = st.selectbox("Genre (Recipientgender)", ["Female", "Male"])
#         recipient_blood = st.selectbox("Groupe sanguin (RecipientABO)", ["0", "A", "B", "AB"])
#         risk_group = st.selectbox("Groupe de risque (Riskgroup)", ["Risque faible", "Risque élevé"])
# 
#     with col2:
#         st.subheader("Renseignements sur le donneur")
#         donor_age = st.number_input("Age (Donorage)", min_value=0, max_value=100, value=30)
#         donor_blood = st.selectbox("Groupe sanguin (DonorABO)", ["0", "A", "B", "AB"])
#         donor_cmv = st.selectbox("Statut CMV (DonorCMV)", ["Présence", "Absence"])
#         cmv_status = st.selectbox("Combined CMV status (CMVstatus)", ["0", "1", "2", "3", "4"])
# 
#     with col3:
#         st.subheader("Paramètres cliniques")
#         cd34_kg = st.number_input("CD34+ cells par kg (CD34kgx10d6)", min_value=0.0, value=5.0)
#         cd3d_kg = st.number_input("CD3+ cells par kg (CD3dkgx10d8)", min_value=0.0, value=1.0)
#         cd3d_cd34 = st.number_input("CD3/CD34 ratio (CD3dCD34)", min_value=0.0, value=1.0)
#         survival_time = st.number_input("Temps de survie en jours (survival_time)", min_value=0, value=100)
# 
#     if st.button("Prédire le résultat de la greffe"):
#         try:
#             pipeline, model = load_pipeline_and_model()
#             input_data = {
#                 'Rbodymass': body_mass,
#                 'Donorage': donor_age,
#                 'survival_time': survival_time,
#                 'CD3dCD34': cd3d_cd34,
#                 'CD34kgx10d6': cd34_kg,
#                 'DonorCMV': 1 if donor_cmv == "Présence" else 0,
#                 'CMVstatus': cmv_status,
#                 'DonorABO': 0 if donor_blood == "0" else (1 if donor_blood == "A" else (-1 if donor_blood == "B" else 2)),
#                 'RecipientABO': 0 if recipient_blood == "0" else (1 if recipient_blood == "A" else (-1 if recipient_blood == "B" else 2)),
#                 'CD3dkgx10d8': cd3d_kg,
#                 'Riskgroup': 1 if risk_group == "Risque élevé" else 0,
#                 'Recipientgender': 1 if recipient_gender == "Male" else 0
#             }
# 
#             input_df = pd.DataFrame([input_data])
#             input_scaled = pipeline.transform(input_df)
#             prediction = model.predict(input_scaled)[0]
# 
#             st.subheader("Résultats des prédictions")
#             if prediction == 1:
#                 st.success("Le patient ne survivra pas")
#             else:
#                 st.error("Le patient survivra")
# 
#         except Exception as e:
#             st.error("Erreur lors de la prédiction. Vérifiez les données d'entrée.")
#             st.error(f"Détails de l'erreur : {str(e)}")
# 
# if __name__ == "__main__":
#     main()
#