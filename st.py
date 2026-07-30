import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import shap
import numpy as np
import matplotlib.pyplot as plt

# পেজ কনফিগারেশন
st.set_page_config(page_title="Sleep Disorder Analysis", page_icon="🌙", layout="wide")

# উন্নত কাস্টম CSS (Glassmorphism Touch)
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
        color: white; font-weight: bold; border-radius: 12px;
        padding: 12px; font-size: 18px; border: none; transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
    .prediction-box {
        padding: 30px; border-radius: 15px; text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05); margin-bottom: 30px;
    }
    h4 { color: #2c3e50; font-weight: 600; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_explainer():
    try:
        data = joblib.load('best_sleep_model.pkl')
        model, scaler, encoders, features = data['model'], data['scaler'], data['label_encoders'], data['feature_names']
        
        # Background data for SHAP
        bg_data = data['train_sample'] if 'train_sample' in data else shap.sample(scaler.transform(pd.DataFrame(np.zeros((10, len(features))), columns=features)), 5)
        explainer = shap.KernelExplainer(model.predict, bg_data)
        return model, scaler, encoders, features, explainer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return [None]*5

model, scaler, encoders, features, explainer = load_model_and_explainer()

st.title("🌙 Advanced Sleep Disorder Dashboard")
st.markdown("Provide your details below to analyze sleep health using AI.")
st.markdown("---")

# --- ইনপুট সেকশন ---
st.markdown("### ⚙️ 1. Enter Your Details")
col_in1, col_in2, col_in3 = st.columns(3)

with col_in1:
    st.markdown("#### 👤 Identity")
    age = st.number_input("Age", 10, 100, 22)
    gender = st.selectbox("Gender", ["Male", "Female"])
    dept = st.selectbox("Department", ["CSE", "EEE","ECE","Civil Engineering","Mechanical Engineering","DVM", "BBA", "Pharmacy", "Agriculture","Sociology", "Mathematics","English","Development Studies","Physics","Statistics","FPE","Economics","Botany","zoology","Chemistry","Finance and Banking"])
    level = st.selectbox("Academic Level", ["Level-1", "Level-2", "Level-3", "Level-4"])
    uni = st.selectbox("University", ["HSTU", "DU", "JnU", "RU", "CU", "KU" ,"MBSTU"])

with col_in2:
    st.markdown("#### 🛌 Lifestyle")
    s_dur = st.slider("Sleep Duration (Hours)", 1.0, 12.0, 7.0, 0.5)
    qual = st.select_slider("Quality of Sleep", options=["Poor", "Fair", "Good", "Excellent"], value="Good")
    act = st.slider("Physical Activity (0-100)", 0, 100, 50)
    stress = st.select_slider("Stress Level", options=["Very Low", "Low", "Moderate", "High", "Very High"], value="Moderate")

with col_in3:
    st.markdown("#### 🩺 Health")
    bmi = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese", "Underweight"])
    hr = st.number_input("Heart Rate (bpm)", 40, 150, 72)
    steps = st.number_input("Daily Steps", 0, 30000, 5000)
    sys = st.number_input("Systolic BP", value=120)
    dia = st.number_input("Diastolic BP", value=80)

st.markdown("<br>", unsafe_allow_html=True)
predict_button = st.button("🚀 Analyze Now", use_container_width=True)
st.markdown("---")

# --- আউটপুট সেকশন ---
if predict_button and model:
    input_dict = {
        'Department': dept, 'Gender': gender, 'Age': age, 'Sleep Duration': s_dur,
        'Quality of Sleep': qual, 'Physical Activity Level': act, 'Stress Level': stress,
        'BMI Category': bmi, 'Heart Rate (bpm)': hr, 'Daily Steps': steps,
        'Academic Level': level, 'University': uni, 'Systolic': sys, 'Diastolic': dia
    }
    df_in = pd.DataFrame([input_dict])[features]

    for col in df_in.columns:
        if col in encoders:
            try: df_in[col] = encoders[col].transform(df_in[col].astype(str))
            except: df_in[col] = 0
                
    scaled = scaler.transform(df_in)
    prediction = model.predict(scaled)[0]
    result_text = encoders['Sleep Disorder'].inverse_transform([prediction])[0]
    probs = model.predict_proba(scaled)[0]
    class_names = encoders['Sleep Disorder'].classes_

    is_healthy = (result_text == "No Sleep Disorder" or result_text == "None")
    color = "#2E7D32" if is_healthy else "#c62828"
    bg = "#e8f5e9" if is_healthy else "#ffebee"
    
    st.markdown(f"""
        <div class='prediction-box' style='background-color: {bg}; border: 2px solid {color}55;'>
            <h1 style='color: {color}; margin: 0;'>{'✅' if is_healthy else '⚠️'} {result_text}</h1>
            <p style='color: #666;'>Predicted based on your clinical and lifestyle metrics.</p>
        </div>
    """, unsafe_allow_html=True)

    col_out1, col_out2 = st.columns(2)
    
    with col_out1:
        st.markdown("#### 📊 Confidence Level")
        fig_bar = go.Figure(go.Bar(x=probs, y=class_names, orientation='h', marker=dict(color=['#2ecc71', '#e74c3c', '#f1c40f'])))
        fig_bar.update_layout(xaxis=dict(range=[0, 1]), height=350, margin=dict(l=10, r=10, t=20, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_out2:
        st.markdown("#### 🕸️ Lifestyle Radar")
        
        # রাডার চার্ট ফিক্স: ক্যাটাগরি এবং ভ্যালু ঠিক করা
        categories = ['Sleep Duration', 'Sleep Quality', 'Physical Activity', 'Stress Level', 'Heart Rate']
        
        # ভ্যালুগুলোকে ০-১ স্কেলে নিয়ে আসা
        qual_map = {"Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4}
        stress_map = {"Very Low": 5, "Low": 4, "Moderate": 3, "High": 2, "Very High": 1} # স্ট্রেস কম হলে ভালো, তাই উল্টো ম্যাপ
        
        r_values = [
            s_dur / 12,
            qual_map[qual] / 4,
            act / 100,
            stress_map[stress] / 5,
            (100 - (hr-40)) / 100 if hr > 40 else 1
        ]
        
        # রাডার লুপ করার জন্য প্রথম পয়েন্টটি শেষে আবার যোগ করা
        r_values += r_values[:1]
        categories += categories[:1]

        fig_radar = go.Figure(go.Scatterpolar(
            r=r_values, theta=categories, fill='toself', 
            line=dict(color='#3498db', width=2),
            fillcolor="rgba(52, 152, 219, 0.3)"
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
                angularaxis=dict(tickfont=dict(size=12, color="#2c3e50"), rotation=90, direction="clockwise")
            ),
            showlegend=False, height=380, 
            margin=dict(l=60, r=60, t=40, b=40) # মার্জিন বাড়ানো হয়েছে লেবেলের জন্য
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # SHAP Plot
    st.markdown("---")
    st.markdown("### 🧬 3. AI Decision Logic (SHAP Interpretation)")
    with st.spinner("Analyzing features..."):
        shap_values = explainer.shap_values(scaled)
        curr_shap = shap_values[prediction] if isinstance(shap_values, list) else shap_values[0]
        exp_val = explainer.expected_value[prediction] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        
        plt.clf()
        shap.force_plot(exp_val, curr_shap, df_in.iloc[0], matplotlib=True, show=False)
        fig = plt.gcf()
        fig.set_size_inches(16, 3)
        st.pyplot(fig, use_container_width=True)
        plt.clf()

elif not predict_button:
    st.info("👆 Please fill the inputs and click **Analyze Now** to see results.")