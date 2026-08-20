import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import shap
import numpy as np
import matplotlib.pyplot as plt

# পেজ কনফিগারেশন
st.set_page_config(page_title="Sleep Disorder Analysis", layout="wide")

# উন্নত কাস্টম CSS
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    
    /* Input Labels Font Size Increased Further */
    label p {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #2c3e50 !important;
        margin-bottom: 5px !important;
    }
    
    /* Input Field Text Size Increased */
    div[data-baseweb="select"] span, div[data-baseweb="input"] input, div[data-baseweb="slider"] div {
        font-size: 18px !important; 
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
        color: white; font-weight: bold; border-radius: 12px;
        padding: 12px; font-size: 22px; border: none; transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
    .prediction-box {
        padding: 30px; border-radius: 15px; text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05); margin-bottom: 30px;
    }
    h4 { color: #2c3e50; font-weight: 600; margin-bottom: 15px; font-size: 24px !important; }
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

st.title("Advanced Sleep Disorder Dashboard")
st.markdown("Provide your details below to analyze sleep health using AI.")
st.markdown("---")

# --- ইনপুট সেকশন ---
st.markdown("### 1. Enter Your Details")
col_in1, col_in2, col_in3 = st.columns(3)

with col_in1:
    st.markdown("#### Identity")
    age = st.number_input("Age", 10, 100, 22)
    gender = st.selectbox("Gender", ["Male", "Female"])
    dept = st.selectbox("Department", ["CSE", "EEE","ECE","Civil Engineering","Mechanical Engineering","DVM", "BBA", "Pharmacy", "Agriculture","Sociology", "Mathematics","English","Development Studies","Physics","Statistics","FPE","Economics","Botany","zoology","Chemistry","Finance and Banking"])
    level = st.selectbox("Academic Level", ["Level-1", "Level-2", "Level-3", "Level-4"])
    uni = st.selectbox("University", ["HSTU", "DU", "JnU", "RU", "CU", "KU" ,"MBSTU"])

with col_in2:
    st.markdown("#### Lifestyle")
    s_dur = st.slider("Sleep Duration (Hours)", 1.0, 12.0, 7.0, 0.5)
    qual = st.select_slider("Quality of Sleep", options=["Poor", "Fair", "Good", "Excellent"], value="Good")
    act = st.slider("Physical Activity (0-100)", 0, 100, 50)
    stress = st.select_slider("Stress Level", options=["Very Low", "Low", "Moderate", "High", "Very High"], value="Moderate")

with col_in3:
    st.markdown("#### Health")
    bmi = st.selectbox("BMI Category", ["Normal", "Overweight", "Obese", "Underweight"])
    hr = st.number_input("Heart Rate (bpm)", 40, 150, 72)
    steps = st.number_input("Daily Steps", 0, 30000, 5000)
    sys = st.number_input("Systolic BP", value=120)
    dia = st.number_input("Diastolic BP", value=80)

st.markdown("<br>", unsafe_allow_html=True)
predict_button = st.button("Analyze Now", use_container_width=True)
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
    status_text = "Status:" 
    
    st.markdown(f"""
        <div class='prediction-box' style='background-color: {bg}; border: 2px solid {color}55;'>
            <h1 style='color: {color}; margin: 0;'>{status_text} {result_text}</h1>
            <p style='color: #666; font-size: 18px;'>Predicted based on your clinical and lifestyle metrics.</p>
        </div>
    """, unsafe_allow_html=True)

    col_out1, col_out2 = st.columns(2)
    
    with col_out1:
        st.markdown("#### Confidence Level")
        
        # --- UPDATE: লম্বালম্বি (Vertical) বার চার্ট এবং টেক্সট ফরম্যাটিং ---
        # লেবেলগুলোতে বোল্ড (<b>) ট্যাগ যুক্ত করা হয়েছে
        bold_class_names = [f"<b>{name}</b>" for name in class_names]
        
        fig_bar = go.Figure(go.Bar(
            x=bold_class_names, # x অক্ষে ক্লাসের নাম 
            y=probs,            # y অক্ষে probability
            text=[f"<b>{p*100:.1f}%</b>" for p in probs], # পার্সেন্টেজ টেক্সট (বোল্ড)
            textposition='auto', # টেক্সট বারের ভেতরে/বাইরে অটোমেটিক বসবে
            textfont=dict(size=20, color='black'), # টেক্সট বড় এবং কালো
            marker=dict(color=['#2ecc71', '#e74c3c', '#f1c40f'])
        ))
        
        fig_bar.update_layout(
            yaxis=dict(
                range=[0, 1.15], # বারের উপরে টেক্সটের জন্য কিছুটা ফাঁকা জায়গা রাখা হলো
                tickfont=dict(size=16, color='black'), # y অক্ষের লেখার সাইজ ও রং
                tickformat=".0%" # y অক্ষে পার্সেন্টেজ দেখাবে
            ),
            xaxis=dict(
                tickfont=dict(size=18, color='black') # x অক্ষের ক্লাস নামের সাইজ ও রং
            ),
            height=400, 
            margin=dict(l=10, r=10, t=20, b=20),
            plot_bgcolor='rgba(0,0,0,0)' # ব্যাকগ্রাউন্ড স্বচ্ছ করা হয়েছে দেখতে সুন্দর লাগার জন্য
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_out2:
        st.markdown("#### Lifestyle Radar")
        categories = ['Sleep Duration', 'Sleep Quality', 'Physical Activity', 'Stress Level', 'Heart Rate']
        
        qual_map = {"Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4}
        stress_map = {"Very Low": 5, "Low": 4, "Moderate": 3, "High": 2, "Very High": 1} 
        
        r_values = [
            s_dur / 12,
            qual_map[qual] / 4,
            act / 100,
            stress_map[stress] / 5,
            (100 - (hr-40)) / 100 if hr > 40 else 1
        ]
        
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
                angularaxis=dict(tickfont=dict(size=14, color="#2c3e50"), rotation=90, direction="clockwise")
            ),
            showlegend=False, height=400, 
            margin=dict(l=60, r=60, t=40, b=40) 
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # SHAP Plot
    st.markdown("---")
    st.markdown("### 3. AI Decision Logic (SHAP Interpretation)")
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

    # --- Recommendation Section ---
    st.markdown("---")
    st.markdown("### 💡 Personalized Recommendations")

    if is_healthy:
        st.success("🎉 Great job! Your sleep profile looks healthy. Here is how you can maintain it:")
        st.markdown("""
        * **Consistency:** Try to go to bed and wake up at the same time every day, even on weekends.
        * **Active Lifestyle:** Keep up with your daily physical activities and step count.
        * **Stress Management:** Continue managing your stress levels through hobbies or relaxation.
        """)
        
    elif "Insomnia" in result_text:
        st.warning("⚠️ Your profile indicates symptoms of Insomnia. Consider these lifestyle adjustments:")
        st.markdown("""
        * **Digital Detox:** Avoid phones, laptops, or TV at least 1 hour before bed (blue light affects sleep).
        * **Limit Caffeine:** Avoid tea, coffee, or energy drinks in the late afternoon and evening.
        * **Relaxation Routine:** Try reading a book, deep breathing exercises, or a warm bath before sleeping.
        * **Environment:** Keep your bedroom dark, quiet, and comfortably cool.
        * **Medical Advice:** If sleep deprivation affects your daily life, please consult a doctor.
        """)
        
    elif "Apnea" in result_text:
        st.error("🚨 Your profile indicates a risk of Sleep Apnea. This often requires medical attention:")
        st.markdown("""
        * **Consult a Doctor:** Sleep apnea can lead to serious health issues. Please see a healthcare professional for a proper diagnosis.
        * **Sleep Position:** Try sleeping on your side instead of your back to keep your airways open.
        * **Weight Management:** If your BMI is in the overweight or obese category, reducing weight can significantly improve symptoms.
        * **Avoid Sedatives:** Avoid alcohol or sleeping pills before bed, as they can relax throat muscles and worsen breathing pauses.
        """)
        
    else:
        st.info(f"Your result is: **{result_text}**. Please consult a healthcare provider for personalized medical advice.")

elif not predict_button:
    st.info("Please fill the inputs and click Analyze Now to see results.")
