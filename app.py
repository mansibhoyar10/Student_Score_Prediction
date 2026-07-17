import os
import pickle
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# -------------------------------------------------------------------
# Page Config & Custom Aesthetic Styling
# -------------------------------------------------------------------
st.set_page_config(
    page_title="EduPredict Smart Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphism UI styling via CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .metric-card {
        background-color: white;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
        border-left: 5px solid #4F46E5;
        margin-bottom: 20px;
    }
    .metric-title {
        font-size: 14px;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 48px;
        font-weight: 800;
        color: #1E1B4B;
        margin-top: 5px;
    }
    .prediction-highlight {
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 64px;
        font-weight: 900;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Model Loading
# -------------------------------------------------------------------
MODEL_PATH = 'model.pkl'

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model()

# -------------------------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/graduation-cap.png", width=80)
    st.title("Control Center")
    st.markdown("Adjust student behavioral and academic metrics to see immediate real-time predictions.")
    st.markdown("---")
    
    st.subheader("⏱️ Time Management")
    hours_studied = st.slider("Daily Study Hours", 0.0, 16.0, 6.5, 0.5, help="Average hours spent studying per day")
    sleep_hours = st.slider("Daily Sleep Hours", 3.0, 12.0, 7.5, 0.5, help="Average sleep cycle duration")
    
    st.subheader("📈 Institutional Engagement")
    attendance_percent = st.slider("Attendance Rate (%)", 0.0, 100.0, 88.0, 1.0)
    previous_scores = st.slider("Last Exam Score (%)", 0.0, 100.0, 74.0, 1.0)
    
    st.markdown("---")
    st.caption("🤖 Powered by scikit-learn KNeighborsRegressor")

# -------------------------------------------------------------------
# Main Workspace
# -------------------------------------------------------------------
if model is None:
    st.error("⚠️ Error: 'model.pkl' not found. Please place the model file in the app directory.")
else:
    # Title Layout
    st.markdown("# 🧠 AI Student Performance Analytics")
    st.markdown("Evaluate academic outcome forecasting using deep behavior tracking metrics.")
    st.markdown("---")

    # Generate ML Prediction Array
    features = np.array([[hours_studied, sleep_hours, attendance_percent, previous_scores]])
    try:
        prediction = model.predict(features)[0]
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        prediction = 0.0

    # Dashboard Cards Row
    col_pred, col_metrics = st.columns([4, 5])
    
    with col_pred:
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #4F46E5; height: 100%;">
                <div class="metric-title">💡 Forecasted Final Score</div>
                <div class="prediction-highlight">{prediction:.1f}%</div>
                <p style="color: #4B5563; margin-top: 10px;">
                    Based on current trends, this student is projected to fall within a stable performing tier.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
    with col_metrics:
        # Mini analytic dashboard inside card layout
        efficiency_index = (hours_studied / (sleep_hours + 1e-5)) * 10
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #06B6D4;">
                <div class="metric-title">⚖️ Study-to-Sleep Balance Index</div>
                <div class="metric-value">{efficiency_index:.1f}</div>
                <p style="color: #4B5563; margin-top: 5px;">Ideal optimal balance index ranges between 6.0 and 9.5.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📊 Live Student Profile Radar Breakdown")
    
    # Interactive Plotly Radar Chart visualization mapping features vs maximum normalized values
    categories = ['Hours Studied', 'Sleep Hours', 'Attendance Rate', 'Previous Scores']
    # Normalize features roughly out of 100 for comparative visualization
    values = [
        (hours_studied / 16.0) * 100,
        (sleep_hours / 12.0) * 100,
        attendance_percent,
        previous_scores
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # close the loop
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(79, 70, 229, 0.2)',
        line=dict(color='#4F46E5', width=3),
        name='Student Profile'
    ))
    
fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="#9CA3AF"),
            angularaxis=dict(
                color="#1E1B4B", 
                tickfont=dict(size=14, family="Arial")  # Fixed: using tickfont specifically
            )
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=40),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Dynamic status alerts using Streamlit alerts
    st.markdown("### 🔍 Strategic Recommendations")
    if attendance_percent < 75:
        st.warning("⚠️ **Critically Low Attendance**: Regular attendance is heavily linked with continuous learning progression. Prioritize class participation.")
    elif hours_studied > 12 and sleep_hours < 6:
        st.error("🚨 **Burnout Risk Detected**: High study hours combined with low sleep triggers severe cognitive diminishing returns. Recommend increasing rest periods.")
    else:
        st.success("✨ **Healthy Balanced Metrics**: The student exhibits a well-balanced academic lifestyle blueprint. Keep up this momentum!")
