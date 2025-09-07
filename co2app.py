import streamlit as st
import pandas as pd
import pickle
from outlier_scorer import OutlierScorer
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# Load model
# -----------------------------
with open("car_emissions_model.pkl", "rb") as f:
    model_pipeline = pickle.load(f)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="CarbonSense – Car CO₂ Predictor",
    page_icon="🌍",
    layout="wide"
)

# -----------------------------
# Sidebar (Logo + Predict)
# -----------------------------
with st.sidebar:
    st.markdown(
        "<h2 style='text-align:center; font-size:28px;'>🌍 CarbonSense</h2>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Predict button
    predict_btn = st.button("♻️ Predict", disabled=False)

    # Placeholder for showing prediction or validation messages
    result_placeholder = st.empty()
    msg_placeholder = st.empty()

# -----------------------------
# Main Layout
# -----------------------------
st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: center;">
        <span style="font-size:40px; margin-right:10px;">🚗💨</span>
        <h1 style="margin: 0; font-size:36px;">CarbonSense - CO₂ Emission Predictor</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# animation
try:
    st.image("co2img.gif", use_container_width=True)
    st.markdown(
        "<p style='text-align:right; font-size:12px; color:gray;'>"
        "Image courtesy of <a href='https://www.cleverciti.com/en/home/' target='_blank'>Cleverciti</a> (#cleveryourcity) | "
        "<em>Used for educational/demo purposes only.</em>"
        "</p>",
        unsafe_allow_html=True
    )
except:
    st.markdown("<p style='text-align:center;'>🚗💨</p>", unsafe_allow_html=True)

# -----------------------------
# Tabs for navigation
# -----------------------------
tabs = st.tabs(["🏠 Home", "📊 Analytics", "ℹ️ About"])

# Custom CSS for larger + bold tab fonts
st.markdown("""
    <style>
    .stTabs [role="tablist"] button p {
        font-size:20px !important;
        font-weight: 700 !important; 
    }
    </style>
    """, unsafe_allow_html=True)


# -----------------------------
# Home Tab
# -----------------------------
with tabs[0]:
    st.subheader("Enter Car Details")

    # Row 1
    col1, col2 = st.columns(2)
    with col1:
        engine_size = st.text_input("**Engine Size (L)**", "0.0")
    with col2:
        cylinders = st.number_input("**Cylinders**", 3, 16, 4)
    st.markdown("---")

    # Row 2
    col1, col2 = st.columns(2)
    with col1:
        city_cons = st.slider("**Fuel Consumption City (L/100km)**", 0.0, 50.0, 0.0, 0.1)
    with col2:
        hwy_cons = st.slider("**Fuel Consumption Hwy (L/100km)**", 0.0, 50.0, 0.0, 0.1)

    # Row 3
    col1, col2 = st.columns(2)
    with col1:
        comb_cons_l = st.slider("**Fuel Consumption Combined (L/100km)**", 0.0, 50.0, 0.0, 0.1)
    with col2:
        comb_cons_mpg = st.slider("**Fuel Consumption Combined (MPG)**", 0.0, 100.0, 0.0, 0.1)
    st.markdown("---")

    # -----------------------------
    # Grouped categories for dropdowns
    # -----------------------------
    makes = ["Toyota", "Ford", "Honda", "BMW", "Audi", "Chevrolet", "Hyundai", "Kia", "Mercedes-Benz", "Volkswagen", "Other"]
    models = ["Corolla", "Civic", "Camry", "Accord", "F-150", "CR-V", "3 Series", "A4", "Elantra", "RAV4", "Other"]
    vehicle_classes = ["Compact", "Midsize", "SUV", "Pickup", "Station Wagon", "Minivan", "Luxury", "Other"]

    # Transmission mapping
    transmission_map = {
        "A": "Automatic",
        "AM": "Automated manual",
        "AS": "Automatic w/ Select Shift",
        "AV": "Continuously variable",
        "M": "Manual"
    }

    # Fuel type mapping
    fuel_type_map = {
        "X": "Regular gasoline",
        "Z": "Premium gasoline",
        "D": "Diesel",
        "E": "Ethanol (E85)",
        "N": "Natural gas"
    }

    # -----------------------------
    # Row 4: Make, Model, Vehicle Class
    # -----------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        make = st.selectbox("**Make**", makes, index=0)
    with col2:
        model = st.selectbox("**Model**", models, index=0)
    with col3:
        vehicle_class = st.selectbox("**Vehicle Class**", vehicle_classes, index=0)

    st.markdown("---")

    # -----------------------------
    # Row 5: Transmission & Fuel Type
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        transmission_label = st.radio("**Transmission**", list(transmission_map.values()), horizontal=True)
        transmission = [k for k, v in transmission_map.items() if v == transmission_label][0]

    with col2:
        fuel_type_label = st.radio("**Fuel Type**", list(fuel_type_map.values()), horizontal=True)
        fuel_type = [k for k, v in fuel_type_map.items() if v == fuel_type_label][0]

# -----------------------------
# Run prediction / show validation below Predict button
# -----------------------------
if predict_btn:
    error_msgs = []

    # Engine size validation
    try:
        engine_size_val = float(engine_size)
        if not (0.5 <= engine_size_val <= 8.0):
            error_msgs.append("⚠️ Engine size must be between 0.5L and 8.0L.")
    except ValueError:
        error_msgs.append("⚠️ Engine size must be a valid number.")

    # Cylinders validation
    if not (3 <= cylinders <= 16):
        error_msgs.append("⚠️ Cylinders must be between 3 and 16.")

    # Fuel consumption realistic ranges
    MIN_CITY = 3.0
    MAX_CITY = 50.0
    MIN_HWY = 3.0
    MAX_HWY = 50.0
    MIN_COMB_L = 3.0
    MAX_COMB_L = 50.0
    MIN_COMB_MPG = 10.0
    MAX_COMB_MPG = 100.0

    # Fuel validations
    if not (MIN_CITY <= city_cons <= MAX_CITY):
        error_msgs.append(f"⚠️ City fuel consumption must be between {MIN_CITY} and {MAX_CITY} L/100km.")
    if not (MIN_HWY <= hwy_cons <= MAX_HWY):
        error_msgs.append(f"⚠️ Highway fuel consumption must be between {MIN_HWY} and {MAX_HWY} L/100km.")
    if not (MIN_COMB_L <= comb_cons_l <= MAX_COMB_L):
        error_msgs.append(f"⚠️ Combined fuel consumption (L/100km) must be between {MIN_COMB_L} and {MAX_COMB_L}.")
    if not (MIN_COMB_MPG <= comb_cons_mpg <= MAX_COMB_MPG):
        error_msgs.append(f"⚠️ Combined fuel consumption (MPG) must be between {MIN_COMB_MPG} and {MAX_COMB_MPG}.")

    if error_msgs:
        result_placeholder.empty()  # clear prediction if errors exist
        msg_placeholder.empty()  # clear previous messages
        # Combine all messages into one string separated by line breaks
        all_errors = "\n".join(error_msgs)
        msg_placeholder.error(all_errors)

    else:
        # No errors then make prediction
        input_df = pd.DataFrame([{
            "engine_size": engine_size_val,
            "cylinders": cylinders,
            "fuel_consumption_city": city_cons,
            "fuel_consumption_hwy": hwy_cons,
            "fuel_consumption_comb(l/100km)": comb_cons_l,
            "fuel_consumption_comb(mpg)": comb_cons_mpg,
            "make": make,
            "model": model,
            "vehicle_class": vehicle_class,
            "transmission": transmission,
            "fuel_type": fuel_type
        }])

        try:
            prediction = model_pipeline.predict(input_df)[0]

            if prediction < 120:
                bg_color, remark = (
                    "#7CFC00",
                    "🌱 Excellent! Low emissions. Keep up the good choice with eco-friendly vehicles."
                )
            elif prediction < 200:
                bg_color, remark = (
                    "#FFD700",
                    "⚠️ Moderate emissions. Consider hybrid options or regular maintenance to reduce footprint."
                )
            else:
                bg_color, remark = (
                    "#FF6347",
                    "🔥 High emissions. Switching to a more fuel-efficient or electric car can greatly help the environment."
                )


            result_placeholder.markdown(
                f"<div style='background-color:{bg_color}; padding:10px; border-radius:8px; text-align:center;'>"
                f"<h3 style='color:black;'>Predicted CO₂ Emissions: {prediction:.2f} g/km</h3></div>",
                unsafe_allow_html=True
            )
            msg_placeholder.info(remark)

            st.session_state["prediction"] = prediction
            st.session_state["remark"] = remark

        except Exception as e:
            result_placeholder.error(f"Error: {e}")

# -----------------------------
# Analytics Tab
# -----------------------------
with tabs[1]:
    st.subheader("📊 Result Analytics")

    if "prediction" in st.session_state:
        pred = st.session_state["prediction"]
        remark = st.session_state.get("remark", "")

        # 1. Gauge-like bar (emission level)
        fig1, ax1 = plt.subplots(figsize=(3, 0.3), dpi=120)
        ax1.barh([0], [pred], color="orange")
        ax1.set_xlim(0, 300)
        ax1.set_yticks([])
        ax1.set_title("Emission Level (g/km)", fontsize=8)
        ax1.tick_params(axis="x", labelsize=6)
        st.pyplot(fig1)

        st.markdown(
            f"<p style='font-size:20px;'>Your CO₂ emission is <b>{pred:.1f} g/km</b> compared to the 300 g/km scale. "
            "We use 300 because it covers the majority of real-world passenger cars.</p>",
            unsafe_allow_html=True
        )

        st.markdown("---")

        # 2. Comparison bar with thresholds (below emission plot)
        thresholds = {"Low<120": 120, "Mod<200": 200, "High": 300}
        fig2, ax2 = plt.subplots(figsize=(3.5, 1.8), dpi=120)
        ax2.bar(thresholds.keys(), thresholds.values(), color=["green", "gold", "red"], alpha=0.6)
        ax2.axhline(pred, color="blue", linestyle="--", label=f"Your Car: {pred:.1f}")
        ax2.legend(fontsize=8)
        ax2.set_title("Threshold Comparison", fontsize=6)
        st.pyplot(fig2)

        if pred < 120:
            msg = "✅ *Low emission* — excellent for the environment."
        elif pred < 200:
            msg = "⚠️ *Moderate emission* — acceptable but could be improved."
        else:
            msg = "🔥 *High emission* — higher costs and more environmental impact."
        
        st.markdown(f"<p style='font-size:20px;'>{msg}</p>", unsafe_allow_html=True)

        st.markdown("---")

        # Row 2: Pie chart + explanation side by side
        col3, col4 = st.columns(2)

        with col3:
            fig3, ax3 = plt.subplots(figsize=(2.5, 2.5), dpi=120)
            ax3.pie([pred, max(0, 300 - pred)],
                    labels=["Your Car", "Remaining"],
                    autopct="%1.1f%%",
                    colors=["red", "lightgray"],
                    textprops={'fontsize': 8})
            ax3.set_title("Your Emission vs 300 g/km", fontsize=10)
            st.pyplot(fig3)

        with col4:
            st.markdown(
                f"\n\n\n\n\n<p style='font-size:20px;'>Your car contributes <b>{(pred/300)*100:.1f}%</b> of the maximum scale we consider (300 g/km). "
                "Lower percentages mean a cleaner, more efficient vehicle.</p>",
                unsafe_allow_html=True
            )

        st.markdown("---")

        # Final conclusion
        st.success(
            f"**Final Thoughts:** {remark} "
            "This analysis helps you see how your car compares to common benchmarks "
            "and whether there’s room to improve your choice for lower emissions."
        )

    else:
        st.warning("⚠️ No prediction yet. Please go to **Home** tab and click Predict.")

# -----------------------------
# About Tab
# -----------------------------
with tabs[2]:
    st.subheader("ℹ️ About CarbonSense")
    st.markdown("""
    **What this app does**  
    CarbonSense helps you estimate how much CO₂ a car produces, based on its engine and fuel details.  
    It’s designed for car buyers, eco-conscious drivers, and policymakers who want quick insights into vehicle emissions.  

    ---

    **How predictions are made (in simple terms)**  
    - The app looks at car details like **engine size, number of cylinders, fuel type, and transmission**.  
    - It compares these with thousands of real cars tested in official records.  
    - Using patterns from this data, it predicts the **average CO₂ emissions** for your car.  

    ---

    **What the numbers mean**  
    - **Lower CO₂ (under ~120 g/km):** 🌱 Great for the environment, efficient driving.  
    - **Moderate CO₂ (120–200 g/km):** ⚠️ Acceptable but could be improved.  
    - **High CO₂ (200+ g/km):** 🔥 Heavy emissions, higher fuel costs, bigger footprint.  

    ---

    **Things to keep in mind**  
    - Predictions are based on **lab test data**; real driving conditions (traffic, road type, weather, driving style) can give different results.  
    - The model focuses on **passenger cars** — trucks, buses, or modified vehicles may not be accurate.  
    - Car maintenance also matters: a well-maintained car usually emits less CO₂.  

    ---

    **Why this matters**  
    Every car journey adds to our carbon footprint. By comparing cars and their emissions, you can:  
    ✅ Choose greener vehicles  
    ✅ Save on fuel costs  
    ✅ Contribute to a cleaner planet  
    """)