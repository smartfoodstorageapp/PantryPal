"""
PantryPal - AI-Powered Smart Food Storage App (Streamlit)
--------------------------------------------------
This app combines two trained models into one simple interface:
    1. Upload/take a food photo -> identifies the food (image recognition model)
    2. Enter storage details -> predicts freshness (Random Forest model)

Plus two tracking features:
    - Pantry Dashboard: a running, color-coded inventory of every item checked
      this session, so the app tells a "reduce food waste through tracking"
      story instead of just a one-off prediction.
    - Food Waste Saved counter: tallies estimated weight + money saved
      whenever the user marks an expiring item as "used" via a suggested recipe.

HOW TO RUN THIS APP:
    1. Install requirements (run once in your terminal):
       pip install streamlit tensorflow scikit-learn joblib pandas numpy pillow --break-system-packages

    2. Put these files in the SAME folder as this script:
       - food_model.keras
       - class_names.txt
       - expiry_model.joblib
       - food_encoder.joblib
       - storage_encoder.joblib
       - status_encoder.joblib

    3. Run this command in your terminal:
       streamlit run app.py

    4. It will automatically open in your browser (usually at localhost:8501)
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from PIL import Image
from datetime import datetime

# ---------- Page setup ----------
st.set_page_config(page_title="PantryPal", page_icon="🧺", layout="centered")

# ---------- Professional-but-colorful styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: var(--background-color);
}

/* Header */
.app-header {
    background: linear-gradient(120deg, #2A9D8F 0%, #40B39B 100%);
    padding: 28px 32px;
    border-radius: 18px;
    margin-bottom: 28px;
    box-shadow: 0 6px 16px rgba(42,157,143,0.25);
}
.app-header h1 {
    color: white;
    margin: 0;
    font-size: 2.1rem;
    font-weight: 700;
    letter-spacing: -0.3px;
}
.app-header p {
    color: rgba(255,255,255,0.92);
    margin: 6px 0 0 0;
    font-size: 1rem;
}

h2 {
    color: #21867A !important;
    font-weight: 600 !important;
    font-size: 1.35rem !important;
    margin-top: 0.4em !important;
}

/* Widget labels (Food item, Days since purchase, Storage type, etc.) —
   uses Streamlit's theme text-color variable so it stays readable
   whether the app is switched to light or dark mode. */
label, .stSelectbox label, .stSlider label, .stNumberInput label,
[data-testid="stWidgetLabel"] p {
    color: var(--text-color) !important;
    font-weight: 500 !important;
}

/* Tabs, file uploader text, and captions — same theme-aware fix */
.stTabs [data-baseweb="tab"] p,
[data-testid="stFileUploaderDropzoneInstructions"] div,
[data-testid="stCaptionContainer"] {
    color: var(--text-color) !important;
}

/* Status badge (shown after a prediction) */
.status-badge {
    display: inline-block;
    padding: 8px 22px;
    border-radius: 999px;
    font-size: 1.2rem;
    font-weight: 600;
    color: white;
    margin: 8px 0 4px 0;
}
.status-fresh   { background: #2E9E5B; }
.status-soon    { background: #E8934A; }
.status-spoiled { background: #D65B5B; }

/* Buttons */
div.stButton > button {
    background: #2A9D8F;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 9px 24px;
    font-weight: 600;
    font-size: 0.95rem;
    box-shadow: 0 3px 8px rgba(42,157,143,0.25);
    transition: transform 0.12s ease, background 0.12s ease;
}
div.stButton > button:hover {
    transform: translateY(-1px);
    background: #23887C;
    color: white;
}

/* Expanders (recipes) */
div[data-testid="stExpander"] {
    border-radius: 12px !important;
    border: 1px solid #E8DFCB !important;
    overflow: hidden;
    margin-bottom: 8px;
    background: white;
}
.streamlit-expanderHeader {
    font-weight: 600 !important;
}

/* Stat cards for the waste-saved counter */
.stat-row { display: flex; gap: 16px; margin: 10px 0 6px 0; flex-wrap: wrap; }
.stat-card {
    flex: 1;
    min-width: 140px;
    background: white;
    border: 1px solid #E8DFCB;
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}
.stat-card .stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #21867A;
}
.stat-card .stat-label {
    font-size: 0.85rem;
    color: #7A7568;
    margin-top: 2px;
}

/* Pantry dashboard cards */
.pantry-card {
    background: white;
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 10px;
    border-left: 6px solid #ccc;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}
.pantry-card.fresh   { border-left-color: #2E9E5B; }
.pantry-card.soon    { border-left-color: #E8934A; }
.pantry-card.spoiled { border-left-color: #D65B5B; }
.pantry-card .pc-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.pantry-card .pc-name {
    font-weight: 600;
    font-size: 1rem;
    text-transform: capitalize;
    color: #2E2A20;
}
.pantry-card .pc-status {
    font-size: 0.78rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 999px;
    color: white;
}
.pc-status.fresh   { background: #2E9E5B; }
.pc-status.soon    { background: #E8934A; }
.pc-status.spoiled { background: #D65B5B; }
.pantry-card .pc-meta {
    font-size: 0.8rem;
    color: #8A8577;
    margin-top: 4px;
}
.pantry-card .pc-saved {
    font-size: 0.78rem;
    color: #2E9E5B;
    font-weight: 600;
    margin-top: 6px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
<h1>🧺 PantryPal</h1>
<p>Your AI kitchen assistant — identify food, track freshness, and cut down on waste.</p>
</div>
""", unsafe_allow_html=True)


# ---------- Simple recipe database ----------
# This is a basic lookup table for the prototype. In the final product, this
# could be replaced with a full recipe API (like Spoonacular) for many more
# recipes and richer instructions.
RECIPE_DATABASE = {
    "apple": [
        {"name": "Apple Cinnamon Oatmeal", "instructions": "Dice the apple, cook with oats, water, and a pinch of cinnamon for 5 minutes."},
        {"name": "Baked Apple Slices", "instructions": "Slice the apple, sprinkle with cinnamon, bake at 180°C for 15 minutes."},
    ],
    "banana": [
        {"name": "Banana Smoothie", "instructions": "Blend the banana with milk (or yogurt) and a spoon of honey."},
        {"name": "Banana Pancakes", "instructions": "Mash the banana, mix with flour, egg, and milk, then pan-fry until golden."},
    ],
    "bread": [
        {"name": "Garlic Bread", "instructions": "Spread butter and crushed garlic on bread slices, toast at 180°C for 8 minutes."},
        {"name": "French Toast", "instructions": "Dip bread in a beaten egg and milk mixture, pan-fry until golden on both sides."},
    ],
    "milk": [
        {"name": "White Sauce (Bechamel)", "instructions": "Melt butter, whisk in flour, then slowly add milk while stirring until thick."},
        {"name": "Rice Pudding", "instructions": "Simmer cooked rice in milk with sugar and a pinch of cinnamon until creamy."},
    ],
    "tomato": [
        {"name": "Simple Tomato Sauce", "instructions": "Chop tomatoes, simmer with garlic and olive oil for 15 minutes."},
        {"name": "Tomato Soup", "instructions": "Blend cooked tomatoes with onion and stock, simmer for 10 minutes."},
    ],
    "lettuce": [
        {"name": "Simple Garden Salad", "instructions": "Chop lettuce, toss with olive oil, lemon juice, and salt."},
    ],
    "chicken": [
        {"name": "Pan-Fried Chicken", "instructions": "Season chicken, pan-fry 6-7 minutes per side until fully cooked."},
    ],
    "cheese": [
        {"name": "Grilled Cheese Sandwich", "instructions": "Place cheese between bread slices, pan-fry both sides until golden."},
    ],
    "yogurt": [
        {"name": "Yogurt Fruit Parfait", "instructions": "Layer yogurt with any fruit and a sprinkle of granola."},
    ],
    "potato": [
        {"name": "Simple Mashed Potatoes", "instructions": "Boil potatoes until soft, mash with butter, milk, and salt."},
    ],
}

# ---------- Rough average weight (kg) + price (QAR) per item ----------
# Prototype-level assumptions used to estimate "food waste saved" impact.
# In a final product these could be pulled from real grocery pricing data.
AVERAGE_ITEM_DATA = {
    "apple":   {"weight_kg": 0.15, "price_qar": 2.18},
    "banana":  {"weight_kg": 0.12, "price_qar": 0.91},
    "bread":   {"weight_kg": 0.50, "price_qar": 10.92},
    "milk":    {"weight_kg": 1.00, "price_qar": 4.37},
    "tomato":  {"weight_kg": 0.15, "price_qar": 1.82},
    "lettuce": {"weight_kg": 0.30, "price_qar": 5.46},
    "chicken": {"weight_kg": 0.40, "price_qar": 10.92},
    "cheese":  {"weight_kg": 0.20, "price_qar": 9.10},
    "yogurt":  {"weight_kg": 0.15, "price_qar": 2.91},
    "potato":  {"weight_kg": 0.20, "price_qar": 1.46},
}

STATUS_META = {
    "fresh":         {"icon": "✅", "label": "Fresh",         "css": "fresh"},
    "expiring_soon": {"icon": "⚠️", "label": "Expiring Soon", "css": "soon"},
    "spoiled":       {"icon": "❌", "label": "Spoiled",       "css": "spoiled"},
}

# ---------- Session state (Pantry Dashboard + Waste Saved counter) ----------
if "pantry_items" not in st.session_state:
    st.session_state.pantry_items = []          # list of dicts, most recent last
if "item_counter" not in st.session_state:
    st.session_state.item_counter = 0
if "waste_weight_kg" not in st.session_state:
    st.session_state.waste_weight_kg = 0.0
if "waste_money_qar" not in st.session_state:
    st.session_state.waste_money_qar = 0.0


def add_pantry_item(food, status, confidence):
    st.session_state.item_counter += 1
    st.session_state.pantry_items.append({
        "id": st.session_state.item_counter,
        "food": food,
        "status": status,
        "confidence": confidence,
        "time": datetime.now().strftime("%I:%M %p"),
        "saved": False,
    })


def mark_item_saved(item):
    if item["saved"]:
        return
    item["saved"] = True
    avg = AVERAGE_ITEM_DATA.get(item["food"], {"weight_kg": 0.2, "price_qar": 3.64})
    st.session_state.waste_weight_kg += avg["weight_kg"]
    st.session_state.waste_money_qar += avg["price_qar"]


# ---------- Load models (cached so it only loads once, not on every click) ----------
@st.cache_resource
def load_image_model():
    model = tf.keras.models.load_model("food_model.keras")
    with open("class_names.txt") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names


@st.cache_resource
def load_expiry_model():
    model = joblib.load("expiry_model.joblib")
    food_encoder = joblib.load("food_encoder.joblib")
    storage_encoder = joblib.load("storage_encoder.joblib")
    status_encoder = joblib.load("status_encoder.joblib")
    return model, food_encoder, storage_encoder, status_encoder


try:
    image_model, class_names = load_image_model()
    expiry_model, food_encoder, storage_encoder, status_encoder = load_expiry_model()
    models_loaded = True
except Exception as e:
    st.error(f"Couldn't load model files. Make sure all 6 files are in this folder. Error: {e}")
    models_loaded = False


if models_loaded:
    # ---------- FOOD WASTE SAVED COUNTER ----------
    st.header("💚 Food Waste Saved")
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-value">{st.session_state.waste_weight_kg:.2f} kg</div>
            <div class="stat-label">Food saved from the trash</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">QR {st.session_state.waste_money_qar:.2f}</div>
            <div class="stat-label">Estimated money saved</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{sum(1 for i in st.session_state.pantry_items if i["saved"])}</div>
            <div class="stat-label">Items rescued this session</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Estimates based on average weight/price per item. Tally grows when you mark a suggested recipe as used below.")

    # ---------- SECTION 1: Food Recognition ----------
    st.header("📸 Step 1: Identify the food")

    tab_camera, tab_upload = st.tabs(["📷 Take a Photo", "🖼️ Upload a Photo"])

    uploaded_photo = None
    with tab_camera:
        camera_photo = st.camera_input("Take a photo of the food")
        if camera_photo is not None:
            uploaded_photo = camera_photo
    with tab_upload:
        file_photo = st.file_uploader("Upload a food photo", type=["jpg", "jpeg", "png"])
        if file_photo is not None:
            uploaded_photo = file_photo

    identified_food = None

    if uploaded_photo is not None:
        img = Image.open(uploaded_photo).convert("RGB")
        st.image(img, caption="Uploaded photo", width=250)

        # Prepare image the same way it was prepared during training
        img_resized = img.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img_resized)
        img_array = tf.expand_dims(img_array, 0)

        predictions = image_model.predict(img_array)
        scores = tf.nn.softmax(predictions[0])
        identified_food = class_names[np.argmax(scores)]
        confidence = 100 * np.max(scores)

        st.success(f"Identified: **{identified_food}** ({confidence:.1f}% confidence)")

    # ---------- SECTION 2: Freshness Prediction ----------
    st.header("🥑 Step 2: Check freshness")

    # If a food was identified above, pre-select it; otherwise let user pick manually
    available_foods = list(food_encoder.classes_)
    identified_food_lower = identified_food.lower() if identified_food else None
    default_index = available_foods.index(identified_food_lower) if identified_food_lower in available_foods else 0

    food_choice = st.selectbox("Food item", available_foods, index=default_index)
    days_since_purchase = st.number_input("Days since purchase", min_value=0.0, value=3.0, step=1.0)
    storage_choice = st.selectbox("Storage type", list(storage_encoder.classes_))
    temperature_c = st.slider("Temperature (°C)", min_value=0.0, max_value=35.0, value=20.0)
    humidity_level = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)

    if st.button("Predict freshness"):
        food_encoded = food_encoder.transform([food_choice])[0]
        storage_encoded = storage_encoder.transform([storage_choice])[0]

        input_data = pd.DataFrame([{
            "food_item_encoded": food_encoded,
            "days_since_purchase": days_since_purchase,
            "storage_type_encoded": storage_encoded,
            "temperature_c": temperature_c,
            "humidity_level": humidity_level,
        }])

        prediction_encoded = expiry_model.predict(input_data)[0]
        prediction = status_encoder.inverse_transform([prediction_encoded])[0]
        probabilities = expiry_model.predict_proba(input_data)[0]
        confidence = max(probabilities) * 100

        meta = STATUS_META.get(prediction, {"icon": "", "label": prediction, "css": "soon"})

        st.markdown(
            f'<div class="status-badge status-{meta["css"]}">{meta["icon"]} {meta["label"]}</div>',
            unsafe_allow_html=True,
        )
        st.write(f"Confidence: {confidence:.1f}%")

        # Log this check into the running Pantry Dashboard
        add_pantry_item(food_choice, prediction, confidence)

        # ---------- SECTION 3: Recipe Suggestion ----------
        if prediction == "spoiled":
            st.warning(f"This {food_choice} is likely spoiled — consider discarding it.")
        else:
            recipes = RECIPE_DATABASE.get(food_choice)
            if prediction == "expiring_soon":
                st.info(f"Your {food_choice} is expiring soon — here are some recipes to use it up:")
            else:
                st.write(f"🍽️ Here are some recipe ideas for your {food_choice}:")

            if recipes:
                for recipe in recipes:
                    with st.expander(f"🍳 {recipe['name']}"):
                        st.write(recipe["instructions"])
            else:
                st.write("No recipes available yet for this food item in our prototype database.")

    # ---------- SECTION 4: PANTRY DASHBOARD ----------
    st.header("🥫 Your Pantry Dashboard")

    if not st.session_state.pantry_items:
        st.caption("Nothing checked yet — identify a food and predict its freshness above to start building your pantry.")
    else:
        col_a, col_b = st.columns([3, 1])
        with col_b:
            if st.button("🗑️ Clear dashboard"):
                st.session_state.pantry_items = []
                st.rerun()

        # Show most recently checked items first
        for item in reversed(st.session_state.pantry_items):
            meta = STATUS_META.get(item["status"], {"icon": "", "label": item["status"], "css": "soon"})

            card_html = f"""
            <div class="pantry-card {meta['css']}">
                <div class="pc-top">
                    <span class="pc-name">{item['food']}</span>
                    <span class="pc-status {meta['css']}">{meta['icon']} {meta['label']}</span>
                </div>
                <div class="pc-meta">Checked at {item['time']} · {item['confidence']:.0f}% confidence</div>
                {'<div class="pc-saved">✅ Rescued with a recipe</div>' if item['saved'] else ''}
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

            # Offer a "mark as saved" action for expiring items that haven't been marked yet
            if item["status"] == "expiring_soon" and not item["saved"]:
                if st.button(f"✅ I used a recipe to save this {item['food']}", key=f"save_{item['id']}"):
                    mark_item_saved(item)
                    st.rerun()
