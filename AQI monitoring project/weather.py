import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import joblib
import json
from datetime import datetime, timedelta
import plotly.express as px

st.set_page_config(page_title="AQI + Weather Forecast", page_icon="🌤️", layout="wide")

# ------------------
# Minimal CSS for attractive background + cards
# ------------------
st.markdown(
    """
    <style>
    .stApp {
      background-image: url('https://images.unsplash.com/photo-1501854140801-50d01698950b?ixlib=rb-4.0.3&auto=format&fit=crop&w=1740&q=80');
      background-attachment: fixed;
      background-size: cover;
    }
    .card { background: rgba(255,255,255,0.92); padding:18px; border-radius:12px; box-shadow:0 6px 18px rgba(0,0,0,0.12); }
    .big { font-size: 36px; font-weight:700; }
    .bold { font-weight:800; font-size:20px; }
    .muted { color:#666; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------
# Sidebar: API keys, model upload, inputs
# ------------------
st.sidebar.header("Settings & Uploads")
st.sidebar.markdown("Get a free API key from OpenWeatherMap (https://openweathermap.org) and paste below.")
ow_api_key = st.sidebar.text_input("OpenWeatherMap API Key", type="password")

st.sidebar.markdown("---")
model_file = st.sidebar.file_uploader("model.pkl", type=["pkl", "joblib"], help="Model should accept a 2D array and return horizon-length predictions or (1, horizon)")
scaler_file = st.sidebar.file_uploader("Optional scaler (.pkl/.joblib)", type=["pkl", "joblib"], help="Upload the scaler used during training if any")
feat_json = st.sidebar.file_uploader("Optional feature_names.json", type=["json"], help="Upload JSON list of feature names in correct order")

st.sidebar.markdown("---")
location_mode = st.sidebar.selectbox("Location input", ["City name (easy)", "Latitude/Longitude (precise)"])
if location_mode == "City name (easy)":
    city = st.sidebar.text_input("City (e.g. Delhi, IN)", value="New Delhi,IN")
    lat = lon = None
else:
    lat = st.sidebar.text_input("Latitude", value="28.644800")
    lon = st.sidebar.text_input("Longitude", value="77.216721")
    city = None

horizon = st.sidebar.slider("Forecast horizon (hours)", 1, 48, 24)
bg_url = st.sidebar.text_input("Background image URL (optional)", value="")
if bg_url:
    st.markdown(f"<style>.stApp {{ background-image: url('{bg_url}'); }}</style>", unsafe_allow_html=True)

# ------------------
# Helpers: OpenWeatherMap API calls
# ------------------

def get_coords_for_city(city_name, api_key):
    # calls OWM geocoding to get lat/lon
    try:
        url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={api_key}"
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        data = res.json()
        if len(data) == 0:
            return None
        return float(data[0]['lat']), float(data[0]['lon'])
    except Exception:
        return None


def get_current_weather(lat, lon, api_key):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    res = requests.get(url, timeout=10)
    res.raise_for_status()
    return res.json()


def get_current_aqi(lat, lon, api_key):
    # OpenWeatherMap returns 'list' with 'main':{'aqi': 1-5}
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    res = requests.get(url, timeout=10)
    res.raise_for_status()
    return res.json()

# ------------------
# Load uploaded model/scaler/features
# ------------------
model = None
scaler = None
feature_names = None

if model_file is not None:
    try:
        # try joblib first
        try:
            model = joblib.load(model_file)
        except Exception:
            model_file.seek(0)
            model = pickle.load(model_file)
        st.sidebar.success("Model loaded ✅")
    except Exception as e:
        st.sidebar.error(f"Failed loading model: {e}")

if scaler_file is not None:
    try:
        try:
            scaler = joblib.load(scaler_file)
        except Exception:
            scaler_file.seek(0)
            scaler = pickle.load(scaler_file)
        st.sidebar.success("Scaler loaded ✅")
    except Exception as e:
        st.sidebar.warning(f"Scaler load failed: {e}")

if feat_json is not None:
    try:
        feature_names = json.load(feat_json)
        st.sidebar.success("Feature list loaded ✅")
    except Exception as e:
        st.sidebar.warning(f"Feature list load failed: {e}")

# ------------------
# Main layout
# ------------------
st.title("🌤️ AQI & Weather Forecast — Live + Model")
st.markdown("<div class='muted'>Shows current AQI (from OpenWeatherMap) and model-predicted hourly AQI.</div>", unsafe_allow_html=True)

col1, col2 = st.columns([2,1])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Live location & weather")

    if not ow_api_key:
        st.warning("Enter your OpenWeatherMap API key in the sidebar to fetch live weather & AQI.")
    else:
        # determine coords
        if city:
            coords = get_coords_for_city(city, ow_api_key)
            if coords is None:
                st.error("Could not find coordinates for the city. Check spelling or use lat/lon.")
            else:
                lat, lon = coords
        else:
            try:
                lat = float(lat); lon = float(lon)
            except Exception:
                st.error("Invalid lat/lon")
                lat = lon = None

        if lat is not None and lon is not None:
            try:
                weather_json = get_current_weather(lat, lon, ow_api_key)
                aqi_json = get_current_aqi(lat, lon, ow_api_key)

                temp = weather_json['main']['temp']
                desc = weather_json['weather'][0]['description'].title()
                humidity = weather_json['main']['humidity']
                wind = weather_json['wind'].get('speed', None)

                # OWM AQI: 1 (Good) ... 5 (Very Poor)
                owm_aqi_index = aqi_json['list'][0]['main']['aqi']
                # Convert OWM 1-5 to rough US AQI scale for display (approx)
                aqi_map = {1: 50, 2: 100, 3: 150, 4: 200, 5: 300}
                current_aqi_est = aqi_map.get(owm_aqi_index, None)

                st.metric(label="Location", value=f"Lat {lat:.3f}, Lon {lon:.3f}")
                st.write(f"**Weather:** {desc} — {temp} °C — Humidity: {humidity}%")
                if wind is not None:
                    st.write(f"**Wind speed:** {wind} m/s")

                st.markdown(f"<div style='margin-top:12px'><span class='muted'>OpenWeatherMap AQI index (1-5):</span> <span class='bold'>{owm_aqi_index}</span></div>", unsafe_allow_html=True)
                if current_aqi_est is not None:
                    st.markdown(f"<div style='margin-top:8px'><span class='muted'>Approx. AQI (mapped):</span> <span class='big'>{current_aqi_est:.0f}</span></div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Failed fetching live data: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

    # show recent demo / history data (we will create minimal history using current values)
    st.markdown('<div class="card" style="margin-top:12px">', unsafe_allow_html=True)
    st.header("Recent (demo) hourly history")
    # build a tiny history using current value as last point
    try:
        last_dt = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        rng = pd.date_range(end=last_dt, periods=72, freq='H')
        # create synthetic AQI history around current_aqi_est if available else 100
        base = (current_aqi_est if 'current_aqi_est' in locals() and current_aqi_est is not None else 100)
        hist = base + np.random.RandomState(0).normal(scale=8, size=len(rng))
        df_hist = pd.DataFrame({'AQI_hourly': hist, 'temp': temp if 'temp' in locals() else 25, 'humidity': humidity if 'humidity' in locals() else 60}, index=rng)
        st.line_chart(df_hist['AQI_hourly'])
        st.write(df_hist.tail(6))
    except Exception:
        st.info("Upload your historical CSV in the sidebar to use real history.")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Prediction & Actions")

    st.markdown("Upload your trained model (.pkl/.joblib) in the sidebar. If not uploaded, persistence baseline used.")

    predict_btn = st.button("Predict next hours")

    if predict_btn:
        # Prepare input features from last row of df_hist (we use our synthetic history or real uploaded history if available)
        try:
            # feature engineering (same as your pipeline)
            def add_features_local(df):
                df = df.copy()
                df = df.sort_index()
                df['hour'] = df.index.hour
                df['dayofweek'] = df.index.dayofweek
                for lag in range(1, 25):
                    df[f"AQI_lag_{lag}"] = df['AQI_hourly'].shift(lag)
                df['AQI_roll_6'] = df['AQI_hourly'].rolling(6).mean().shift(1)
                df['AQI_roll_24'] = df['AQI_hourly'].rolling(24).mean().shift(1)
                df = df.fillna(method='ffill').fillna(method='bfill')
                return df

            feat_df = add_features_local(df_hist)
            X_input = feat_df.drop('AQI_hourly', axis=1).iloc[-1].values.reshape(1, -1)

            if scaler is not None:
                try:
                    X_input = scaler.transform(X_input)
                except Exception as e:
                    st.warning(f"Scaler transform failed: {e} — proceeding without scaler")

            if feature_names is not None:
                try:
                    X_input = feat_df.drop('AQI_hourly', axis=1)[feature_names].iloc[-1].values.reshape(1, -1)
                except Exception as e:
                    st.warning(f"Applying feature_names failed: {e} — using default order")

            if model is None:
                st.info("No model uploaded — using persistence baseline (repeat last AQI)")
                last = df_hist['AQI_hourly'].iloc[-1]
                preds = np.array([last for _ in range(horizon)])
            else:
                ypred = model.predict(X_input)
                ypred = np.array(ypred).reshape(-1)
                if len(ypred) >= horizon:
                    preds = ypred[:horizon]
                else:
                    preds = np.pad(ypred, (0, horizon - len(ypred)), 'edge')

            index = pd.date_range(start=df_hist.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq='H')
            pred_series = pd.Series(preds, index=index)

            # Show predicted AQI prominently
            st.markdown(f"<div class='big' style='margin-top:6px'>Predicted AQI (next {horizon}h): <span style='color:#c7254e'>{pred_series.iloc[0]:.1f}</span></div>", unsafe_allow_html=True)
            st.markdown('<div class="muted">(first hour shown in bold below)</div>', unsafe_allow_html=True)

            # large bold first value
            st.markdown(f"<div style='font-size:28px; font-weight:800; margin-top:10px'>First hour prediction: <span style='color:#111'>{pred_series.iloc[0]:.1f}</span></div>", unsafe_allow_html=True)

            # Chart and table
            fig = px.line(pred_series.reset_index(), x='index', y=0, labels={'index':'time', 0:'AQI'}, title='Predicted AQI')
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pred_series.to_frame('Predicted_AQI'))

            # Download
            csv = pred_series.to_frame('Predicted_AQI').to_csv().encode('utf-8')
            st.download_button("Download predictions (CSV)", data=csv, file_name='predicted_aqi.csv', mime='text/csv')

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

