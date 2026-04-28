import pandas as pd
import requests
import time
from tqdm import tqdm
import plotly.express as px

# ==============================
# 1. Load Data
# ==============================
df = pd.read_csv("final_output.csv")

# ==============================
# 2. Clean Service Names
# ==============================
df["Service"] = df["Service"].astype(str).str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)

# ==============================
# 3. Enhance Address (for better accuracy)
# ==============================
def enhance_address(row):
    service = row["Service"]
    district = row["District"]
    
    s = service.lower()
    
    if "fuel" in s or "petrol" in s:
        return f"{service}, petrol pump, {district}, West Bengal, India"
    elif "hotel" in s or "dhaba" in s:
        return f"{service}, restaurant, {district}, West Bengal, India"
    elif "garage" in s or "workshop" in s:
        return f"{service}, auto repair, {district}, West Bengal, India"
    else:
        return f"{service}, {district}, West Bengal, India"

df["full_address"] = df.apply(enhance_address, axis=1)

# ==============================
# 4. Google Maps API
# ==============================
API_KEY = os.getenv("GOOGLE_MAP_API_KEY")  

def get_lat_lng(address):
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    
    params = {
        "address": address,
        "key": API_KEY,
        "region": "in",
        "components": "administrative_area:West Bengal|country:IN"
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data["status"] == "OK":
            location = data["results"][0]["geometry"]["location"]
            return location["lat"], location["lng"]
        else:
            return None, None
    except:
        return None, None

# ==============================
# 5. Fetch Coordinates
# ==============================
latitudes = []
longitudes = []

for addr in tqdm(df["full_address"]):
    lat, lng = get_lat_lng(addr)
    latitudes.append(lat)
    longitudes.append(lng)
    
    time.sleep(0.1)  # avoid rate limit

df["Latitude"] = latitudes
df["Longitude"] = longitudes

# Drop rows where location not found
df = df.dropna(subset=["Latitude", "Longitude"])

# Save file
df.to_csv("final_with_coordinates_better.csv", index=False)

print("Geocoding Done!")

# ==============================
# 6. Plot on Map (Plotly)
# ==============================
fig = px.scatter_mapbox(
    df,
    lat="Latitude",
    lon="Longitude",
    hover_name="Service",
    hover_data=["District"],
    zoom=6,
    height=700,
    title="Service Locations Map"
)

fig.update_layout(mapbox_style="open-street-map")
fig.show()