import pandas as pd

df = pd.read_csv("final_output.csv")

district_col = "District"
service_cols = [col for col in df.columns if col != district_col]

rows = []

for _, row in df.iterrows():
    district = row[district_col]
    
    for col in service_cols:
        value = row[col]
        
        if pd.notna(value) and str(value).strip() != "":
            rows.append([district, str(value).strip()])

df_final = pd.DataFrame(rows, columns=["District", "Service"])

# ❌ Remove numeric-only rows (IMPORTANT)
df_final = df_final[
    ~df_final["Service"].astype(str).str.match(r'^\d+$')
]

# Add Sr No.
df_final.insert(0, "Sr No.", range(1, len(df_final) + 1))

df_final.to_csv("final_output.csv", index=False)

print("Done!")