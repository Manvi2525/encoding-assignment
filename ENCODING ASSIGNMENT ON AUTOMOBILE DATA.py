import pandas as pd
import numpy as np


# dataset

raw_data = {
    "Category":[
        "Passenger Vehicles (PVs)", "Passenger Vehicles (PVs)", "Passenger Vehicles (PVs)",
        "Commercial Vehicles - M & HCVs", "Commercial Vehicles - M & HCVs", "Commercial Vehicles - M & HCVs",
        "Commercial Vehicles - LCVs", "Commercial Vehicles - LCVs", "Commercial Vehicles - LCVs",
        "Two Wheelers", "Two Wheelers", "Two Wheelers", "Two Wheelers"
    ],
    "Segment":[
        "Passenger Cars", "Multi-Utility Vehicles", "Total Passenger Vehicles (PVs)",
        "Passenger Carriers", "Goods Carriers", "Total M & HCVs",
        "Passenger Carriers", "Goods Carriers", "Total LCVs",
        "Scooter/Scooterettee", "Motorcycles/Step-Throughs", "Mopeds", "Electric Two Wheelers"
    ],
    "2006-07":[
        1238032, 307202, 1545234, 32828, 261438, 294266,
        29443, 196291, 225734, 943974, 7112225, np.nan ,379987
    ]
}

DATA = pd.DataFrame(raw_data)
print("Original DataFrame:\n", DATA,"\n")


# ORDINAL ENCODING

category_order = {
    "Passenger Vehicles (PVs)": 1,
    "Commercial Vehicles - M & HCVs": 2,
    "Commercial Vehicles - LCVs": 3,
    "Two Wheelers": 4
}
DATA["Category_Encoded"] = DATA["Category"].map(category_order)
print("After Ordinal Encoding:\n", DATA[["Category", "Category_Encoded"]],"\n")


#  ONE-HOT ENCODING

one_hotEncoding = pd.get_dummies(DATA["Segment"], prefix="Segment")
DATA = pd.concat([DATA, one_hotEncoding], axis=1)
print("After One Hot Encoding:\n", DATA.head(), "\n")


#  MISSING VALUE IMPUTATION

# Mean Imputation
mean_value = DATA["2006-07"].mean()
DATA["Imputed_Mean"] = DATA["2006-07"].fillna(mean_value)

# Mode Imputation
mode_value = DATA["2006-07"].mode()[0]
DATA["Imputed_Mode"] = DATA["2006-07"].fillna(mode_value)

print("After Mean & Mode Imputation:\n", DATA[["Segment", "2006-07", "Imputed_Mean", "Imputed_Mode"]], "\n")


# NEW DATA

print("Encoded & Imputed Data:\n")
print(DATA.to_string(index=False))
