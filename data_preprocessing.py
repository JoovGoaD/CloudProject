import pandas as pd

df1 = pd.read_csv("C:/Users/JoovGoaD/Desktop/project/apartments_rent_pl_2023_11.csv")
df2 = pd.read_csv("C:/Users/JoovGoaD/Desktop/project/apartments_rent_pl_2023_12.csv")
df3 = pd.read_csv("C:/Users/JoovGoaD/Desktop/project/apartments_rent_pl_2024_01.csv")
df4 = pd.read_csv("C:/Users/JoovGoaD/Desktop/project/apartments_rent_pl_2024_02.csv")
df5 = pd.read_csv("C:/Users/JoovGoaD/Desktop/project/apartments_rent_pl_2024_03.csv")
df6 = pd.read_csv("C:/Users/JoovGoaD/Desktop/project/apartments_rent_pl_2024_04.csv")
df7 = pd.read_csv("C:/Users/JoovGoaD/Desktop/project/apartments_rent_pl_2024_05.csv")
df8 = pd.read_csv("C:/Users/JoovGoaD/Desktop/project/apartments_rent_pl_2024_06.csv")

df1["date"] = pd.to_datetime("2023-11-01")
df2["date"] = pd.to_datetime("2023-12-01")
df3["date"] = pd.to_datetime("2024-01-01")
df4["date"] = pd.to_datetime("2024-02-01")
df5["date"] = pd.to_datetime("2024-03-01")
df6["date"] = pd.to_datetime("2024-04-01")
df7["date"] = pd.to_datetime("2024-05-01")
df8["date"] = pd.to_datetime("2024-06-01")

dfs = [
    df1,
    df2,
    df3,
    df4,
    df5,
    df6,
    df7,
    df8
]

df_all = pd.concat(dfs, axis=0, ignore_index=True)

cities_keep = [
    "warszawa",
    "krakow",
    "katowice",
    "lodz",
    "wroclaw",
    "poznan",
]

df_sub = df_all[df_all["city"].isin(cities_keep)].copy()

# df_final = (
#     df_sub
#     .groupby("city", group_keys=False)
#     .apply(lambda x: x.sample(
#         n=min(len(x), 1000),
#         random_state=42
#     ))
#     .reset_index(drop=True)
# )

df_final = df_sub

cols = [
    "city",
    "rooms",
    "squareMeters",
    "centreDistance",
    "hasParkingSpace",
    "hasElevator",
    "hasSecurity",
    "price",
    "date"
]

final_dataset = df_final[cols].copy()

bool_cols = [
    "hasElevator",
    "hasParkingSpace",
    "hasSecurity",
]

bool_map = {
    "yes": 1, "no": 0,
    "true": 1, "false": 0,
    "1": 1, "0": 0,
    1: 1, 0: 0,
}

for c in bool_cols:
    final_dataset[c] = (
        final_dataset[c]
        .astype("string")
        .str.lower()
        .map(bool_map)
        .fillna(0)
        .astype("int8")
    )

final_dataset.info()
final_dataset.isna().mean().sort_values(ascending=False)
final_dataset.describe(include="all")


final_dataset.to_csv(
    "C:/Users/JoovGoaD/Desktop/final_dataset.csv",
    index=False
)
