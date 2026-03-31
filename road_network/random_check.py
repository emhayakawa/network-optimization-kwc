import pandas as pd

df = pd.read_csv("./road_network/data/gmns/link.csv")
print(df['travel_time_min'].isna().sum())

