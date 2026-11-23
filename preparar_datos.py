import os
import numpy as np
import pandas as pd

# ==========================================================
# 1. GENERACIÓN DEL DATASET (6,000 FILAS)
# Incluye variables continuas, categóricas,
# temporales y geoespaciales (cumple >= 5,000 filas)
# ==========================================================
rng = np.random.default_rng(42)
n_rows = 6000

regions_geo = {
    "North America": ("United States", 39.8283, -98.5795),
    "Europe": ("Germany", 51.1657, 10.4515),
    "Latin America": ("Mexico", 23.6345, -102.5528),
    "Asia Pacific": ("Japan", 36.2048, 138.2529),
}

categories = ["Electronics", "Apparel", "Home & Living", "Industrial", "Health"]
channels = ["Online", "Retail Store", "B2B Direct", "Marketplace"]
region_keys = list(regions_geo.keys())

dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(
    rng.integers(0, 680, size=n_rows), unit="D"
)
chosen_regions = rng.choice(region_keys, size=n_rows, p=[0.35, 0.25, 0.20, 0.20])

units = rng.integers(1, 120, size=n_rows).astype(float)
unit_price = rng.lognormal(mean=3.6, sigma=0.65, size=n_rows)
discount_rate = np.clip(rng.normal(0.10, 0.08, size=n_rows), 0.0, 0.45)
shipping_cost = units * rng.uniform(1.2, 8.5, size=n_rows)

revenue = units * unit_price * (1 - discount_rate)
cogs = units * unit_price * rng.uniform(0.45, 0.72, size=n_rows)
profit = revenue - cogs - shipping_cost

df = pd.DataFrame({
    "order_id": [f"ORD-{100000 + i}" for i in range(n_rows)],
    "order_date": dates,
    "region": chosen_regions,
    "country": [regions_geo[r][0] for r in chosen_regions],
    "lat": [regions_geo[r][1] + rng.normal(0, 2.5) for r in chosen_regions],
    "lon": [regions_geo[r][2] + rng.normal(0, 3.0) for r in chosen_regions],
    "category": rng.choice(categories, size=n_rows),
    "sales_channel": rng.choice(channels, size=n_rows),
    "units_sold": units,
    "unit_price": np.round(unit_price, 2),
    "discount_rate": np.round(discount_rate, 3),
    "revenue": np.round(revenue, 2),
    "profit": np.round(profit, 2),
    "delivery_days": np.clip(rng.normal(5.2, 2.3, size=n_rows), 1, 25).round(1),
})

# ==========================================================
# 2. SIMULACIÓN Y LIMPIEZA DE VALORES FALTANTES
# ==========================================================
na_idx_profit = rng.choice(n_rows, size=120, replace=False)
na_idx_days = rng.choice(n_rows, size=90, replace=False)
df.loc[na_idx_profit, "profit"] = np.nan
df.loc[na_idx_days, "delivery_days"] = np.nan

# Convertir fechas y eliminar duplicados o fechas inválidas
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
df = df.dropna(subset=["order_date"]).drop_duplicates(subset=["order_id"])

# Imputar valores faltantes usando la mediana por grupo
df["profit"] = df["profit"].fillna(
    df.groupby("category")["profit"].transform("median")
)
df["delivery_days"] = df["delivery_days"].fillna(
    df.groupby("region")["delivery_days"].transform("median")
)

# Columnas derivadas para análisis
df["profit_margin"] = np.where(df["revenue"] > 0, df["profit"] / df["revenue"], 0.0)
df["year_month"] = df["order_date"].dt.to_period("M").astype(str)

# Validación de tamaño mínimo requerido
assert len(df) >= 5000, "El dataset debe tener al menos 5,000 filas."

# ==========================================================
# 3. GUARDAR ARCHIVO LIMPIO
# ==========================================================
os.makedirs("data/processed", exist_ok=True)
output_path = "data/processed/clean_enterprise_data.csv"
df.to_csv(output_path, index=False)

print(f"Archivo guardado en: {output_path}")
print(f"Total de filas limpias: {len(df)}")
print(f"Valores nulos restantes: {int(df.isna().sum().sum())}")
print(df.head(3))