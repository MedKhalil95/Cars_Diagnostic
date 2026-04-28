"""
generate_data.py
================
Generates a realistic synthetic car engine diagnostic dataset that mirrors
real-world OBD-II sensor readings.

Now includes a 'model' column — each brand maps to its real model lineup,
and sensor ranges are tuned per model (sports cars run hotter/higher RPM,
economy cars have lower HP, etc.).

Run:
    python generate_data.py                  # writes to ../data/processed/
    python generate_data.py --out custom/    # custom output directory
    python generate_data.py --rows 10000     # generate more rows
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_ROWS = 5000
DEFAULT_SEED = 42
CURRENT_YEAR = 2024

ENGINE_TYPES = ['Petrol', 'Diesel', 'Hybrid', 'Electric']

# ── Brand → Model catalog ─────────────────────────────────────────────────────
# Each entry: model_name → (engine_probs, hp_mean, hp_std, rpm_bias, temp_bias)
#   engine_probs : [Petrol, Diesel, Hybrid, Electric]
#   hp_mean/std  : horsepower distribution
#   rpm_bias     : added to base RPM (sports = +400, economy = -200)
#   temp_bias    : added to base engine temp (performance = +5, mild = 0)

BRAND_MODELS: dict[str, dict] = {
    "Toyota": {
        "Corolla":  dict(ep=[0.60, 0.20, 0.15, 0.05], hp=(122, 15), rpm_b=-100, temp_b=0),
        "Yaris":    dict(ep=[0.65, 0.10, 0.20, 0.05], hp=(100, 12), rpm_b=-200, temp_b=-1),
        "Camry":    dict(ep=[0.45, 0.10, 0.40, 0.05], hp=(203, 20), rpm_b=0,    temp_b=1),
        "RAV4":     dict(ep=[0.35, 0.20, 0.40, 0.05], hp=(203, 18), rpm_b=0,    temp_b=1),
        "Hilux":    dict(ep=[0.10, 0.80, 0.08, 0.02], hp=(150, 20), rpm_b=-50,  temp_b=2),
        "Land Cruiser": dict(ep=[0.10, 0.75, 0.13, 0.02], hp=(245, 25), rpm_b=0, temp_b=3),
        "Prius":    dict(ep=[0.05, 0.00, 0.90, 0.05], hp=(121, 10), rpm_b=-300, temp_b=-2),
    },
    "BMW": {
        "320i":     dict(ep=[0.70, 0.20, 0.08, 0.02], hp=(184, 15), rpm_b=200,  temp_b=3),
        "520i":     dict(ep=[0.55, 0.30, 0.12, 0.03], hp=(184, 15), rpm_b=150,  temp_b=2),
        "X3":       dict(ep=[0.40, 0.40, 0.15, 0.05], hp=(184, 20), rpm_b=100,  temp_b=2),
        "X5":       dict(ep=[0.30, 0.45, 0.18, 0.07], hp=(265, 25), rpm_b=200,  temp_b=3),
        "M3":       dict(ep=[0.95, 0.00, 0.04, 0.01], hp=(480, 20), rpm_b=600,  temp_b=6),
        "M5":       dict(ep=[0.95, 0.00, 0.04, 0.01], hp=(600, 20), rpm_b=700,  temp_b=7),
        "118i":     dict(ep=[0.75, 0.15, 0.08, 0.02], hp=(136, 12), rpm_b=0,    temp_b=1),
    },
    "Peugeot": {
        "208":      dict(ep=[0.60, 0.25, 0.12, 0.03], hp=(100, 15), rpm_b=-150, temp_b=0),
        "308":      dict(ep=[0.50, 0.35, 0.12, 0.03], hp=(130, 18), rpm_b=-50,  temp_b=0),
        "3008":     dict(ep=[0.35, 0.40, 0.20, 0.05], hp=(130, 20), rpm_b=0,    temp_b=1),
        "508":      dict(ep=[0.45, 0.35, 0.17, 0.03], hp=(180, 20), rpm_b=100,  temp_b=2),
        "2008":     dict(ep=[0.55, 0.30, 0.12, 0.03], hp=(110, 15), rpm_b=-100, temp_b=0),
        "301":      dict(ep=[0.55, 0.40, 0.04, 0.01], hp=(115, 12), rpm_b=-100, temp_b=0),
    },
    "Renault": {
        "Clio":     dict(ep=[0.65, 0.20, 0.12, 0.03], hp=(100, 15), rpm_b=-150, temp_b=0),
        "Megane":   dict(ep=[0.50, 0.35, 0.12, 0.03], hp=(130, 18), rpm_b=0,    temp_b=1),
        "Duster":   dict(ep=[0.40, 0.50, 0.08, 0.02], hp=(115, 15), rpm_b=-50,  temp_b=1),
        "Kadjar":   dict(ep=[0.40, 0.45, 0.12, 0.03], hp=(130, 18), rpm_b=0,    temp_b=1),
        "Zoe":      dict(ep=[0.00, 0.00, 0.00, 1.00], hp=(135, 10), rpm_b=-200, temp_b=-2),
        "Symbol":   dict(ep=[0.70, 0.28, 0.01, 0.01], hp=(90,  10), rpm_b=-200, temp_b=-1),
    },
    "Volkswagen": {
        "Golf":     dict(ep=[0.55, 0.30, 0.12, 0.03], hp=(150, 20), rpm_b=0,    temp_b=1),
        "Polo":     dict(ep=[0.65, 0.25, 0.08, 0.02], hp=(95,  15), rpm_b=-150, temp_b=0),
        "Passat":   dict(ep=[0.45, 0.40, 0.12, 0.03], hp=(150, 18), rpm_b=0,    temp_b=1),
        "Tiguan":   dict(ep=[0.40, 0.40, 0.15, 0.05], hp=(150, 20), rpm_b=0,    temp_b=2),
        "Touareg":  dict(ep=[0.30, 0.45, 0.18, 0.07], hp=(286, 25), rpm_b=150,  temp_b=3),
        "ID.4":     dict(ep=[0.00, 0.00, 0.00, 1.00], hp=(204, 15), rpm_b=-200, temp_b=-2),
    },
    "Ford": {
        "Focus":    dict(ep=[0.60, 0.28, 0.10, 0.02], hp=(125, 18), rpm_b=0,    temp_b=1),
        "Fiesta":   dict(ep=[0.70, 0.20, 0.08, 0.02], hp=(95,  12), rpm_b=-150, temp_b=0),
        "Kuga":     dict(ep=[0.40, 0.35, 0.20, 0.05], hp=(150, 20), rpm_b=0,    temp_b=1),
        "Mustang":  dict(ep=[0.95, 0.00, 0.04, 0.01], hp=(450, 30), rpm_b=700,  temp_b=7),
        "Explorer": dict(ep=[0.45, 0.20, 0.30, 0.05], hp=(300, 25), rpm_b=200,  temp_b=3),
        "Ranger":   dict(ep=[0.15, 0.78, 0.05, 0.02], hp=(170, 20), rpm_b=0,    temp_b=2),
    },
    "Honda": {
        "Civic":    dict(ep=[0.70, 0.10, 0.15, 0.05], hp=(158, 18), rpm_b=100,  temp_b=1),
        "CR-V":     dict(ep=[0.45, 0.20, 0.30, 0.05], hp=(190, 20), rpm_b=0,    temp_b=1),
        "Jazz":     dict(ep=[0.50, 0.05, 0.40, 0.05], hp=(109, 10), rpm_b=-200, temp_b=0),
        "HR-V":     dict(ep=[0.55, 0.10, 0.30, 0.05], hp=(130, 15), rpm_b=-100, temp_b=0),
        "Accord":   dict(ep=[0.50, 0.10, 0.35, 0.05], hp=(192, 20), rpm_b=50,   temp_b=1),
        "e":        dict(ep=[0.00, 0.00, 0.00, 1.00], hp=(154, 10), rpm_b=-250, temp_b=-2),
    },
    "Mercedes": {
        "A-Class":  dict(ep=[0.60, 0.28, 0.10, 0.02], hp=(163, 18), rpm_b=50,   temp_b=1),
        "C-Class":  dict(ep=[0.50, 0.30, 0.15, 0.05], hp=(204, 20), rpm_b=150,  temp_b=2),
        "E-Class":  dict(ep=[0.40, 0.35, 0.20, 0.05], hp=(258, 22), rpm_b=200,  temp_b=3),
        "GLC":      dict(ep=[0.40, 0.35, 0.20, 0.05], hp=(258, 22), rpm_b=150,  temp_b=2),
        "GLE":      dict(ep=[0.30, 0.40, 0.22, 0.08], hp=(367, 28), rpm_b=200,  temp_b=3),
        "AMG GT":   dict(ep=[0.97, 0.00, 0.02, 0.01], hp=(530, 25), rpm_b=700,  temp_b=7),
        "EQC":      dict(ep=[0.00, 0.00, 0.00, 1.00], hp=(408, 15), rpm_b=-200, temp_b=-2),
    },
    "Hyundai": {
        "i10":      dict(ep=[0.80, 0.10, 0.08, 0.02], hp=(66,  10), rpm_b=-250, temp_b=-1),
        "i20":      dict(ep=[0.70, 0.18, 0.10, 0.02], hp=(84,  12), rpm_b=-200, temp_b=0),
        "i30":      dict(ep=[0.55, 0.30, 0.12, 0.03], hp=(120, 15), rpm_b=0,    temp_b=0),
        "Tucson":   dict(ep=[0.40, 0.35, 0.20, 0.05], hp=(150, 18), rpm_b=0,    temp_b=1),
        "Santa Fe": dict(ep=[0.30, 0.40, 0.22, 0.08], hp=(185, 20), rpm_b=50,   temp_b=2),
        "IONIQ 5":  dict(ep=[0.00, 0.00, 0.00, 1.00], hp=(218, 15), rpm_b=-200, temp_b=-2),
        "Kona":     dict(ep=[0.45, 0.20, 0.20, 0.15], hp=(120, 15), rpm_b=-100, temp_b=0),
    },
    "Kia": {
        "Rio":      dict(ep=[0.70, 0.18, 0.10, 0.02], hp=(100, 12), rpm_b=-200, temp_b=0),
        "Ceed":     dict(ep=[0.55, 0.30, 0.12, 0.03], hp=(120, 15), rpm_b=0,    temp_b=0),
        "Sportage": dict(ep=[0.40, 0.35, 0.20, 0.05], hp=(150, 18), rpm_b=0,    temp_b=1),
        "Sorento":  dict(ep=[0.30, 0.40, 0.22, 0.08], hp=[185, 20], rpm_b=50,   temp_b=2),
        "Stinger":  dict(ep=[0.90, 0.00, 0.08, 0.02], hp=(365, 20), rpm_b=500,  temp_b=5),
        "EV6":      dict(ep=[0.00, 0.00, 0.00, 1.00], hp=(229, 15), rpm_b=-200, temp_b=-2),
        "Picanto":  dict(ep=[0.85, 0.12, 0.02, 0.01], hp=(67,  10), rpm_b=-300, temp_b=-1),
    },
}

# Flat lookup used by Flask: brand → sorted list of model names
BRAND_MODEL_NAMES: dict[str, list[str]] = {
    brand: sorted(models.keys()) for brand, models in BRAND_MODELS.items()
}


def generate(n: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    brands_list  = list(BRAND_MODELS.keys())
    brand_arr    = rng.choice(brands_list, n)

    model_arr       = np.empty(n, dtype=object)
    engine_type_arr = np.empty(n, dtype=object)
    hp_arr          = np.zeros(n)
    rpm_bias_arr    = np.zeros(n)
    temp_bias_arr   = np.zeros(n)

    for i, b in enumerate(brand_arr):
        models   = BRAND_MODELS[b]
        m_names  = list(models.keys())
        m_name   = rng.choice(m_names)
        cfg      = models[m_name]

        model_arr[i]       = m_name
        engine_type_arr[i] = rng.choice(ENGINE_TYPES, p=cfg["ep"])
        hp_arr[i]          = rng.normal(*cfg["hp"])
        rpm_bias_arr[i]    = cfg["rpm_b"]
        temp_bias_arr[i]   = cfg["temp_b"]

    year    = rng.integers(2000, CURRENT_YEAR, n)
    age     = CURRENT_YEAR - year
    mileage = (rng.exponential(50_000, n) + age * 15_000).clip(0, 500_000).astype(int)
    hp_arr  = hp_arr.clip(60, 650).astype(int)

    deg = np.clip((mileage / 350_000) + (age / 30), 0, 1)

    rpm         = (rng.normal(2800, 800, n) + rpm_bias_arr).clip(600, 7000).astype(int)
    engine_temp = (rng.normal(88, 8, n) + deg * 30 + temp_bias_arr).clip(60, 150).round(1)
    oil_pressure= (rng.normal(4.0, 0.8, n) - deg * 2.0).clip(0.5, 7.0).round(2)
    coolant_temp= (engine_temp + rng.normal(0, 3, n)).clip(55, 150).round(1)
    fuel_pressure=(rng.normal(10, 2, n) - deg * 2.5).clip(2, 20).round(2)
    battery_voltage=(rng.normal(13.8, 0.4, n) - deg * 0.8).clip(10.5, 15.0).round(2)

    # ── Health scoring ────────────────────────────────────────────────────
    score = np.zeros(n, dtype=int)
    score += np.where(engine_temp > 125, 4, np.where(engine_temp > 110, 3,
             np.where(engine_temp > 100, 2, np.where(engine_temp > 95,  1, 0))))
    score += np.where(oil_pressure < 1.0, 4, np.where(oil_pressure < 1.5, 3,
             np.where(oil_pressure < 2.0, 2, np.where(oil_pressure < 2.5, 1, 0))))
    score += np.where(mileage > 350_000, 3, np.where(mileage > 250_000, 2,
             np.where(mileage > 150_000, 1, 0)))
    score += np.where(battery_voltage < 11.5, 3, np.where(battery_voltage < 12.0, 2,
             np.where(battery_voltage < 12.8, 1, 0)))
    score += np.where(fuel_pressure < 3, 3, np.where(fuel_pressure < 5, 2,
             np.where(fuel_pressure < 7, 1, 0)))
    score += np.where(rpm > 5500, 2, np.where(rpm > 5000, 1, 0))
    score += np.where(age > 20, 2, np.where(age > 15, 1, 0))
    score += rng.integers(0, 2, n)

    health = np.where(score >= 8, 'Poor', np.where(score >= 4, 'Fair', 'Good'))

    return pd.DataFrame({
        'brand':           brand_arr,
        'model':           model_arr,
        'engine_type':     engine_type_arr,
        'year':            year,
        'horsepower':      hp_arr,
        'mileage':         mileage,
        'rpm':             rpm,
        'engine_temp':     engine_temp,
        'oil_pressure':    oil_pressure,
        'coolant_temp':    coolant_temp,
        'fuel_pressure':   fuel_pressure,
        'battery_voltage': battery_voltage,
        'engine_health':   health,
    })


def main():
    parser = argparse.ArgumentParser(description="Generate car engine diagnostic data.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out",  type=str, default="model/data/processed/")
    args = parser.parse_args()

    print(f"Generating {args.rows:,} rows (seed={args.seed})…")
    df = generate(args.rows, args.seed)

    print("\n── Class Distribution ──────────────────────────────────")
    dist = df['engine_health'].value_counts()
    for label, cnt in dist.items():
        bar = '█' * int(cnt / args.rows * 40)
        print(f"  {label:<5}  {cnt:>5} ({cnt/args.rows*100:5.1f}%)  {bar}")

    print("\n── Sensor Statistics ───────────────────────────────────")
    num_cols = ['engine_temp', 'oil_pressure', 'coolant_temp',
                'fuel_pressure', 'battery_voltage', 'mileage', 'rpm']
    print(df[num_cols].describe().round(2).to_string())

    print("\n── Models per Brand ────────────────────────────────────")
    print(df.groupby('brand')['model'].value_counts().to_string())

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train_data.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✅  Saved {len(df):,} rows → {out_path.resolve()}")


if __name__ == "__main__":
    main()