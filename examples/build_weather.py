"""CLI: build a GreenLight-Gym2 weather CSV from a raw weather/price table.

The input CSV needs a timestamp column plus ``radiation`` [W/m2], ``wind_speed``
[m/s], ``air_temp`` [degC], ``rh`` [%], and optionally ``elec_price`` [EUR/MWh].

    python examples/build_weather.py --input raw.csv --timestamp-col timestamp \
        --year 2026 --location Custom_Location \
        --weather-dir gl_gym/environments/weather

Then point the env at it: weather_data_dir=<weather-dir>, location=Custom_Location,
growthYear=2026.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from greenlight_bench.data import build_weather_year, write_weather_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="raw weather CSV")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--location", default="Custom_Location")
    parser.add_argument("--weather-dir", default="gl_gym/environments/weather")
    args = parser.parse_args()

    raw = pd.read_csv(args.input, parse_dates=[args.timestamp_col]).set_index(args.timestamp_col)
    frame = build_weather_year(raw, args.year)
    out = Path(args.weather_dir) / args.location / f"{args.year}.csv"
    write_weather_csv(frame, out)
    print(f"wrote {len(frame):,} rows -> {out}")


if __name__ == "__main__":
    main()
