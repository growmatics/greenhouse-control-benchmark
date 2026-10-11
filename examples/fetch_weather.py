"""Fetch real weather (+ optional ENTSO-E prices) into GreenLight-Gym CSVs.

One command, one file per year, ready for the sim. Whole years are the
benchmark's generalisation axis, so fetch several and hold some out.

    # weather only (free, no key):
    python examples/fetch_weather.py --lat 52.37 --lon 4.90 --years 2019-2023 \
        --location Amsterdam_OM --weather-dir gl_gym/environments/weather

    # with day-ahead prices (free ENTSO-E token; pip install entsoe-py):
    python examples/fetch_weather.py --lat 52.37 --lon 4.90 --years 2022,2023 \
        --location Amsterdam_OM --entsoe-token $ENTSOE_TOKEN --entsoe-zone NL

Then point the env at it: weather_data_dir=<weather-dir>, location=<location>,
growthYear=<year>.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from greenlight_bench.data import build_weather_year, write_weather_csv
from greenlight_bench.data.fetch import fetch_year_raw, parse_years


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--years", required=True, help="e.g. 2019-2023 or 2020,2022")
    parser.add_argument("--location", default="Custom_Location")
    parser.add_argument("--weather-dir", default="gl_gym/environments/weather")
    parser.add_argument("--timezone", default="UTC")
    parser.add_argument("--entsoe-token", default=os.environ.get("ENTSOE_TOKEN"))
    parser.add_argument("--entsoe-zone", default=None, help="ENTSO-E bidding zone, e.g. NL")
    args = parser.parse_args()

    for year in parse_years(args.years):
        raw = fetch_year_raw(
            args.lat,
            args.lon,
            year,
            timezone=args.timezone,
            entsoe_token=args.entsoe_token,
            entsoe_zone=args.entsoe_zone,
        )
        frame = build_weather_year(raw, year)
        out = Path(args.weather_dir) / args.location / f"{year}.csv"
        write_weather_csv(frame, out)
        priced = "electricity_price" in frame.columns
        print(f"{year}: {len(frame):,} rows (prices: {priced}) -> {out}")


if __name__ == "__main__":
    main()
