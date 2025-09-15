import geopandas as gpd
from rest_api.db.accessors.raw_transaction_stats_accessors import get_raw_transaction_stats
from rest_api.db.accessors.raw_transaction_data_accessors import get_transaction_data

gdf = gpd.read_file(r"C:\Users\pc\OneDrive\Documents\Housing Market Project\source\data/Online_ONS_Postcode_Directory_Live_-48057019277614511.gpkg")

def get_heatmap(level: str, code: str, min_date: str | None = None, max_date: str | None = None):
    """
    Returns GeoJSON + stats for a given geography level:
      - district (e.g. SE1)
      - sector (e.g. SE1 8)
      - unit (e.g. SE1 8XX)
    """

    if level == "district":
        # Parent polygon
        district = gdf[gdf["pcd_district"] == code]
        district_stats = get_raw_transaction_stats(geography=code, min_date=min_date, max_date=max_date)
        district = district.merge(district_stats, left_on="pcd_district", right_on="district", how="left")

        # Children = all sectors inside
        children = gdf[gdf["pcd_sector"].str.startswith(code)]
        sector_stats = get_raw_transaction_stats(geography=list(children["pcd_sector"].unique()),
                                                 min_date=min_date, max_date=max_date)
        children = children.merge(sector_stats, left_on="pcd_sector", right_on="sector", how="left")

        return {
            "district": district.to_json(),
            "sectors": children.to_json()
        }

    elif level == "sector":
        # Parent polygon
        sector = gdf[gdf["pcd_sector"] == code]
        sector_stats = get_raw_transaction_stats(geography=code, min_date=min_date, max_date=max_date)
        sector = sector.merge(sector_stats, left_on="pcd_sector", right_on="sector", how="left")

        # Children = all units (full postcodes) inside
        children = gdf[gdf["pcd"].str.startswith(code)]
        unit_stats = get_raw_transaction_stats(geography=list(children["pcd"].unique()),
                                               min_date=min_date, max_date=max_date)
        children = children.merge(unit_stats, left_on="pcd", right_on="unit", how="left")

        return {
            "sector": sector.to_json(),
            "units": children.to_json()
        }

    elif level == "unit":
        # Single postcode polygon
        unit = gdf[gdf["pcd"] == code]
        unit_stats = get_raw_transaction_stats(geography=code, min_date=min_date, max_date=max_date)
        unit = unit.merge(unit_stats, left_on="pcd", right_on="unit", how="left")

        # Transactions (points) inside this postcode
        df = get_transaction_data(field="postcode", value=code, min_date=min_date)
        features = [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [row["lng"], row["lat"]]},
                "properties": {"price": row["price"], "date": str(row["date"]), "address": row["address"]}
            }
            for _, row in df.iterrows()
        ]

        return {
            "unit": unit.to_json(),
            "flats": {"type": "FeatureCollection", "features": features}
        }

    return {"error": "Invalid level"}

get_heatmap('sector', 'SE1')