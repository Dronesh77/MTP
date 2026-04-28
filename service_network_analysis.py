from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


EARTH_RADIUS_KM = 6371.0
OUTPUT_DIR = Path("masters_outputs")
INPUT_FILE = Path("final_with_coordinates_better.csv")


DISTRICT_RENAME_MAP = {
    "S24": "South 24 Parganas",
}


CATEGORY_WEIGHTS = {
    "Vehicle Repair": 1.40,
    "Fuel Station": 1.00,
    "Food/Lodging": 0.85,
    "Other": 0.90,
}


@dataclass
class AnalysisConfig:
    num_new_hubs: int = 5
    service_radius_km: float = 45.0
    hotspot_grid_deg: float = 0.30


def haversine_vectorized(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    lat1_rad = np.radians(lat1)[:, None]
    lon1_rad = np.radians(lon1)[:, None]
    lat2_rad = np.radians(lat2)[None, :]
    lon2_rad = np.radians(lon2)[None, :]

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return EARTH_RADIUS_KM * c


def safe_minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    values = series.astype(float)
    span = values.max() - values.min()
    if span == 0:
        scaled = pd.Series(np.zeros(len(values)), index=values.index)
    else:
        scaled = (values - values.min()) / span
    return 1.0 - scaled if invert else scaled


def infer_service_category(service: str, full_address: str) -> str:
    text = f"{service} {full_address}".lower()

    fuel_terms = [
        "petrol",
        "fuel",
        "filling station",
        "service station",
        "indian oil",
        "bharat petroleum",
        "hp petrol",
        "pump",
        "oil",
        "iocl",
    ]
    food_terms = [
        "dhaba",
        "hotel",
        "restaurant",
        "food",
        "apyan",
        "inn",
        "bhoj",
    ]
    repair_terms = [
        "garage",
        "workshop",
        "auto repair",
        "repair",
        "tyre",
        "tire",
        "motor works",
        "automobile",
        "auto mobile",
        "motors",
        "auto service",
        "service centre",
        "service center",
        "auto parts",
    ]

    if any(term in text for term in repair_terms):
        return "Vehicle Repair"
    if any(term in text for term in fuel_terms):
        return "Fuel Station"
    if any(term in text for term in food_terms):
        return "Food/Lodging"
    return "Other"


def load_and_prepare_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["District"] = df["District"].replace(DISTRICT_RENAME_MAP)
    df["Service"] = df["Service"].astype(str).str.strip()
    df["full_address"] = df["full_address"].astype(str).str.strip()
    df["service_category"] = [
        infer_service_category(service, address)
        for service, address in zip(df["Service"], df["full_address"])
    ]
    return df


def compute_point_level_metrics(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    coordinates = df[["Latitude", "Longitude"]].to_numpy(dtype=float)
    distance_matrix = haversine_vectorized(
        coordinates[:, 0],
        coordinates[:, 1],
        coordinates[:, 0],
        coordinates[:, 1],
    )
    np.fill_diagonal(distance_matrix, np.inf)

    nearest_neighbor = distance_matrix.min(axis=1)
    covered_by_existing = (distance_matrix <= 15.0).sum(axis=1)

    df = df.copy()
    df["nearest_neighbor_km"] = nearest_neighbor
    df["local_support_count_15km"] = covered_by_existing
    return df, distance_matrix


def flag_suspected_geocode_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["district_center_distance_km"] = np.nan
    df["suspected_geocode_outlier"] = False

    for district, group in df.groupby("District"):
        lat = group["Latitude"].to_numpy(dtype=float)
        lon = group["Longitude"].to_numpy(dtype=float)
        center_lat = np.median(lat)
        center_lon = np.median(lon)
        distances = haversine_vectorized(
            lat,
            lon,
            np.array([center_lat]),
            np.array([center_lon]),
        ).reshape(-1)

        q1 = float(np.percentile(distances, 25))
        q3 = float(np.percentile(distances, 75))
        iqr = q3 - q1
        threshold = max(q3 + 3.0 * iqr, 45.0)

        df.loc[group.index, "district_center_distance_km"] = distances
        df.loc[group.index, "suspected_geocode_outlier"] = distances > threshold

    return df


def approximate_bbox_area_km2(group: pd.DataFrame) -> float:
    lat_min, lat_max = np.percentile(group["Latitude"], [5, 95])
    lon_min, lon_max = np.percentile(group["Longitude"], [5, 95])

    height_km = abs(lat_max - lat_min) * 111.0
    mean_lat = math.radians((lat_min + lat_max) / 2.0)
    width_km = abs(lon_max - lon_min) * 111.0 * math.cos(mean_lat)
    return max(height_km * width_km, 1.0)


def compute_district_metrics(df: pd.DataFrame) -> pd.DataFrame:
    base = (
        df.groupby("District")
        .agg(
            total_services=("Service", "size"),
            mean_nearest_neighbor_km=("nearest_neighbor_km", "mean"),
            median_nearest_neighbor_km=("nearest_neighbor_km", "median"),
            p90_nearest_neighbor_km=("nearest_neighbor_km", lambda s: float(np.percentile(s, 90))),
            latitude_mean=("Latitude", "mean"),
            longitude_mean=("Longitude", "mean"),
        )
        .reset_index()
    )

    category_counts = (
        pd.crosstab(df["District"], df["service_category"])
        .reindex(columns=["Fuel Station", "Food/Lodging", "Vehicle Repair", "Other"], fill_value=0)
        .reset_index()
    )

    metrics = base.merge(category_counts, on="District", how="left")

    areas = (
        df.groupby("District")
        .apply(approximate_bbox_area_km2, include_groups=False)
        .rename("bbox_area_km2")
        .reset_index()
    )
    metrics = metrics.merge(areas, on="District", how="left")

    metrics["service_density_per_1000km2"] = (
        metrics["total_services"] / metrics["bbox_area_km2"] * 1000.0
    )
    metrics["repair_share"] = metrics["Vehicle Repair"] / metrics["total_services"]
    metrics["fuel_share"] = metrics["Fuel Station"] / metrics["total_services"]
    metrics["food_share"] = metrics["Food/Lodging"] / metrics["total_services"]

    proportions = metrics[["Fuel Station", "Food/Lodging", "Vehicle Repair", "Other"]].div(
        metrics["total_services"], axis=0
    )
    entropy = -(proportions.replace(0, np.nan) * np.log(proportions.replace(0, np.nan))).sum(axis=1)
    metrics["category_diversity"] = entropy / math.log(4)

    metrics["density_gap_score"] = safe_minmax(metrics["service_density_per_1000km2"], invert=True)
    metrics["repair_gap_score"] = safe_minmax(metrics["repair_share"], invert=True)
    metrics["dispersion_score"] = safe_minmax(metrics["mean_nearest_neighbor_km"])
    metrics["diversity_gap_score"] = safe_minmax(metrics["category_diversity"], invert=True)

    metrics["expansion_priority_score"] = (
        0.35 * metrics["density_gap_score"]
        + 0.25 * metrics["repair_gap_score"]
        + 0.20 * metrics["dispersion_score"]
        + 0.20 * metrics["diversity_gap_score"]
    )
    metrics["expansion_priority_rank"] = (
        metrics["expansion_priority_score"].rank(ascending=False, method="dense").astype(int)
    )

    return metrics.sort_values(
        ["expansion_priority_score", "mean_nearest_neighbor_km"],
        ascending=[False, False],
    )


def compute_hotspots(df: pd.DataFrame, grid_deg: float) -> pd.DataFrame:
    hotspot_df = df.copy()
    hotspot_df["grid_lat"] = (hotspot_df["Latitude"] / grid_deg).round() * grid_deg
    hotspot_df["grid_lon"] = (hotspot_df["Longitude"] / grid_deg).round() * grid_deg

    hotspot_summary = (
        hotspot_df.groupby(["grid_lat", "grid_lon"])
        .agg(
            total_services=("Service", "size"),
            districts=("District", lambda s: ", ".join(sorted(set(s)))),
            dominant_category=("service_category", lambda s: s.value_counts().idxmax()),
            avg_latitude=("Latitude", "mean"),
            avg_longitude=("Longitude", "mean"),
        )
        .reset_index()
        .sort_values("total_services", ascending=False)
    )
    return hotspot_summary


def choose_candidate_rows(df: pd.DataFrame, district_metrics: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rank_lookup = district_metrics.set_index("District")["expansion_priority_rank"]
    score_lookup = district_metrics.set_index("District")["expansion_priority_score"]

    df["district_priority_rank"] = df["District"].map(rank_lookup)
    df["district_priority_score"] = df["District"].map(score_lookup)
    df["category_weight"] = df["service_category"].map(CATEGORY_WEIGHTS).fillna(0.9)

    top_districts = district_metrics.nsmallest(999, "expansion_priority_rank").head(10)["District"]
    candidates = df[df["District"].isin(top_districts)].copy()

    district_centroids = (
        candidates.groupby("District")[["Latitude", "Longitude"]].mean().rename(
            columns={"Latitude": "district_lat", "Longitude": "district_lon"}
        )
    )
    candidates = candidates.join(district_centroids, on="District")

    centroid_distance = haversine_vectorized(
        candidates["Latitude"].to_numpy(),
        candidates["Longitude"].to_numpy(),
        candidates["district_lat"].to_numpy(),
        candidates["district_lon"].to_numpy(),
    )
    candidates["distance_to_district_centroid_km"] = np.diag(centroid_distance)

    candidates = candidates.sort_values(
        [
            "district_priority_rank",
            "distance_to_district_centroid_km",
            "category_weight",
        ],
        ascending=[True, True, False],
    )
    return candidates


def greedy_hub_selection(
    df: pd.DataFrame,
    district_metrics: pd.DataFrame,
    config: AnalysisConfig,
) -> pd.DataFrame:
    candidates = choose_candidate_rows(df, district_metrics).reset_index(drop=True)
    demand = df.reset_index(drop=True).copy()

    district_priority = district_metrics.set_index("District")["expansion_priority_score"]
    demand["demand_weight"] = (
        1.0
        + demand["District"].map(district_priority).fillna(0.0)
        + 0.35 * demand["service_category"].map(CATEGORY_WEIGHTS).fillna(0.9)
    )

    candidate_dist = haversine_vectorized(
        candidates["Latitude"].to_numpy(),
        candidates["Longitude"].to_numpy(),
        demand["Latitude"].to_numpy(),
        demand["Longitude"].to_numpy(),
    )

    best_distance = np.full(len(demand), np.inf)
    selected_indices: list[int] = []

    unique_candidate_rows = candidates.drop_duplicates(
        subset=["District", "Latitude", "Longitude", "Service"]
    ).reset_index(drop=True)
    candidate_dist = haversine_vectorized(
        unique_candidate_rows["Latitude"].to_numpy(),
        unique_candidate_rows["Longitude"].to_numpy(),
        demand["Latitude"].to_numpy(),
        demand["Longitude"].to_numpy(),
    )

    for _ in range(config.num_new_hubs):
        best_idx = None
        best_objective = np.inf

        for candidate_idx in range(len(unique_candidate_rows)):
            if candidate_idx in selected_indices:
                continue
            updated = np.minimum(best_distance, candidate_dist[candidate_idx])
            penalized = np.where(updated <= config.service_radius_km, updated, updated * 1.5)
            objective = float(np.sum(penalized * demand["demand_weight"].to_numpy()))
            if objective < best_objective:
                best_objective = objective
                best_idx = candidate_idx

        if best_idx is None:
            break

        selected_indices.append(best_idx)
        best_distance = np.minimum(best_distance, candidate_dist[best_idx])

    selected = unique_candidate_rows.iloc[selected_indices].copy()
    selected["selected_hub_rank"] = range(1, len(selected) + 1)
    selected["estimated_covered_points"] = [
        int((candidate_dist[idx] <= config.service_radius_km).sum()) for idx in selected_indices
    ]
    selected["avg_distance_to_covered_points_km"] = [
        float(candidate_dist[idx][candidate_dist[idx] <= config.service_radius_km].mean())
        if (candidate_dist[idx] <= config.service_radius_km).any()
        else float("nan")
        for idx in selected_indices
    ]
    selected["hub_role"] = "Proposed integrated service hub"

    return selected[
        [
            "selected_hub_rank",
            "District",
            "Service",
            "service_category",
            "Latitude",
            "Longitude",
            "district_priority_rank",
            "district_priority_score",
            "estimated_covered_points",
            "avg_distance_to_covered_points_km",
            "hub_role",
        ]
    ]


def write_summary(
    df: pd.DataFrame,
    validated_df: pd.DataFrame,
    district_metrics: pd.DataFrame,
    hotspots: pd.DataFrame,
    hubs: pd.DataFrame,
) -> None:
    def frame_to_markdown(frame: pd.DataFrame) -> str:
        columns = list(frame.columns)
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        rows = []
        for _, row in frame.iterrows():
            cells = [str(row[col]) for col in columns]
            rows.append("| " + " | ".join(cells) + " |")
        return "\n".join([header, separator] + rows)

    top_priority = district_metrics.head(5)[
        [
            "District",
            "expansion_priority_score",
            "total_services",
            "Vehicle Repair",
            "service_density_per_1000km2",
            "mean_nearest_neighbor_km",
        ]
    ]
    top_hotspots = hotspots.head(5)[
        ["districts", "total_services", "dominant_category", "avg_latitude", "avg_longitude"]
    ]

    lines = [
        "# Statewide Automotive Service Network Analysis",
        "",
        f"- Total geocoded service points analysed: {len(df)}",
        f"- Validated points used for optimization: {len(validated_df)}",
        f"- Suspected geocoding outliers flagged: {int(df['suspected_geocode_outlier'].sum())}",
        f"- Districts covered: {df['District'].nunique()}",
        f"- Dominant category: {df['service_category'].value_counts().idxmax()}",
        "",
        "## Highest-priority districts for service expansion",
        frame_to_markdown(top_priority),
        "",
        "## Strongest service hotspots",
        frame_to_markdown(top_hotspots),
        "",
        "## Recommended new service hub anchors",
        frame_to_markdown(hubs),
        "",
        "## Interpretation",
        "- Higher expansion-priority scores indicate districts with low proxy service density, weaker repair presence, wider spatial dispersion, and lower category balance.",
        "- Proposed hubs are selected from real observed service nodes in high-priority districts and optimized with a greedy weighted-distance heuristic.",
        "- The current model is data-driven and reproducible, but it should later be enriched with road-network travel time, vehicle registration counts, and demand forecasts.",
    ]

    (OUTPUT_DIR / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def create_visualizations(
    df: pd.DataFrame,
    validated_df: pd.DataFrame,
    district_metrics: pd.DataFrame,
    hotspots: pd.DataFrame,
    hubs: pd.DataFrame,
) -> None:
    category_map = px.scatter_mapbox(
        validated_df,
        lat="Latitude",
        lon="Longitude",
        color="service_category",
        hover_name="Service",
        hover_data=["District", "nearest_neighbor_km", "local_support_count_15km"],
        zoom=5.6,
        height=720,
        title="Statewide service point distribution by inferred category",
    )
    category_map.update_layout(mapbox_style="open-street-map")
    category_map.write_html(OUTPUT_DIR / "statewide_service_categories_map.html")

    ranked = district_metrics.sort_values("expansion_priority_score", ascending=False)
    priority_chart = px.bar(
        ranked,
        x="District",
        y="expansion_priority_score",
        color="Vehicle Repair",
        hover_data=["total_services", "service_density_per_1000km2", "mean_nearest_neighbor_km"],
        title="District expansion priority score",
    )
    priority_chart.update_layout(xaxis_tickangle=-35)
    priority_chart.write_html(OUTPUT_DIR / "district_expansion_priority.html")

    hotspot_map = px.scatter_mapbox(
        hotspots.head(40),
        lat="avg_latitude",
        lon="avg_longitude",
        size="total_services",
        color="dominant_category",
        hover_data=["districts"],
        zoom=5.6,
        height=720,
        title="Top statewide service hotspots",
    )
    hotspot_map.update_layout(mapbox_style="open-street-map")
    hotspot_map.write_html(OUTPUT_DIR / "statewide_hotspots.html")

    recommendation_map = go.Figure()
    recommendation_map.add_trace(
        go.Scattermapbox(
            lat=validated_df["Latitude"],
            lon=validated_df["Longitude"],
            mode="markers",
            marker={"size": 7, "color": "rgba(80,80,80,0.45)"},
            text=validated_df["Service"],
            name="Observed service points",
            hovertemplate="%{text}<br>%{lat:.3f}, %{lon:.3f}<extra></extra>",
        )
    )
    recommendation_map.add_trace(
        go.Scattermapbox(
            lat=hubs["Latitude"],
            lon=hubs["Longitude"],
            mode="markers+text",
            marker={"size": 16, "color": "#c62828"},
            text=hubs["selected_hub_rank"].astype(str),
            textposition="top right",
            name="Recommended hubs",
            hovertemplate=(
                "Rank %{text}<br>%{customdata[0]}<br>%{customdata[1]}"
                "<br>Covered points: %{customdata[2]}<extra></extra>"
            ),
            customdata=hubs[["District", "Service", "estimated_covered_points"]].to_numpy(),
        )
    )
    recommendation_map.update_layout(
        mapbox_style="open-street-map",
        mapbox={"center": {"lat": float(validated_df["Latitude"].mean()), "lon": float(validated_df["Longitude"].mean())}, "zoom": 5.6},
        height=760,
        title="Recommended integrated service hub anchors",
    )
    recommendation_map.write_html(OUTPUT_DIR / "recommended_hubs_map.html")


def main() -> None:
    config = AnalysisConfig()
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_and_prepare_data(INPUT_FILE)
    df, _ = compute_point_level_metrics(df)
    df = flag_suspected_geocode_outliers(df)
    validated_df = df[~df["suspected_geocode_outlier"]].copy()

    district_metrics = compute_district_metrics(validated_df)
    hotspots = compute_hotspots(validated_df, config.hotspot_grid_deg)
    hubs = greedy_hub_selection(validated_df, district_metrics, config)

    df.to_csv(OUTPUT_DIR / "enriched_service_points.csv", index=False)
    validated_df.to_csv(OUTPUT_DIR / "validated_service_points.csv", index=False)
    district_metrics.to_csv(OUTPUT_DIR / "district_metrics.csv", index=False)
    hotspots.to_csv(OUTPUT_DIR / "service_hotspots.csv", index=False)
    hubs.to_csv(OUTPUT_DIR / "recommended_hubs.csv", index=False)

    write_summary(df, validated_df, district_metrics, hotspots, hubs)
    create_visualizations(df, validated_df, district_metrics, hotspots, hubs)

    print(f"Analysis completed. Results saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
