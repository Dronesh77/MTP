from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from service_network_analysis import (
    CATEGORY_WEIGHTS,
    compute_district_metrics,
    compute_point_level_metrics,
    flag_suspected_geocode_outliers,
    haversine_vectorized,
    load_and_prepare_data,
)


INPUT_FILE = Path("final_with_coordinates_better.csv")
OUTPUT_DIR = Path("traffic_aware_outputs")


DISTRICT_TRAFFIC_FACTOR = {
    "Howrah": 1.45,
    "Hooghly": 1.25,
    "Nadia": 1.20,
    "South 24 Parganas": 1.30,
    "Murshidabad": 1.18,
    "Darjeeling": 1.28,
    "Kalimpong": 1.32,
    "Jalpaiguri": 1.14,
}


HILL_DISTRICTS = {"Darjeeling", "Kalimpong"}


SERVICE_TYPE_INDEX = {
    "Fuel Station": 1.0,
    "Food/Lodging": 0.9,
    "Vehicle Repair": 1.4,
    "Other": 0.85,
}


@dataclass
class OptimizationConfig:
    max_hubs: int = 5
    service_radius_km: float = 140.0
    candidate_pool_size: int = 120
    neighborhood_radius_km: float = 25.0
    alpha_time_per_km: float = 2.2
    lambda_distance: float = 1.0
    lambda_traffic: float = 0.55
    lambda_underserved: float = 9000.0
    lambda_capability_bonus: float = 40.0
    big_m_penalty: float = 1e6
    total_budget: float = 700.0


def safe_divide(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_demand_weights(
    df: pd.DataFrame,
    district_metrics: pd.DataFrame,
) -> pd.DataFrame:
    demand = df.reset_index(drop=True).copy()
    eps_lookup = district_metrics.set_index("District")["expansion_priority_score"]
    rank_lookup = district_metrics.set_index("District")["expansion_priority_rank"]

    demand["district_eps"] = demand["District"].map(eps_lookup).fillna(0.0)
    demand["district_rank"] = demand["District"].map(rank_lookup).fillna(rank_lookup.max())
    demand["category_weight"] = demand["service_category"].map(CATEGORY_WEIGHTS).fillna(0.9)

    demand["w_i"] = 1.0 + demand["district_eps"] + 0.40 * demand["category_weight"]
    demand["q_i"] = 6.0 + 4.0 * demand["district_eps"] + 2.5 * demand["category_weight"]
    return demand


def build_candidate_pool(
    demand: pd.DataFrame,
    district_metrics: pd.DataFrame,
    config: OptimizationConfig,
) -> pd.DataFrame:
    candidates = demand.copy()
    eps_lookup = district_metrics.set_index("District")["expansion_priority_score"]
    rank_lookup = district_metrics.set_index("District")["expansion_priority_rank"]

    candidates["district_eps"] = candidates["District"].map(eps_lookup).fillna(0.0)
    candidates["district_rank"] = candidates["District"].map(rank_lookup).fillna(rank_lookup.max())

    lat = candidates["Latitude"].to_numpy()
    lon = candidates["Longitude"].to_numpy()
    dist = haversine_vectorized(lat, lon, lat, lon)

    nearby_counts = (dist <= config.neighborhood_radius_km).sum(axis=1) - 1
    repair_neighborhood = np.zeros(len(candidates))
    food_neighborhood = np.zeros(len(candidates))
    fuel_neighborhood = np.zeros(len(candidates))

    categories = candidates["service_category"].to_numpy()
    for idx in range(len(candidates)):
        mask = dist[idx] <= config.neighborhood_radius_km
        repair_neighborhood[idx] = int(np.sum(categories[mask] == "Vehicle Repair"))
        food_neighborhood[idx] = int(np.sum(categories[mask] == "Food/Lodging"))
        fuel_neighborhood[idx] = int(np.sum(categories[mask] == "Fuel Station"))

    candidates["nearby_service_count"] = nearby_counts
    candidates["nearby_repair_count"] = repair_neighborhood
    candidates["nearby_food_count"] = food_neighborhood
    candidates["nearby_fuel_count"] = fuel_neighborhood
    candidates["service_capability_score"] = (
        (repair_neighborhood > 0).astype(int) * 1.5
        + (food_neighborhood > 0).astype(int) * 1.0
        + (fuel_neighborhood > 0).astype(int) * 1.0
        + 0.05 * nearby_counts
    )

    base_cost = 82.0
    candidates["fixed_cost"] = (
        base_cost
        + 18.0 * candidates["district_eps"]
        + 8.0 * candidates["category_weight"]
        + 0.12 * candidates["nearby_service_count"]
    )
    candidates["capacity"] = (
        2200.0
        + 18.0 * candidates["nearby_service_count"]
        + 55.0 * candidates["service_capability_score"]
    )
    candidates["min_utilization"] = 0.20 * candidates["capacity"]

    candidates["candidate_score"] = (
        2.2 * candidates["district_eps"]
        + 0.45 * candidates["category_weight"]
        + 0.20 * candidates["service_capability_score"]
        + 0.02 * candidates["nearby_service_count"]
    )
    candidates = candidates.sort_values(
        ["district_rank", "candidate_score", "service_capability_score"],
        ascending=[True, False, False],
    )
    candidates = candidates.drop_duplicates(subset=["District", "Latitude", "Longitude", "Service"])

    district_balanced = (
        candidates.groupby("District", group_keys=False)
        .head(8)
        .copy()
    )
    remaining = candidates.loc[~candidates.index.isin(district_balanced.index)].copy()
    remaining = remaining.sort_values(
        ["candidate_score", "service_capability_score"],
        ascending=[False, False],
    )

    if len(district_balanced) < config.candidate_pool_size:
        needed = config.candidate_pool_size - len(district_balanced)
        district_balanced = pd.concat([district_balanced, remaining.head(needed)], ignore_index=True)

    return district_balanced.head(config.candidate_pool_size).reset_index(drop=True)


def compute_traffic_multiplier(
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
    distance_matrix: np.ndarray,
) -> np.ndarray:
    origin_base = demand["District"].map(DISTRICT_TRAFFIC_FACTOR).fillna(1.08).to_numpy()
    destination_base = candidates["District"].map(DISTRICT_TRAFFIC_FACTOR).fillna(1.08).to_numpy()

    traffic = (origin_base[None, :] + destination_base[:, None]) / 2.0
    traffic += np.where(distance_matrix > 60.0, 0.18, 0.0)
    traffic += np.where(distance_matrix > 120.0, 0.25, 0.0)

    hill_origin = demand["District"].isin(HILL_DISTRICTS).to_numpy()
    hill_destination = candidates["District"].isin(HILL_DISTRICTS).to_numpy()
    hill_adjustment = np.logical_or(hill_destination[:, None], hill_origin[None, :]).astype(float) * 0.22

    return traffic + hill_adjustment


def build_cost_matrices(
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
    config: OptimizationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distance_matrix = haversine_vectorized(
        candidates["Latitude"].to_numpy(),
        candidates["Longitude"].to_numpy(),
        demand["Latitude"].to_numpy(),
        demand["Longitude"].to_numpy(),
    )
    traffic_multiplier = compute_traffic_multiplier(demand, candidates, distance_matrix)
    travel_time = config.alpha_time_per_km * distance_matrix * traffic_multiplier
    objective_cost = (
        config.lambda_distance * distance_matrix
        + config.lambda_traffic * travel_time
    )

    infeasible_mask = distance_matrix > config.service_radius_km
    objective_cost = np.where(infeasible_mask, config.big_m_penalty, objective_cost)
    travel_time = np.where(infeasible_mask, np.nan, travel_time)
    return distance_matrix, travel_time, objective_cost


def assign_demand_to_selected_hubs(
    selected_indices: list[int],
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
    distance_matrix: np.ndarray,
    travel_time: np.ndarray,
    objective_cost: np.ndarray,
    config: OptimizationConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if not selected_indices:
        penalties = {
            "objective": config.big_m_penalty,
            "distance_cost": config.big_m_penalty,
            "traffic_cost": config.big_m_penalty,
            "underserved_penalty": config.big_m_penalty,
            "capability_bonus": 0.0,
            "unassigned_count": float(len(demand)),
        }
        return pd.DataFrame(), penalties

    selected = candidates.iloc[selected_indices].reset_index(names="candidate_original_index")
    cost_sub = objective_cost[selected_indices]
    distance_sub = distance_matrix[selected_indices]
    time_sub = travel_time[selected_indices]

    remaining_capacity = selected["capacity"].to_numpy(dtype=float).copy()
    assignment_rows: list[dict[str, float | int | str]] = []

    demand_order = (
        demand.assign(sort_key=demand["q_i"] * demand["w_i"])
        .sort_values("sort_key", ascending=False)
        .index
        .to_list()
    )

    for demand_idx in demand_order:
        demand_qty = float(demand.loc[demand_idx, "q_i"])
        feasible = np.where(cost_sub[:, demand_idx] < config.big_m_penalty)[0]
        if len(feasible) == 0:
            assignment_rows.append(
                {
                    "demand_index": demand_idx,
                    "candidate_index": -1,
                    "assigned": 0,
                    "q_i": demand_qty,
                    "distance_km": np.nan,
                    "travel_time_minutes": np.nan,
                    "assignment_cost": config.big_m_penalty,
                }
            )
            continue

        ranked = sorted(
            feasible,
            key=lambda idx: (
                cost_sub[idx, demand_idx],
                -remaining_capacity[idx],
            ),
        )

        chosen_idx = None
        for idx in ranked:
            if remaining_capacity[idx] >= demand_qty:
                chosen_idx = idx
                break

        if chosen_idx is None:
            chosen_idx = max(ranked, key=lambda idx: remaining_capacity[idx])

        remaining_capacity[chosen_idx] -= demand_qty
        assigned = int(remaining_capacity[chosen_idx] >= -1e-9)
        overflow_penalty = 0.0 if assigned else config.big_m_penalty / 100.0
        assignment_rows.append(
            {
                "demand_index": demand_idx,
                "candidate_index": int(selected.loc[chosen_idx, "candidate_original_index"]),
                "assigned": assigned,
                "q_i": demand_qty,
                "distance_km": float(distance_sub[chosen_idx, demand_idx]),
                "travel_time_minutes": float(time_sub[chosen_idx, demand_idx]),
                "assignment_cost": float(cost_sub[chosen_idx, demand_idx] * demand_qty + overflow_penalty),
            }
        )

    assignments = pd.DataFrame(assignment_rows)
    assignments = assignments.merge(
        demand.reset_index(names="demand_index")[
            ["demand_index", "District", "Service", "service_category", "w_i", "q_i"]
        ],
        on=["demand_index", "q_i"],
        how="left",
    )
    assignments = assignments.merge(
        selected.rename(
            columns={
                "District": "hub_district",
                "Service": "hub_service",
                "service_category": "hub_category",
            }
        )[
            [
                "candidate_original_index",
                "hub_district",
                "hub_service",
                "hub_category",
                "capacity",
                "min_utilization",
                "fixed_cost",
                "service_capability_score",
            ]
        ],
        left_on="candidate_index",
        right_on="candidate_original_index",
        how="left",
    )

    selected_districts = set(selected["District"])
    district_penalty = demand[["District", "district_eps"]].drop_duplicates()
    underserved_penalty = config.lambda_underserved * float(
        district_penalty.loc[~district_penalty["District"].isin(selected_districts), "district_eps"].sum()
    )

    hub_loads = assignments.groupby("candidate_index")["q_i"].sum().rename("assigned_load")
    selected = selected.merge(hub_loads, left_on="candidate_original_index", right_index=True, how="left")
    selected["assigned_load"] = selected["assigned_load"].fillna(0.0)
    utilization_penalty = float(
        np.maximum(selected["min_utilization"] - selected["assigned_load"], 0.0).sum() * 60.0
    )
    capability_bonus = config.lambda_capability_bonus * float(selected["service_capability_score"].sum())

    total_distance_cost = float((assignments["distance_km"] * assignments["q_i"]).fillna(0.0).sum())
    total_traffic_cost = float((assignments["travel_time_minutes"] * assignments["q_i"]).fillna(0.0).sum())
    assignment_cost = float(assignments["assignment_cost"].sum())
    fixed_cost_total = float(selected["fixed_cost"].sum())
    unassigned_count = float((assignments["assigned"] == 0).sum())
    objective = assignment_cost + underserved_penalty + utilization_penalty + fixed_cost_total - capability_bonus

    metrics = {
        "objective": objective,
        "distance_cost": total_distance_cost,
        "traffic_cost": total_traffic_cost,
        "underserved_penalty": underserved_penalty,
        "utilization_penalty": utilization_penalty,
        "capability_bonus": capability_bonus,
        "fixed_cost_total": fixed_cost_total,
        "unassigned_count": unassigned_count,
    }
    return assignments, metrics


def greedy_select_hubs(
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
    distance_matrix: np.ndarray,
    travel_time: np.ndarray,
    objective_cost: np.ndarray,
    config: OptimizationConfig,
) -> tuple[list[int], pd.DataFrame, dict[str, float]]:
    selected_indices: list[int] = []
    spent_budget = 0.0
    best_assignments = pd.DataFrame()
    best_metrics = {
        "objective": config.big_m_penalty,
    }

    for _ in range(config.max_hubs):
        round_number = len(selected_indices) + 1
        print(
            f"Selecting hub {round_number}/{config.max_hubs} | current budget used: {spent_budget:.2f}/{config.total_budget:.2f}",
            flush=True,
        )
        best_candidate = None
        round_best_assignments = None
        round_best_metrics = None
        evaluated_candidates = 0

        for candidate_idx in range(len(candidates)):
            if candidate_idx in selected_indices:
                continue
            candidate_cost = float(candidates.loc[candidate_idx, "fixed_cost"])
            if spent_budget + candidate_cost > config.total_budget:
                continue

            evaluated_candidates += 1
            if evaluated_candidates % 25 == 0:
                print(
                    f"  Round {round_number}: evaluated {evaluated_candidates} candidate hubs...",
                    flush=True,
                )

            trial_indices = selected_indices + [candidate_idx]
            trial_assignments, trial_metrics = assign_demand_to_selected_hubs(
                trial_indices,
                demand,
                candidates,
                distance_matrix,
                travel_time,
                objective_cost,
                config,
            )
            if (
                best_candidate is None
                or trial_metrics["objective"] < round_best_metrics["objective"]
            ):
                best_candidate = candidate_idx
                round_best_assignments = trial_assignments
                round_best_metrics = trial_metrics

        if best_candidate is None:
            print("  No additional feasible hub found within budget.", flush=True)
            break

        selected_indices.append(best_candidate)
        spent_budget += float(candidates.loc[best_candidate, "fixed_cost"])
        best_assignments = round_best_assignments
        best_metrics = round_best_metrics
        print(
            "  Selected hub:"
            f" {candidates.loc[best_candidate, 'District']} |"
            f" {candidates.loc[best_candidate, 'Service'][:70]}"
            f" | objective={best_metrics['objective']:.2f}",
            flush=True,
        )

    improved = True
    print("Starting local improvement search...", flush=True)
    improvement_checks = 0
    while improved:
        improved = False
        for pos, current_idx in enumerate(selected_indices.copy()):
            for replacement_idx in range(len(candidates)):
                if replacement_idx in selected_indices:
                    continue
                improvement_checks += 1
                if improvement_checks % 50 == 0:
                    print(
                        f"  Local improvement: checked {improvement_checks} swap options...",
                        flush=True,
                    )
                trial = selected_indices.copy()
                trial[pos] = replacement_idx
                trial_cost = float(candidates.iloc[trial]["fixed_cost"].sum())
                if trial_cost > config.total_budget:
                    continue
                trial_assignments, trial_metrics = assign_demand_to_selected_hubs(
                    trial,
                    demand,
                    candidates,
                    distance_matrix,
                    travel_time,
                    objective_cost,
                    config,
                )
                if trial_metrics["objective"] + 1e-9 < best_metrics["objective"]:
                    selected_indices = trial
                    best_assignments = trial_assignments
                    best_metrics = trial_metrics
                    improved = True
                    print(
                        "  Improvement found by swapping hub"
                        f" {current_idx} with {replacement_idx}."
                        f" New objective={best_metrics['objective']:.2f}",
                        flush=True,
                    )
                    break
            if improved:
                break

    return selected_indices, best_assignments, best_metrics


def build_selected_hub_summary(
    selected_indices: list[int],
    candidates: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    hubs = candidates.iloc[selected_indices].copy().reset_index(names="candidate_index")
    load_summary = (
        assignments.groupby("candidate_index")
        .agg(
            assigned_points=("demand_index", "count"),
            assigned_load=("q_i", "sum"),
            avg_distance_km=("distance_km", "mean"),
            avg_travel_time_minutes=("travel_time_minutes", "mean"),
        )
        .reset_index()
    )
    hubs = hubs.merge(load_summary, on="candidate_index", how="left")
    hubs["assigned_points"] = hubs["assigned_points"].fillna(0).astype(int)
    hubs["assigned_load"] = hubs["assigned_load"].fillna(0.0)
    hubs["capacity_utilization"] = hubs["assigned_load"] / hubs["capacity"]
    hubs["selected_rank"] = range(1, len(hubs) + 1)
    return hubs[
        [
            "selected_rank",
            "District",
            "Service",
            "service_category",
            "Latitude",
            "Longitude",
            "district_eps",
            "fixed_cost",
            "capacity",
            "min_utilization",
            "assigned_points",
            "assigned_load",
            "capacity_utilization",
            "avg_distance_km",
            "avg_travel_time_minutes",
            "service_capability_score",
        ]
    ]


def write_summary(
    demand: pd.DataFrame,
    selected_hubs: pd.DataFrame,
    metrics: dict[str, float],
) -> None:
    lines = [
        "# Traffic-Aware Hub Optimization Summary",
        "",
        f"- Demand nodes used: {len(demand)}",
        f"- Selected hubs: {len(selected_hubs)}",
        f"- Objective value: {metrics['objective']:.2f}",
        f"- Distance term: {metrics['distance_cost']:.2f}",
        f"- Traffic-delay term: {metrics['traffic_cost']:.2f}",
        f"- Underserved-district penalty: {metrics['underserved_penalty']:.2f}",
        f"- Utilization penalty: {metrics.get('utilization_penalty', 0.0):.2f}",
        f"- Fixed-cost total: {metrics.get('fixed_cost_total', 0.0):.2f}",
        f"- Unassigned demand nodes: {metrics['unassigned_count']:.0f}",
        "",
        "## Selected hub anchors",
    ]

    columns = [
        "selected_rank",
        "District",
        "Service",
        "service_category",
        "assigned_points",
        "assigned_load",
        "capacity_utilization",
        "avg_distance_km",
        "avg_travel_time_minutes",
    ]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines.extend([header, separator])
    for _, row in selected_hubs[columns].iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")

    (OUTPUT_DIR / "optimization_summary.md").write_text("\n".join(lines), encoding="utf-8")


def create_visualizations(
    demand: pd.DataFrame,
    assignments: pd.DataFrame,
    selected_hubs: pd.DataFrame,
) -> None:
    assignment_map = demand.reset_index(names="demand_index").merge(
        assignments[["demand_index", "hub_district", "hub_service", "distance_km", "travel_time_minutes"]],
        on="demand_index",
        how="left",
    )
    assignment_map["assigned_hub"] = (
        assignment_map["hub_district"].fillna("Unassigned")
        + " | "
        + assignment_map["hub_service"].fillna("None")
    )

    fig = px.scatter_mapbox(
        assignment_map,
        lat="Latitude",
        lon="Longitude",
        color="assigned_hub",
        hover_name="Service",
        hover_data=["District", "service_category", "distance_km", "travel_time_minutes"],
        zoom=5.6,
        height=760,
        title="Traffic-aware assignment of demand nodes to selected hubs",
    )
    fig.update_layout(mapbox_style="open-street-map")
    fig.write_html(OUTPUT_DIR / "traffic_aware_assignments_map.html")

    hub_map = go.Figure()
    hub_map.add_trace(
        go.Scattermapbox(
            lat=demand["Latitude"],
            lon=demand["Longitude"],
            mode="markers",
            marker={"size": 7, "color": "rgba(60,60,60,0.38)"},
            text=demand["Service"],
            name="Demand nodes",
            hovertemplate="%{text}<extra></extra>",
        )
    )
    hub_map.add_trace(
        go.Scattermapbox(
            lat=selected_hubs["Latitude"],
            lon=selected_hubs["Longitude"],
            mode="markers+text",
            marker={"size": 16, "color": "#d32f2f"},
            text=selected_hubs["selected_rank"].astype(str),
            textposition="top right",
            customdata=selected_hubs[["District", "Service", "capacity_utilization"]].to_numpy(),
            name="Selected hubs",
            hovertemplate=(
                "Rank %{text}<br>%{customdata[0]}<br>%{customdata[1]}"
                "<br>Utilization %{customdata[2]:.2f}<extra></extra>"
            ),
        )
    )
    hub_map.update_layout(
        mapbox_style="open-street-map",
        mapbox={"center": {"lat": float(demand["Latitude"].mean()), "lon": float(demand["Longitude"].mean())}, "zoom": 5.6},
        height=760,
        title="Selected traffic-aware integrated service hubs",
    )
    hub_map.write_html(OUTPUT_DIR / "traffic_aware_selected_hubs_map.html")


def main() -> None:
    config = OptimizationConfig()
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("Loading and validating location data...", flush=True)

    df = load_and_prepare_data(INPUT_FILE)
    df, _ = compute_point_level_metrics(df)
    df = flag_suspected_geocode_outliers(df)
    validated = df.loc[~df["suspected_geocode_outlier"]].copy().reset_index(drop=True)
    print(
        f"Loaded {len(df)} total rows | {len(validated)} validated rows after outlier filtering.",
        flush=True,
    )

    print("Computing district metrics and demand weights...", flush=True)
    district_metrics = compute_district_metrics(validated)
    demand = compute_demand_weights(validated, district_metrics)
    print("Building candidate hub pool...", flush=True)
    candidates = build_candidate_pool(demand, district_metrics, config)
    print(f"Candidate hub pool size: {len(candidates)}", flush=True)
    print("Building distance, travel-time, and traffic-cost matrices...", flush=True)
    distance_matrix, travel_time, objective_cost = build_cost_matrices(demand, candidates, config)
    print("Running greedy traffic-aware hub optimization...", flush=True)

    selected_indices, assignments, metrics = greedy_select_hubs(
        demand,
        candidates,
        distance_matrix,
        travel_time,
        objective_cost,
        config,
    )
    print("Preparing output tables and maps...", flush=True)
    selected_hubs = build_selected_hub_summary(selected_indices, candidates, assignments)

    demand.to_csv(OUTPUT_DIR / "traffic_demand_nodes.csv", index=False)
    candidates.to_csv(OUTPUT_DIR / "candidate_hubs.csv", index=False)
    assignments.to_csv(OUTPUT_DIR / "hub_assignments.csv", index=False)
    selected_hubs.to_csv(OUTPUT_DIR / "selected_hubs.csv", index=False)
    district_metrics.to_csv(OUTPUT_DIR / "district_metrics.csv", index=False)
    write_summary(demand, selected_hubs, metrics)
    create_visualizations(demand, assignments, selected_hubs)

    print(f"Traffic-aware optimization completed. Results saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
