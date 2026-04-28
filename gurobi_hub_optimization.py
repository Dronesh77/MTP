from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd

from service_network_analysis import (
    compute_district_metrics,
    compute_point_level_metrics,
    flag_suspected_geocode_outliers,
    haversine_vectorized,
    load_and_prepare_data,
)
from traffic_aware_hub_optimization import (
    INPUT_FILE,
    OUTPUT_DIR as HEURISTIC_OUTPUT_DIR,
    OptimizationConfig,
    build_candidate_pool,
    build_cost_matrices,
    compute_demand_weights,
)


OUTPUT_DIR = Path("gurobi_outputs")


@dataclass
class GurobiConfig:
    max_hubs: int = 5
    candidate_pool_size: int = 30
    service_radius_km: float = 300.0
    total_budget: float = 700.0
    time_limit_seconds: int = 300
    mip_gap: float = 0.02
    threads: int = 0
    demand_grid_deg: float = 1.3


def aggregate_demand_nodes(demand: pd.DataFrame, grid_deg: float) -> pd.DataFrame:
    aggregated = demand.copy()
    aggregated["grid_lat"] = (aggregated["Latitude"] / grid_deg).round() * grid_deg
    aggregated["grid_lon"] = (aggregated["Longitude"] / grid_deg).round() * grid_deg

    grouped = (
        aggregated.groupby(["District", "grid_lat", "grid_lon"])
        .agg(
            Latitude=("Latitude", "mean"),
            Longitude=("Longitude", "mean"),
            q_i=("q_i", "sum"),
            w_i=("w_i", "mean"),
            point_count=("Service", "size"),
            dominant_category=("service_category", lambda s: s.value_counts().idxmax()),
            sample_service=("Service", "first"),
            district_eps=("district_eps", "first"),
        )
        .reset_index()
    )
    grouped["Service"] = grouped["sample_service"] + " [aggregated node]"
    grouped["service_category"] = grouped["dominant_category"]
    return grouped.drop(columns=["sample_service", "dominant_category"])


def prepare_inputs(config: GurobiConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    print("Loading and validating location data...", flush=True)
    df = load_and_prepare_data(INPUT_FILE)
    df, _ = compute_point_level_metrics(df)
    df = flag_suspected_geocode_outliers(df)
    validated = df.loc[~df["suspected_geocode_outlier"]].copy().reset_index(drop=True)
    print(f"Loaded {len(df)} total rows | {len(validated)} validated rows.", flush=True)

    district_metrics = compute_district_metrics(validated)
    raw_demand = compute_demand_weights(validated, district_metrics)
    demand = aggregate_demand_nodes(raw_demand, config.demand_grid_deg)
    print(f"Aggregated demand nodes for Gurobi: {len(demand)}", flush=True)

    heuristic_config = OptimizationConfig(
        max_hubs=config.max_hubs,
        service_radius_km=config.service_radius_km,
        candidate_pool_size=config.candidate_pool_size,
        total_budget=config.total_budget,
    )
    print("Building candidate hub pool...", flush=True)
    candidates = build_candidate_pool(raw_demand, district_metrics, heuristic_config)
    print(f"Candidate hub pool size: {len(candidates)}", flush=True)

    distance_matrix, travel_time, objective_cost = build_cost_matrices(
        demand,
        candidates,
        heuristic_config,
    )
    return demand, candidates, district_metrics, distance_matrix, travel_time, objective_cost


def build_capability_matrix(candidates: pd.DataFrame) -> dict[tuple[int, str], int]:
    capability_thresholds = {
        "Fuel Station": "nearby_fuel_count",
        "Food/Lodging": "nearby_food_count",
        "Vehicle Repair": "nearby_repair_count",
    }
    capability = {}
    for j, row in candidates.iterrows():
        for category, col in capability_thresholds.items():
            capability[(j, category)] = int(float(row[col]) > 0)
    return capability


def solve_model(
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
    district_metrics: pd.DataFrame,
    distance_matrix: np.ndarray,
    travel_time: np.ndarray,
    objective_cost: np.ndarray,
    config: GurobiConfig,
) -> tuple[gp.Model, dict[str, gp.tupledict]]:
    print("Building Gurobi model...", flush=True)
    model = gp.Model("traffic_aware_hub_location")
    model.Params.TimeLimit = config.time_limit_seconds
    model.Params.MIPGap = config.mip_gap
    if config.threads:
        model.Params.Threads = config.threads
    model.Params.OutputFlag = 1

    I = list(range(len(demand)))
    J = list(range(len(candidates)))
    districts = sorted(demand["District"].unique())
    district_to_candidates = {
        district: [j for j in J if candidates.loc[j, "District"] == district]
        for district in districts
    }
    service_types = ["Fuel Station", "Food/Lodging", "Vehicle Repair"]
    capability = build_capability_matrix(candidates)

    feasible_pairs = [
        (i, j)
        for j in J
        for i in I
        if objective_cost[j, i] < 1e6 and np.isfinite(travel_time[j, i])
    ]
    if not feasible_pairs:
        raise RuntimeError("No feasible demand-hub assignment pairs found.")

    print(
        f"Model dimensions | demand nodes: {len(I)} | candidate hubs: {len(J)} | feasible assignments: {len(feasible_pairs)}",
        flush=True,
    )

    x = model.addVars(J, vtype=GRB.BINARY, name="x")
    y = model.addVars(feasible_pairs, vtype=GRB.BINARY, name="y")
    z = model.addVars(districts, vtype=GRB.BINARY, name="z")
    s = model.addVars(
        [(j, k) for j in J for k in service_types],
        vtype=GRB.BINARY,
        name="s",
    )

    w = demand["w_i"].to_numpy(dtype=float)
    q = demand["q_i"].to_numpy(dtype=float)
    fixed_cost = candidates["fixed_cost"].to_numpy(dtype=float)
    capacity = candidates["capacity"].to_numpy(dtype=float)
    min_util = candidates["min_utilization"].to_numpy(dtype=float)
    eps_lookup = district_metrics.set_index("District")["expansion_priority_score"].to_dict()
    beta = {"Fuel Station": 1.0, "Food/Lodging": 1.0, "Vehicle Repair": 1.5}
    eta = 2.5
    lambda_distance = 1.0
    lambda_traffic = 0.55
    lambda_underserved = 9000.0

    weighted_distance = gp.quicksum(
        lambda_distance * w[i] * q[i] * distance_matrix[j, i] * y[i, j]
        for (i, j) in feasible_pairs
    )
    weighted_traffic = gp.quicksum(
        lambda_traffic * w[i] * q[i] * travel_time[j, i] * y[i, j]
        for (i, j) in feasible_pairs
    )
    underserved_penalty = gp.quicksum(
        lambda_underserved * eps_lookup[d] * z[d]
        for d in districts
    )

    model.setObjective(weighted_distance + weighted_traffic + underserved_penalty, GRB.MINIMIZE)

    print("Adding assignment constraints...", flush=True)
    feasible_by_demand: dict[int, list[int]] = {i: [] for i in I}
    feasible_by_hub: dict[int, list[int]] = {j: [] for j in J}
    for i, j in feasible_pairs:
        feasible_by_demand[i].append(j)
        feasible_by_hub[j].append(i)

    for i in I:
        if not feasible_by_demand[i]:
            raise RuntimeError(f"Demand node {i} has no feasible candidate hub within service radius.")
        model.addConstr(gp.quicksum(y[i, j] for j in feasible_by_demand[i]) == 1, name=f"assign_{i}")

    print("Adding hub opening and budget constraints...", flush=True)
    for i, j in feasible_pairs:
        model.addConstr(y[i, j] <= x[j], name=f"open_link_{i}_{j}")

    model.addConstr(gp.quicksum(x[j] for j in J) <= config.max_hubs, name="hub_limit")
    model.addConstr(gp.quicksum(fixed_cost[j] * x[j] for j in J) <= config.total_budget, name="budget")

    print("Adding capacity and utilization constraints...", flush=True)
    for j in J:
        assigned_load = gp.quicksum(q[i] * y[i, j] for i in feasible_by_hub[j])
        model.addConstr(assigned_load <= capacity[j] * x[j], name=f"capacity_{j}")
        model.addConstr(assigned_load >= min_util[j] * x[j], name=f"min_util_{j}")

    print("Adding service capability constraints...", flush=True)
    for j in J:
        for k in service_types:
            model.addConstr(s[j, k] <= capability[(j, k)] * x[j], name=f"cap_avail_{j}_{k}")
            model.addConstr(s[j, k] <= x[j], name=f"cap_open_{j}_{k}")
        model.addConstr(
            gp.quicksum(beta[k] * s[j, k] for k in service_types) >= eta * x[j],
            name=f"capability_min_{j}",
        )

    print("Adding underserved-district penalty constraints...", flush=True)
    for district in districts:
        cand_list = district_to_candidates[district]
        if cand_list:
            model.addConstr(
                z[district] >= 1 - gp.quicksum(x[j] for j in cand_list),
                name=f"underserved_{district}",
            )
        else:
            model.addConstr(z[district] == 1, name=f"underserved_{district}")

    print("Optimizing with Gurobi...", flush=True)
    model.optimize()

    return model, {"x": x, "y": y, "z": z, "s": s}


def extract_solution(
    model: gp.Model,
    variables: dict[str, gp.tupledict],
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
    district_metrics: pd.DataFrame,
    distance_matrix: np.ndarray,
    travel_time: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = variables["x"]
    y = variables["y"]
    z = variables["z"]

    if model.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL}:
        raise RuntimeError(f"Gurobi did not return a usable solution. Status code: {model.Status}")

    selected_indices = [j for j in range(len(candidates)) if x[j].X > 0.5]
    assignments = []
    for (i, j), var in y.items():
        if var.X > 0.5:
            assignments.append(
                {
                    "demand_index": i,
                    "candidate_index": j,
                    "demand_district": demand.loc[i, "District"],
                    "demand_service": demand.loc[i, "Service"],
                    "demand_category": demand.loc[i, "service_category"],
                    "hub_district": candidates.loc[j, "District"],
                    "hub_service": candidates.loc[j, "Service"],
                    "hub_category": candidates.loc[j, "service_category"],
                    "q_i": demand.loc[i, "q_i"],
                    "w_i": demand.loc[i, "w_i"],
                    "distance_km": float(distance_matrix[j, i]),
                    "travel_time_minutes": float(travel_time[j, i]),
                }
            )

    assignments_df = pd.DataFrame(assignments)
    if assignments_df.empty:
        raise RuntimeError("No assignments were extracted from the Gurobi solution.")

    selected_hubs = candidates.iloc[selected_indices].copy().reset_index(names="candidate_index")
    hub_loads = assignments_df.groupby("candidate_index")["q_i"].sum().rename("assigned_load")
    hub_points = assignments_df.groupby("candidate_index")["demand_index"].count().rename("assigned_points")
    avg_distance = assignments_df.groupby("candidate_index")["distance_km"].mean().rename("avg_distance_km")
    avg_time = assignments_df.groupby("candidate_index")["travel_time_minutes"].mean().rename("avg_travel_time_minutes")

    selected_hubs = selected_hubs.merge(hub_loads, on="candidate_index", how="left")
    selected_hubs = selected_hubs.merge(hub_points, on="candidate_index", how="left")
    selected_hubs = selected_hubs.merge(avg_distance, on="candidate_index", how="left")
    selected_hubs = selected_hubs.merge(avg_time, on="candidate_index", how="left")
    selected_hubs["capacity_utilization"] = selected_hubs["assigned_load"] / selected_hubs["capacity"]
    selected_hubs["selected_rank"] = range(1, len(selected_hubs) + 1)

    underserved = pd.DataFrame(
        {
            "District": sorted(demand["District"].unique()),
            "underserved_binary": [int(z[d].X > 0.5) for d in sorted(demand["District"].unique())],
        }
    ).merge(
        district_metrics[["District", "expansion_priority_score"]],
        on="District",
        how="left",
    )

    return selected_hubs, assignments_df, underserved


def write_outputs(
    model: gp.Model,
    demand: pd.DataFrame,
    candidates: pd.DataFrame,
    selected_hubs: pd.DataFrame,
    assignments_df: pd.DataFrame,
    underserved: pd.DataFrame,
    district_metrics: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    demand.to_csv(OUTPUT_DIR / "aggregated_demand_nodes.csv", index=False)
    candidates.to_csv(OUTPUT_DIR / "candidate_hubs.csv", index=False)
    selected_hubs.to_csv(OUTPUT_DIR / "selected_hubs.csv", index=False)
    assignments_df.to_csv(OUTPUT_DIR / "hub_assignments.csv", index=False)
    underserved.to_csv(OUTPUT_DIR / "underserved_districts.csv", index=False)
    district_metrics.to_csv(OUTPUT_DIR / "district_metrics.csv", index=False)

    lines = [
        "# Gurobi Traffic-Aware Hub Optimization Summary",
        "",
        f"- Solver status: {model.Status}",
        f"- Objective value: {model.ObjVal:.2f}" if model.SolCount else "- Objective value: no solution",
        f"- Best bound: {model.ObjBound:.2f}" if model.SolCount else "- Best bound: unavailable",
        f"- MIP gap: {model.MIPGap:.4f}" if model.SolCount else "- MIP gap: unavailable",
        f"- Selected hubs: {len(selected_hubs)}",
        f"- Assigned demand nodes: {len(assignments_df)}",
        "",
        "## Selected hubs",
    ]

    cols = [
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
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines.extend([header, separator])
    for _, row in selected_hubs[cols].iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")

    (OUTPUT_DIR / "optimization_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = GurobiConfig()
    demand, candidates, district_metrics, distance_matrix, travel_time, objective_cost = prepare_inputs(config)
    model, variables = solve_model(
        demand,
        candidates,
        district_metrics,
        distance_matrix,
        travel_time,
        objective_cost,
        config,
    )
    print("Extracting Gurobi solution...", flush=True)
    selected_hubs, assignments_df, underserved = extract_solution(
        model,
        variables,
        demand,
        candidates,
        district_metrics,
        distance_matrix,
        travel_time,
    )
    print("Writing outputs...", flush=True)
    write_outputs(
        model,
        demand,
        candidates,
        selected_hubs,
        assignments_df,
        underserved,
        district_metrics,
    )
    print(f"Gurobi optimization completed. Results saved in: {OUTPUT_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
