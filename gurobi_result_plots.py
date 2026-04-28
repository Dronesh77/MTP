from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from gurobi_hub_optimization import GurobiConfig, prepare_inputs


OUTPUT_DIR = Path("gurobi_outputs")
PLOTS_DIR = OUTPUT_DIR / "plots"


def ensure_plot_dir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_or_rebuild_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregated_path = OUTPUT_DIR / "aggregated_demand_nodes.csv"
    candidate_path = OUTPUT_DIR / "candidate_hubs.csv"
    selected_path = OUTPUT_DIR / "selected_hubs.csv"
    assignments_path = OUTPUT_DIR / "hub_assignments.csv"
    district_metrics_path = OUTPUT_DIR / "district_metrics.csv"
    underserved_path = OUTPUT_DIR / "underserved_districts.csv"

    if all(path.exists() for path in [aggregated_path, candidate_path, selected_path, assignments_path, district_metrics_path, underserved_path]):
        demand = pd.read_csv(aggregated_path)
        candidates = pd.read_csv(candidate_path)
        selected_hubs = pd.read_csv(selected_path)
        assignments = pd.read_csv(assignments_path)
        district_metrics = pd.read_csv(district_metrics_path)
        underserved = pd.read_csv(underserved_path)
        district_metrics = district_metrics.merge(underserved, on="District", how="left", suffixes=("", "_underserved"))
        return demand, candidates, selected_hubs, assignments, district_metrics

    print("Rebuilding aggregated inputs because saved demand-node files were not found...", flush=True)
    demand, candidates, district_metrics, _, _, _ = prepare_inputs(GurobiConfig())
    selected_hubs = pd.read_csv(selected_path)
    assignments = pd.read_csv(assignments_path)
    underserved = pd.read_csv(underserved_path)
    district_metrics = district_metrics.merge(underserved, on="District", how="left", suffixes=("", "_underserved"))
    return demand, candidates, selected_hubs, assignments, district_metrics


def export_figure(fig: go.Figure, stem: str) -> None:
    html_path = PLOTS_DIR / f"{stem}.html"
    fig.write_html(html_path)

    try:
        png_path = PLOTS_DIR / f"{stem}.png"
        fig.write_image(png_path, scale=2, width=1400, height=900)
    except Exception:
        pass


def build_assignment_frame(demand: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    demand_with_id = demand.reset_index(names="demand_index")
    assignment_map = demand_with_id.merge(assignments, on="demand_index", how="left")
    if "q_i_x" in assignment_map.columns:
        assignment_map = assignment_map.rename(columns={"q_i_x": "q_i"})
    if "w_i_x" in assignment_map.columns:
        assignment_map = assignment_map.rename(columns={"w_i_x": "w_i"})
    assignment_map["assigned_hub_label"] = (
        assignment_map["hub_district"].fillna("Unassigned")
        + " | "
        + assignment_map["hub_service"].fillna("None")
    )
    return assignment_map


def plot_selected_hubs_map(demand: pd.DataFrame, selected_hubs: pd.DataFrame) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scattermapbox(
            lat=demand["Latitude"],
            lon=demand["Longitude"],
            mode="markers",
            marker={"size": 10, "color": "rgba(80,80,80,0.45)"},
            text=demand["District"],
            name="Aggregated demand nodes",
            hovertemplate="District: %{text}<br>Lat %{lat:.3f}<br>Lon %{lon:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scattermapbox(
            lat=selected_hubs["Latitude"],
            lon=selected_hubs["Longitude"],
            mode="markers+text",
            marker={"size": 18, "color": "#c62828"},
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
    fig.update_layout(
        title="Exact Gurobi solution: selected hubs and aggregated demand nodes",
        mapbox_style="open-street-map",
        mapbox={"center": {"lat": float(demand["Latitude"].mean()), "lon": float(demand["Longitude"].mean())}, "zoom": 5.6},
        height=760,
        legend={"orientation": "h"},
    )
    export_figure(fig, "selected_hubs_map")


def plot_assignment_map(assignment_map: pd.DataFrame, selected_hubs: pd.DataFrame) -> None:
    fig = px.scatter_mapbox(
        assignment_map,
        lat="Latitude",
        lon="Longitude",
        color="hub_district",
        size="q_i",
        hover_name="Service",
        hover_data=["District", "hub_service", "distance_km", "travel_time_minutes"],
        zoom=5.6,
        height=760,
        title="Exact Gurobi solution: assignment of demand nodes to selected hubs",
    )
    fig.add_trace(
        go.Scattermapbox(
            lat=selected_hubs["Latitude"],
            lon=selected_hubs["Longitude"],
            mode="markers+text",
            marker={"size": 18, "color": "#111111"},
            text=selected_hubs["selected_rank"].astype(str),
            textposition="top right",
            name="Selected hubs",
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(mapbox_style="open-street-map")
    export_figure(fig, "assignment_map")


def plot_priority_selection_chart(district_metrics: pd.DataFrame, selected_hubs: pd.DataFrame) -> None:
    selected_districts = set(selected_hubs["District"])
    chart_df = district_metrics.copy()
    chart_df["selected_hub_present"] = chart_df["District"].isin(selected_districts).map({True: "Yes", False: "No"})
    chart_df = chart_df.sort_values("expansion_priority_score", ascending=False)

    fig = px.bar(
        chart_df,
        x="District",
        y="expansion_priority_score",
        color="selected_hub_present",
        hover_data=["expansion_priority_rank", "repair_share", "service_density_per_1000km2"],
        title="District expansion priority score vs direct hub selection",
        color_discrete_map={"Yes": "#c62828", "No": "#607d8b"},
    )
    fig.update_layout(xaxis_tickangle=-35)
    export_figure(fig, "priority_vs_selection")


def plot_utilization_chart(selected_hubs: pd.DataFrame) -> None:
    util = selected_hubs.sort_values("capacity_utilization", ascending=False).copy()
    util["hub_label"] = util["District"] + " | " + util["service_category"]
    fig = px.bar(
        util,
        x="hub_label",
        y="capacity_utilization",
        color="service_category",
        hover_data=["assigned_points", "assigned_load", "avg_distance_km", "avg_travel_time_minutes"],
        title="Capacity utilization of selected hubs",
    )
    fig.update_layout(xaxis_title="Selected hub", yaxis_title="Capacity utilization", xaxis_tickangle=-25)
    export_figure(fig, "hub_capacity_utilization")


def plot_assignment_distance_travel(selected_hubs: pd.DataFrame) -> None:
    hubs = selected_hubs.copy()
    hubs["hub_label"] = hubs["District"] + " | " + hubs["service_category"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=hubs["hub_label"],
            y=hubs["avg_distance_km"],
            name="Average distance (km)",
            marker_color="#1565c0",
        )
    )
    fig.add_trace(
        go.Bar(
            x=hubs["hub_label"],
            y=hubs["avg_travel_time_minutes"],
            name="Average travel time (minutes)",
            marker_color="#ef6c00",
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="Average service distance and traffic-adjusted travel time by selected hub",
        xaxis_title="Selected hub",
        yaxis={"title": "Average distance (km)"},
        yaxis2={"title": "Average travel time (minutes)", "overlaying": "y", "side": "right"},
        barmode="group",
        xaxis_tickangle=-25,
    )
    export_figure(fig, "distance_and_travel_time_by_hub")


def plot_underserved_chart(district_metrics: pd.DataFrame) -> None:
    chart_df = district_metrics.sort_values(
        ["underserved_binary", "expansion_priority_score"],
        ascending=[False, False],
    ).copy()
    chart_df["underserved_label"] = chart_df["underserved_binary"].map({1: "Indirectly served", 0: "Direct hub selected"})

    fig = px.bar(
        chart_df,
        x="District",
        y="expansion_priority_score",
        color="underserved_label",
        hover_data=["expansion_priority_rank", "repair_share", "mean_nearest_neighbor_km"],
        title="Underserved-district indicator and expansion priority",
        color_discrete_map={"Indirectly served": "#8e24aa", "Direct hub selected": "#2e7d32"},
    )
    fig.update_layout(xaxis_tickangle=-35)
    export_figure(fig, "underserved_priority_chart")


def plot_allocation_sankey(assignments: pd.DataFrame) -> None:
    flow = (
        assignments.groupby(["demand_district", "hub_district"])["q_i"]
        .sum()
        .reset_index()
    )
    demand_nodes = list(flow["demand_district"].unique())
    hub_nodes = [f"Hub: {d}" for d in flow["hub_district"].unique()]
    labels = demand_nodes + hub_nodes
    node_index = {label: idx for idx, label in enumerate(labels)}

    sources = [node_index[d] for d in flow["demand_district"]]
    targets = [node_index[f"Hub: {d}"] for d in flow["hub_district"]]
    values = flow["q_i"].tolist()

    fig = go.Figure(
        data=[
            go.Sankey(
                node={"label": labels, "pad": 18, "thickness": 16},
                link={"source": sources, "target": targets, "value": values},
            )
        ]
    )
    fig.update_layout(title_text="District-to-hub allocation flow in the exact Gurobi solution", font_size=11)
    export_figure(fig, "district_to_hub_sankey")


def main() -> None:
    ensure_plot_dir()
    print("Loading Gurobi result files...", flush=True)
    demand, candidates, selected_hubs, assignments, district_metrics = load_or_rebuild_inputs()
    assignment_map = build_assignment_frame(demand, assignments)

    print("Creating selected hub map...", flush=True)
    plot_selected_hubs_map(demand, selected_hubs)

    print("Creating assignment map...", flush=True)
    plot_assignment_map(assignment_map, selected_hubs)

    print("Creating priority and selection chart...", flush=True)
    plot_priority_selection_chart(district_metrics, selected_hubs)

    print("Creating utilization chart...", flush=True)
    plot_utilization_chart(selected_hubs)

    print("Creating distance and travel time chart...", flush=True)
    plot_assignment_distance_travel(selected_hubs)

    print("Creating underserved district chart...", flush=True)
    plot_underserved_chart(district_metrics)

    print("Creating district-to-hub Sankey flow diagram...", flush=True)
    plot_allocation_sankey(assignments)

    print(f"All Gurobi result plots saved in: {PLOTS_DIR.resolve()}", flush=True)


if __name__ == "__main__":
    main()
