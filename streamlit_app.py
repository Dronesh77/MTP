from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures_mtp2_final"
GUROBI_DIR = BASE_DIR / "gurobi_outputs"
GUROBI_PLOTS_DIR = GUROBI_DIR / "plots"


st.set_page_config(
    page_title="West Bengal Service Network Dashboard",
    page_icon="📍",
    layout="wide",
)


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def show_html_plot(path: Path, height: int = 720) -> None:
    if not path.exists():
        st.warning(f"Plot not found: {path.name}")
        return
    components.html(path.read_text(encoding="utf-8"), height=height, scrolling=True)


def show_image_with_caption(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Image not found: {path.name}")


def metrics_row(summary_df: pd.DataFrame) -> None:
    cols = st.columns(5)
    cols[0].metric("Validated Points", "898")
    cols[1].metric("Aggregated Demand Nodes", "37")
    cols[2].metric("Selected Hubs", "5")
    cols[3].metric("Objective Value", "4.29M")
    cols[4].metric("MIP Gap", "0.16%")


def app_overview() -> None:
    st.title("Automotive Service Network Optimization Dashboard")
    st.markdown(
        """
        This dashboard presents the complete workflow of the project:
        data cleaning, geocoding, service categorization, traffic-aware formulation,
        exact Gurobi optimization, and visual interpretation of the results.
        """
    )

    summary_md = load_text(str(GUROBI_DIR / "optimization_summary.md"))
    selected_hubs = load_csv(str(GUROBI_DIR / "selected_hubs.csv"))
    metrics_row(selected_hubs)

    st.subheader("Project Summary")
    st.markdown(
        """
        The final model solves a traffic-aware hub location-allocation problem for
        roadside automotive support services across West Bengal. The exact solution
        was obtained with Gurobi on an aggregated demand network and selects five
        strategic hub anchors while respecting budget, assignment, and capacity logic.
        """
    )

    with st.expander("Open Gurobi Optimization Summary"):
        st.markdown(summary_md)

    st.subheader("Selected Hubs")
    st.dataframe(selected_hubs, use_container_width=True)


def app_data_cleaning() -> None:
    st.title("Data Cleaning and Processing")
    st.markdown(
        """
        The raw service records were first converted into a tabular CSV format,
        then cleaned, standardized, and enriched with district-aware address strings
        so they could be geocoded into latitude and longitude.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        show_image_with_caption(
            FIGURES_DIR / "data_before_cleaning.png",
            "Raw extracted data before cleaning",
        )
    with col2:
        show_image_with_caption(
            FIGURES_DIR / "data_after_cleaning.png",
            "Structured data after cleaning and normalization",
        )

    st.subheader("Dataset Preview")
    before_tab, after_tab, geo_tab = st.tabs(
        ["Before Geocoding", "After Geocoding", "Aggregated Demand Nodes"]
    )

    with before_tab:
        before_df = load_csv(str(BASE_DIR / "final_output_without_geocoding.csv"))
        st.write("Cleaned service records before coordinate generation")
        st.dataframe(before_df.head(100), use_container_width=True)
        st.download_button(
            "Download Before-Geocoding CSV",
            before_df.to_csv(index=False).encode("utf-8"),
            file_name="final_output_without_geocoding.csv",
            mime="text/csv",
        )

    with after_tab:
        after_df = load_csv(str(BASE_DIR / "final_with_coordinates_better.csv"))
        st.write("Geocoded dataset with latitude and longitude")
        st.dataframe(after_df.head(100), use_container_width=True)
        st.download_button(
            "Download Geocoded CSV",
            after_df.to_csv(index=False).encode("utf-8"),
            file_name="final_with_coordinates_better.csv",
            mime="text/csv",
        )

    with geo_tab:
        demand_df = load_csv(str(GUROBI_DIR / "aggregated_demand_nodes.csv"))
        st.write("Aggregated demand nodes used by the exact Gurobi model")
        st.dataframe(demand_df, use_container_width=True)

    st.subheader("How Coordinates Were Obtained")
    st.markdown(
        """
        1. Service names and district names were standardized.
        2. A refined address string was built in the form:
           `Service Name, Service Type, District, West Bengal, India`.
        3. The Google Maps Geocoding API was called for each cleaned record.
        4. Returned latitude and longitude were stored in the final CSV.
        5. District-level spatial validation was used to flag suspicious outliers.
        """
    )


def app_formulation() -> None:
    st.title("Mathematical Formulation")
    st.markdown(
        """
        The model is a traffic-aware mixed-integer location-allocation formulation.
        It selects service hubs, assigns demand nodes, and penalizes districts that
        remain without direct hub support.
        """
    )

    st.subheader("Decision Variables")
    st.latex(r"x_j = \begin{cases}1 & \text{if candidate hub } j \text{ is selected}\\0 & \text{otherwise}\end{cases}")
    st.latex(r"y_{ij} = \begin{cases}1 & \text{if demand node } i \text{ is assigned to hub } j\\0 & \text{otherwise}\end{cases}")
    st.latex(r"s_{jk} = \begin{cases}1 & \text{if service type } k \text{ is activated at hub } j\\0 & \text{otherwise}\end{cases}")

    st.subheader("Traffic-Aware Travel Time")
    st.latex(r"t_{ij} = \alpha d_{ij}\tau_{ij}")

    st.subheader("Objective")
    st.latex(
        r"""
        \min Z =
        \lambda_1 \sum_{i \in I} \sum_{j \in J} w_i d_{ij} y_{ij}
        +
        \lambda_2 \sum_{i \in I} \sum_{j \in J} w_i t_{ij} y_{ij}
        +
        \lambda_3 \sum_{d \in D} EPS_d \left(1 - \sum_{j \in J_d} x_j\right)
        """
    )

    st.subheader("Key Constraints")
    st.latex(r"\sum_{j \in J} y_{ij} = 1 \qquad \forall i \in I")
    st.latex(r"y_{ij} \le x_j \qquad \forall i \in I, \forall j \in J")
    st.latex(r"\sum_{j \in J} x_j \le P")
    st.latex(r"\sum_{j \in J} F_j x_j \le B")
    st.latex(r"\sum_{i \in I} q_i y_{ij} \le U_j x_j \qquad \forall j \in J")
    st.latex(r"\sum_{i \in I} q_i y_{ij} \ge L_j x_j \qquad \forall j \in J")

    st.subheader("Interpretation")
    st.markdown(
        """
        The model jointly minimizes geographic burden, traffic-adjusted access time,
        and the penalty of leaving high-priority districts without direct hubs.
        This makes it stronger than a simple nearest-location or distance-only model.
        """
    )


def app_results() -> None:
    st.title("Exact Gurobi Results")
    selected_hubs = load_csv(str(GUROBI_DIR / "selected_hubs.csv"))
    assignments = load_csv(str(GUROBI_DIR / "hub_assignments.csv"))
    district_metrics = load_csv(str(GUROBI_DIR / "district_metrics.csv"))
    underserved = load_csv(str(GUROBI_DIR / "underserved_districts.csv"))

    metrics_row(selected_hubs)

    st.subheader("Results Explanation")
    st.markdown(
        """
        The exact Gurobi model selected five hubs and assigned all aggregated demand
        nodes to those hubs. The selected network is regional in nature: a hub does
        not only serve its own district, but also nearby districts when the weighted
        traffic-aware cost is favorable.
        """
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Selected Hubs", "Assignments", "District Metrics", "Underserved Districts"]
    )
    with tab1:
        st.dataframe(selected_hubs, use_container_width=True)
    with tab2:
        st.dataframe(assignments, use_container_width=True)
    with tab3:
        st.dataframe(district_metrics, use_container_width=True)
    with tab4:
        st.dataframe(underserved, use_container_width=True)

    st.subheader("Interpretive Notes")
    st.markdown(
        """
        - Murshidabad and South 24 Parganas emerged as major strategic regions.
        - Repair-oriented hubs are especially important because they absorb broader support demand.
        - Some districts remain indirectly served, which is expected under a fixed hub-count and budget limit.
        - The exact model should be interpreted as a first-phase deployment plan.
        """
    )


def app_plots() -> None:
    st.title("Plots and Visual Analytics")

    st.subheader("Data Exploration Plots")
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        show_image_with_caption(
            FIGURES_DIR / "distribution_service_category.png",
            "Overall service category distribution",
        )
        show_image_with_caption(
            FIGURES_DIR / "services_heatmap.png",
            "Heatmap of services",
        )
    with img_col2:
        show_image_with_caption(
            FIGURES_DIR / "total_services_per_district.png",
            "Total services per district",
        )
        show_image_with_caption(
            FIGURES_DIR / "category_distribution_across_districts.png",
            "Category distribution across districts",
        )

    st.subheader("Exact Gurobi Plot Outputs")
    plot_choice = st.selectbox(
        "Choose a Gurobi result plot",
        [
            "Selected Hubs Map",
            "Assignment Map",
            "Hub Capacity Utilization",
            "District-to-Hub Sankey",
        ],
    )

    if plot_choice == "Selected Hubs Map":
        st.markdown("This map shows where the exact Gurobi-selected hub anchors are located.")
        show_html_plot(GUROBI_PLOTS_DIR / "selected_hubs_map.html", height=760)
    elif plot_choice == "Assignment Map":
        st.markdown("This map shows how aggregated demand nodes are allocated to the selected hubs.")
        show_html_plot(GUROBI_PLOTS_DIR / "assignment_map.html", height=760)
    elif plot_choice == "Hub Capacity Utilization":
        st.markdown("This plot shows how strongly each selected hub is used relative to its capacity.")
        show_html_plot(GUROBI_PLOTS_DIR / "hub_capacity_utilization.html", height=760)
    else:
        st.markdown("This flow diagram shows district-to-hub allocation in the exact optimized network.")
        show_html_plot(GUROBI_PLOTS_DIR / "district_to_hub_sankey.html", height=760)

    png_path = GUROBI_PLOTS_DIR / "district_to_hub_allocation.png"
    if png_path.exists():
        st.subheader("Static Sankey Export")
        st.image(str(png_path), caption="District-to-hub allocation flow", use_container_width=True)


def app_downloads() -> None:
    st.title("Downloads")
    st.markdown("Use this section to download the main datasets, results tables, and report text.")

    download_files = [
        BASE_DIR / "final_output_without_geocoding.csv",
        BASE_DIR / "final_with_coordinates_better.csv",
        GUROBI_DIR / "aggregated_demand_nodes.csv",
        GUROBI_DIR / "selected_hubs.csv",
        GUROBI_DIR / "hub_assignments.csv",
        GUROBI_DIR / "district_metrics.csv",
        GUROBI_DIR / "underserved_districts.csv",
        BASE_DIR / "overleaf_report_section.tex",
        BASE_DIR / "data_cleaning_processing_section.tex",
        BASE_DIR / "gurobi_results_with_figures.tex",
    ]

    for path in download_files:
        if path.exists():
            st.download_button(
                label=f"Download {path.name}",
                data=path.read_bytes(),
                file_name=path.name,
                mime="application/octet-stream",
            )


def main() -> None:
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Overview",
            "Data Cleaning",
            "Formulation",
            "Gurobi Results",
            "Plots",
            "Downloads",
        ],
    )

    if page == "Overview":
        app_overview()
    elif page == "Data Cleaning":
        app_data_cleaning()
    elif page == "Formulation":
        app_formulation()
    elif page == "Gurobi Results":
        app_results()
    elif page == "Plots":
        app_plots()
    else:
        app_downloads()


if __name__ == "__main__":
    main()
