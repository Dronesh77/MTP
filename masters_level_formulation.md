# Improved Masters-Level Problem Formulation

## Proposed Title
Spatial Decision Support Framework for Automotive Service Network Expansion in West Bengal Using Geocoded Roadside Infrastructure Data

## Reformulated Problem Statement
The original mid-term report focused mainly on Kolkata and presented a profit-versus-service-time formulation for service centre placement. That framing is too narrow for the richer dataset now available. The cleaned and geocoded statewide dataset of petrol pumps, food stops, and repair-related facilities across West Bengal supports a stronger problem: identifying spatial imbalance in automotive support infrastructure and recommending candidate locations for new integrated service hubs.

The core research problem is not only where service-related facilities already exist, but where the current network is structurally weak. A district may have many total services but still be underserved if repair capacity is low, services are spatially dispersed, or categories are poorly balanced. This motivates a multi-stage spatial analytics and optimization framework rather than a simple descriptive map.

## Research Gap
Most lightweight analyses stop at plotting service points on a map. That shows visibility, but it does not quantify:

- which districts are genuinely underserved,
- where repair support is disproportionately low,
- how clustered or fragmented the infrastructure is, and
- which candidate hub locations can reduce access burden across the network.

This project addresses that gap by combining data cleaning, semantic categorization, spatial metrics, hotspot detection, and service-hub recommendation in one reproducible workflow.

## Study Objectives
1. Build a district-wise geospatial database of automotive support infrastructure across West Bengal.
2. Infer functional service categories from noisy text records and validate statewide service composition.
3. Measure spatial imbalance using density, category balance, and nearest-neighbor dispersion.
4. Rank districts according to an expansion-priority index.
5. Recommend candidate integrated service hub locations that improve statewide coverage.

## Sets and Indices
- `d in D`: districts
- `i in I`: observed geocoded service points
- `j in J`: candidate hub locations drawn from observed nodes
- `c in C`: inferred service categories

## Input Parameters
- `lat_i, lon_i`: latitude and longitude of point `i`
- `g(i)`: district containing point `i`
- `cat(i)`: inferred category of point `i`
- `d_ij`: Haversine distance between demand point `i` and candidate site `j`
- `w_i`: demand importance weight for point `i`
- `R`: service radius threshold
- `P`: number of new hubs to recommend

## Stage 1: Data Enrichment and Semantic Categorization
Each raw service record is cleaned and mapped to one of four categories:

- Fuel Station
- Food/Lodging
- Vehicle Repair
- Other

This stage transforms noisy administrative data into structured inputs for spatial analysis.

## Stage 2: District Expansion-Priority Model
For each district `d`, compute:

- `N_d`: total number of services
- `A_d`: approximate service spread area from the district bounding box
- `rho_d = N_d / A_d`: proxy service density
- `r_d`: share of repair-related services
- `m_d`: mean nearest-neighbor distance among service points
- `h_d`: normalized category diversity

The district deficiency components are:

- density gap: high when `rho_d` is low
- repair gap: high when `r_d` is low
- dispersion gap: high when `m_d` is high
- diversity gap: high when `h_d` is low

The composite district expansion-priority score is:

`EPS_d = 0.35*DG_d + 0.25*RG_d + 0.20*SG_d + 0.20*VG_d`

where:

- `DG_d` = normalized density gap
- `RG_d` = normalized repair gap
- `SG_d` = normalized spatial dispersion
- `VG_d` = normalized diversity gap

Districts with higher `EPS_d` are more critical for future service expansion.

## Stage 3: Weighted Hub Recommendation Model
Let:

- `x_j = 1` if candidate site `j` is selected as a new integrated hub, else `0`
- `z_ij = 1` if demand point `i` is assigned to hub `j`, else `0`

Objective:

`min sum_i sum_j w_i * d_ij * z_ij`

Subject to:

- `sum_j z_ij = 1` for all `i`
- `z_ij <= x_j` for all `i, j`
- `sum_j x_j = P`
- `z_ij = 0` if `d_ij > R`
- `x_j in {0,1}, z_ij in {0,1}`

## Interpretation of the Optimization Model
This model selects `P` hub anchors so that weighted travel burden from observed service demand points to selected hubs is minimized within a practical coverage radius. The demand weights can be increased for repair-deficient districts and for higher-priority service types, making the model more aligned with real infrastructure need than simple point clustering.

## Why This Formulation Is Stronger
- It scales from Kolkata to the full West Bengal study area.
- It uses real geocoded service data instead of only narrative market arguments.
- It quantifies infrastructure imbalance rather than only visualizing it.
- It links descriptive analytics directly to decision-making through hub recommendation.
- It is reproducible and extensible to road networks, travel time, vehicle population, and forecast demand.

## Suggested Thesis Flow
1. Introduction and motivation
2. Literature review on automotive after-sales infrastructure and spatial service planning
3. Data collection, cleaning, geocoding, and category inference
4. Spatial exploratory analysis and district comparison
5. Expansion-priority model
6. Hub recommendation model
7. Results, managerial insights, and limitations
8. Future work with road-network and demand forecasting integration
