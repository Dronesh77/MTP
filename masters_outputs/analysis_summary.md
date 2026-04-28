# Statewide Automotive Service Network Analysis

- Total geocoded service points analysed: 915
- Validated points used for optimization: 898
- Suspected geocoding outliers flagged: 17
- Districts covered: 19
- Dominant category: Fuel Station

## Highest-priority districts for service expansion
| District | expansion_priority_score | total_services | Vehicle Repair | service_density_per_1000km2 | mean_nearest_neighbor_km |
| --- | --- | --- | --- | --- | --- |
| Murshidabad | 0.8237122478532444 | 69 | 6 | 12.259627451374406 | 2.043154676764032 |
| South 24 Parganas | 0.8186511323814682 | 61 | 8 | 5.892615371678371 | 2.108624372253185 |
| Paschim Medinipur | 0.7621470106604482 | 18 | 4 | 16.249394677521437 | 4.019137027445198 |
| Birbhum | 0.7580767225593473 | 43 | 5 | 12.564416851901388 | 2.0382823281207236 |
| Malda | 0.7165641275278329 | 23 | 4 | 11.104068264190044 | 3.6646252335269454 |

## Strongest service hotspots
| districts | total_services | dominant_category | avg_latitude | avg_longitude |
| --- | --- | --- | --- | --- |
| Hooghly, Howrah, Purba Medinipur, South 24 Parganas | 69 | Fuel Station | 22.558133107246377 | 88.28399376521739 |
| Darjeeling, Jalpaiguri, Kalimpong | 49 | Other | 26.715686679591837 | 88.41511077959184 |
| Hooghly | 48 | Food/Lodging | 22.73909384375 | 88.31147213333333 |
| Hooghly, Nadia | 34 | Fuel Station | 23.04262748235294 | 88.49881924117646 |
| Nadia | 33 | Food/Lodging | 23.408709572727272 | 88.49914986060605 |

## Recommended new service hub anchors
| selected_hub_rank | District | Service | service_category | Latitude | Longitude | district_priority_rank | district_priority_score | estimated_covered_points | avg_distance_to_covered_points_km | hub_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Bankura | Majumder Auto Mobiles Netaji More 9474018824 | Vehicle Repair | 23.4897791 | 87.74572119999999 | 9 | 0.6653913814009995 | 42 | 34.16148309811297 | Proposed integrated service hub |
| 2 | Kalimpong | Mukul Garage Testa Bazar NH10 Kalimpong Phno 9800487893 | Vehicle Repair | 26.7049347 | 88.4483662 | 6 | 0.7013748346838267 | 112 | 20.476427206629204 | Proposed integrated service hub |
| 3 | South 24 Parganas | K C Paul Service Station Budge Budge South 24 Parganas Ph 9681244927 | Fuel Station | 22.4736564 | 88.1739326 | 2 | 0.8186511323814682 | 185 | 27.304642128853093 | Proposed integrated service hub |
| 4 | Murshidabad | Gouranga Fuel House Raghunathganj 9832112455 | Fuel Station | 24.458656 | 88.06129179999999 | 1 | 0.8237122478532444 | 69 | 30.18669683710392 | Proposed integrated service hub |
| 5 | Bankura | Maa Tara Hindu Hotel Bhedra More NH60 Mob8509659630 Tara Maa Dhaba Saluni petrol pump BankuraSaltora road Basudeb Hotel Barbra near Mataji Fuel centre BankuraChhatna Pitch | Fuel Station | 23.3741802 | 86.9035511 | 9 | 0.6653913814009995 | 86 | 32.755338685860195 | Proposed integrated service hub |

## Interpretation
- Higher expansion-priority scores indicate districts with low proxy service density, weaker repair presence, wider spatial dispersion, and lower category balance.
- Proposed hubs are selected from real observed service nodes in high-priority districts and optimized with a greedy weighted-distance heuristic.
- The current model is data-driven and reproducible, but it should later be enriched with road-network travel time, vehicle registration counts, and demand forecasts.