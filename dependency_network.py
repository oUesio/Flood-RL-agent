import networkx as nx
import matplotlib.pyplot as plt

# FEATURES AND DEPENDENCIES
dependencies = {
    "Urban/rural": [],
    "Population density": ["Urban/rural"],
    "Disability rate": [],
    "English proficiency": ["Urban/rural"],
    "Low-income fraction": [],
    "Mean property value": ["Accomodation type", "Tenure of household"],
    "Distance to water": [],
    "Mean income": ["Economic activity status", "Occupation current"],
    "Mean building age": [],
    "Proportion of buildings brick/stone": ["Mean building age"],
    "24-hr precipitation total": [],
    "Emergency response time": ["24-hr precipitation total", "Population density"],
    "Flood risk level": [],
    "Flood depth": ["Flood risk level", "24-hr precipitation total"],
    "Day of week": [],
    "Season": [],
    "Holiday binary": [],
    "Elevation above sea level": [],
    "Depth-damage curve": ["Flood depth", "Physical vulnerability"],
    "Exposure": ["Population density", "Mean property value", "Day of week", "Holiday binary"],
    "Physical vulnerability": ["Proportion of buildings brick/stone", "Mean building age", "Elevation above sea level", "Distance to water", "Groundwater level", "River flow", "River level", "Soil moisture saturation", "Impervious surface area", "Drainage", "Historical flood", "Road network density", "Vegetation/land-cover index", "Seasonal vegetation cycle", "Storm overflows"],
    "Socioeconomic vulnerability": ["English proficiency", "Disability rate", "Low-income fraction", "Mean income", "General health", "Ns-Sec", "Elderly rate", "Children rate", "Employment history", "Highest qualification", "Long-term health condition", "HBAI statistics", "Access to communications", "Tenure of household", "Vehicle", "Second address", "Household", "Mental health cost", "Mental health types"],
    "Preparedness": ["Warning issued", "Emergency response time", "Season"],
    "Recovery capacity": ["Mean property value",  "Mean income", "Local government budget", "Ambulance handover delays", "Hospital bed availability", "Hospital locations", "Home insurance coverage", "Health insurance coverage", "Reconstruction time", "Household access to internet", "Access to communications"],
    "Overall vulnerability": ["Physical vulnerability", "Socioeconomic vulnerability", "Preparedness", "Recovery capacity"],
    "Impact score": ["Physical vulnerability", "Socioeconomic vulnerability", "Overall vulnerability", "Flood depth", "Population density", "Exposure", "Depth-damage curve"],
    "Groundwater level": ["24-hr precipitation total", "Distance to water", "Elevation above sea level"],
    "River flow": ["24-hr precipitation total"],
    "River level": ["24-hr precipitation total"],
    "Soil moisture saturation": ["24-hr precipitation total", "Temperature"],
    "Impervious surface area": ["Urban/rural"],
    "Drainage": ["Urban/rural"],
    "General health": [],
    "Ns-Sec": [],
    "Age": [],
    "Elderly rate": ["Age", "Population density"],
    "Children rate": ["Age", "Population density"],
    "Employment history": ["Age"],
    "Highest qualification": ["Age"],
    "HBAI statistics": [],
    "Access to communications": ["Urban/rural"],
    "Tenure of household": [],
    "Ambulance handover delays": [],
    "Hospital bed availability": ["Population density"],
    "Historical flood": [],
    "Road network density": ["Urban/rural"],
    "Economic activity status": ["Age"], ###
    "Occupation current": ["Age"],
    "Accomodation type": [],
    "Vehicle": [],
    "Second address": [],
    "Household size": [],
    "Adults employed in household": ["Household size"],
    "Disabled in household": ["Household size"],
    "Long-term health in household": ["Household size"],
    "Deprived in education": ["Highest qualification"],
    "Deprived in employment": ["Economic activity status", "Employment history"],
    "Deprived in health and disability": ["Disabled in household", "Long-term health in household", "General health"],
    "Deprived in housing": ["People per room in household", "Occupancy rating for rooms"],
    "Lifestage of HRP": ["Age"],
    "Household composition": ["Household size", "Adults and children in household",],
    "Household deprivation": ["Deprived in education", "Deprived in employment", "Deprived in health and disability", "Deprived in housing"],
    "Families in household": ["Household size"],
    "People per room in household": ["Household size"],
    "Occupancy rating for rooms": ["Household size", "People per room in household"],
    "Adults and children in household": ["Household size"],
    "Household": ["Household size", "Families in household", "Adults and children in household", "Household composition", "Lifestage of HRP", "Household deprivation", "Tenure of household", "Accomodation type", "Adults employed in household", "Disabled in household", "Long-term health in household", "Occupancy rating for rooms", "Household access to internet"],
    "Hospital locations": [],
    "Home insurance coverage": ["Household", "Mean property value", "Mean income"],
    "Health insurance coverage": ["Household", "Mean income"],
    "Reconstruction time": ["Mean property value", "Mean income", "Household"],
    "Mental health cost": [],
    "Mental health types": [],
    "Vegetation/land-cover index": ["Impervious surface area", "Urban/rural"],
    "Seasonal vegetation cycle": ["Season", "Vegetation/land-cover index"],
    "Household access to internet": [],
    "Temperature": ["Season"],
    "Storm overflows": ["24-hr precipitation total", "Urban/rural"],
}

flood_hazard = [
    "24-hr precipitation total",
    "Temperature",
    "Season",
    "Storm overflows",
    "River level",
    "River flow",
    "Soil moisture saturation",
    "Groundwater level",
    "Historical flood",
    "Flood risk level",
    "Flood depth",
    "Depth-damage curve",
    "Seasonal vegetation cycle",
    "Vegetation/land-cover index",
]

socioeconomic = [
    "Age",
    "Elderly rate",
    "Children rate",
    "Disability rate",
    "English proficiency",
    "Mean income",
    "Low-income fraction",
    "Employment history",
    "Highest qualification",
    "Economic activity status",
    "Occupation current",
    "Ns-Sec",
    "Tenure of household",
    "HBAI statistics",
    "General health",
    "Long-term health condition",
    "Mean property value",
    "Vehicle",
    "Second address",
    "Accomodation type",
    "Household",
    "Deprived in education",
    "Deprived in employment",
    "Deprived in health and disability",
    "Deprived in housing",
    "Household size",
    "Families in household",
    "Adults and children in household",
    "Adults employed in household",
    "Disabled in household",
    "Long-term health in household",
    "People per room in household",
    "Occupancy rating for rooms",
    "Household composition",
    "Household deprivation",
    "Lifestage of HRP",
    "Mental health cost",
    "Mental health types",
]

physical = [
    # Physical vulnerability inputs
    "Mean building age",
    "Proportion of buildings brick/stone",
    "Impervious surface area",
    "Drainage",
    "Road network density",
    "Distance to water",
    "Urban/rural",
    "Elevation above sea level",
    "Population density",
]

preparedness = [
    "Preparedness",
    "Warning issued",
    "Emergency response time",
    "Access to communications",
    "Household access to internet",
]

recovery = [
    "Recovery capacity",
    "Ambulance handover delays",
    "Hospital bed availability",
    "Hospital locations",
    "Reconstruction time",
    "Home insurance coverage",
    "Health insurance coverage",
    "Local government budget",
]


# DIRECTED GRAPH
G = nx.DiGraph()
for node, deps in dependencies.items():
    G.add_node(node)
    for dep in deps:
        G.add_edge(dep, node)

# COLOUR CODING

color_map = []
for node in G.nodes():
    if node == "Impact score":
        color_map.append("#FF9999")
    elif node in dependencies['Impact score']:
        color_map.append("#FFCC66")
    elif node in flood_hazard:
        color_map.append("#99CCFF") 
    elif node in socioeconomic:
        color_map.append("#CC99FF") 
    elif node in physical:
        color_map.append("#FFFF99") 
    elif node in preparedness:
        color_map.append("#66FFCC") 
    elif node in recovery:
        color_map.append("#FF66FC")
    else:
        color_map.append("#989898")


 #FF9999 FFCC66 99CCFF CC99FF FFFF99 66FFCC FFB366 66CC99

# LAYOUT
#ks = [0.1, 0.4, 0.8, 1.5, 2]
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 42, 100]
#for k in ks:
'''
for seed in seeds:
    pos = nx.spring_layout(G, seed=seed)
    plt.figure(figsize=(16, 16))
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=2600, alpha=0.80)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=12, edge_color="k", width=2.5,min_source_margin=25,min_target_margin=25)

    plt.title("Feature Dependency Network", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"dependency_network{seed}.png")
'''

'''for seed in seeds:
    rest = [n for n in G.nodes() if n not in ["Impact score"]+dependencies['Impact score']]
    shell_pos = nx.shell_layout(G, nlist=[["Impact score"], dependencies['Impact score']])

    pos = nx.spring_layout(
        G,
        pos=shell_pos,     # initial positions include the fixed shell positions
        k=0.2,
        seed=seed,
        fixed=["Impact score"]+dependencies['Impact score']  # these nodes will not move
    )
    plt.figure(figsize=(16, 16))
    nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=2600, alpha=0.80)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=12, edge_color="k", width=2.5,min_source_margin=25,min_target_margin=25)
    plt.title("Feature Dependency Network", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"dependency_network{0.2}-{seed}.png")
'''


shells = [
    ["Impact score"],
    dependencies['Impact score'],
    [n for n in G.nodes() if n != "Impact score" and n not in dependencies['Impact score']],
]
pos = nx.shell_layout(G, nlist=shells)

last_shell = shells[-1]
scale_factor = 1.5  # increase to push the shell further out

for node in last_shell:
    x, y = pos[node]
    pos[node] = (x * scale_factor, y * scale_factor)

# DRAW
plt.figure(figsize=(16, 16))
nx.draw_networkx_nodes(G, pos, node_color=color_map, node_size=2600, alpha=0.80)
nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=12, edge_color="k", width=2.5,min_source_margin=25,min_target_margin=25)

plt.title("Feature Dependency Network", fontsize=14, fontweight="bold")
plt.axis("off")
plt.tight_layout()
plt.savefig("dependency_network.png")
