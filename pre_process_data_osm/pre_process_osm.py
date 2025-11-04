import osmnx as ox
import pickle
from functools import reduce
import pandas as pd
from statistics import median
from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import linemerge, substring
import sys

def fill_lanes_directional(g):
    """
    For each edge, this function adds/updates the 'lanes_est' attribute,
    representing the number of lanes available for traffic in the direction
    of the edge (from u to v).

    Algorithm explanation:
    -------------------------------------------------
    The function aims to recover as faithfully as possible the number of lanes
    in the direction of the edge, even when OSM tags are incomplete or inconsistent.

    1. It first tries to parse a numeric 'lanes' tag directly (main OSM tag).
    2. If missing, it reconstructs the value by summing 'lanes:forward' and
       'lanes:both_ways'. If a value is found but the reverse edge has a
       different ('lanes:backward') count, this is flagged as an
       inconsistency: the lane count is averaged between the two values and
       flagged as such.
    3. If no directional info is found at all, it falls back to the median value
       for all edges in the same 'highway' class (when available).
    4. As a last resort, assigns 1 lane if no other information is available.

    The function prints out diagnostics:
      - The number of edge lane values reconstructed because of missing data.
      - The number of edge lane values adjusted due to inconsistencies between
        edge and reverse edge OSM directional tags.

    Modifies g in place.
    """

    def parse_lanes(val):
        """Try to parse a lane value (int or list/composite string)."""
        if val is None:
            return None
        if isinstance(val, (list, tuple)):
            val = ";".join(str(v) for v in val)
        val = str(val)
        for sep in (";", "|", "/", ","):
            if sep in val:
                try:
                    # If there are composite values, take the maximum as protective
                    return max(int(float(s)) for s in val.split(sep))
                except Exception:
                    continue
        try:
            return int(float(val))
        except Exception:
            return None

    def normalize_highway(val):
        """Ensure the highway tag is a string (not a list), use first if multiple."""
        if isinstance(val, (list, tuple)):
            return val[0]
        return val

    # Build the GeoDataFrame and clean highway tags for grouping
    edges_gdf = ox.graph_to_gdfs(g, nodes=False)
    edges_gdf["highway"] = edges_gdf["highway"].apply(normalize_highway)
    edges_gdf["lanes_num"] = edges_gdf["lanes"].apply(parse_lanes)
    medians_by_hwy = edges_gdf.groupby("highway")["lanes_num"].median()

    def get_reverse_edge(u, v, k):
        """
        Try to find the reverse edge (v,u,k) in the graph.
        Looks for matching key first; if not found, any parallel edge is returned.
        """
        if g.has_edge(v, u):
            if k in g[v][u]:
                return g[v][u][k]
            keys = list(g[v][u].keys())
            if keys:
                return g[v][u][keys[0]]
        return None

    missing_count = 0
    inconsistent_count = 0

    for u, v, k, data in g.edges(keys=True, data=True):
        # Test if multiplicity present
        """if k > 0:
            print("Sommet u ")
            print(u)
            print("Sommet v ")
            print(v)
            print("Data")
            print(data)
            sys.exit()"""
        # Step 1: Try direct 'lanes' value
        lanes = parse_lanes(data.get("lanes"))
        # Sanity check
        rev = get_reverse_edge(u, v, k)
        if "lanes:forward" in data or "lanes:both_ways" in data or\
           (rev and "lanes:backward" in rev):
            print("Found 'lanes:forward' or 'lanes:both_ways' or 'lanes:backward'.")
        # Step 2: Reconstruct if missing using directional tags
        if lanes is None:
            lf = parse_lanes(data.get("lanes:forward"))
            lbw = parse_lanes(data.get("lanes:both_ways")) or 0
            if lf is not None:
                lanes = lf + lbw
                missing_count += 1  # count for stats

        # Step 3: Consistency check with reverse ('lanes:backward')
        if lanes is not None:
            rev = get_reverse_edge(u, v, k)
            if rev:
                lbwd_rev = parse_lanes(rev.get("lanes:backward"))
                if lbwd_rev is not None and lbwd_rev != lanes:
                    inconsistent_count += 1
                    # Use average if values are inconsistent
                    lanes = int(round((lanes + lbwd_rev) / 2))
            data["lanes_est"] = int(lanes)
            continue

        # Step 4: Fallback to median by highway type
        hwy = data.get("highway")
        if isinstance(hwy, (list, tuple)):
            hwy = hwy[0]  # extra safety, in case we're not normalized
        med = medians_by_hwy.get(hwy, None)
        if isinstance(med, pd.Series):
            med_val = med.median()
        else:
            med_val = med
        if pd.notna(med_val):
            data["lanes_est"] = int(med_val)
            continue

        # Step 5: Default to 1 lane if all else fails
        data["lanes_est"] = 1
    print(f"Number of lanes values reconstructed due to missing data: {missing_count}\n",
        f"Number of lanes values adjusted due to inconsistencies: {inconsistent_count}"
    )
    """    
    dynamic_print(persistent_message=
        f"Number of lanes values reconstructed due to missing data: {missing_count}\n"
        f"Number of lanes values adjusted due to inconsistencies: {inconsistent_count}"
    )
    """


def convert_attr_str_to_int(g):
    # Convert the attributes "maxspeed", "length" and "lanes"
    # from str to numerical data types
    for u, v, k in g.edges(keys = True):
        # Treating "maxspeed" attribute
        if "maxspeed" in g[u][v][k]:
            if  isinstance(g[u][v][k]["maxspeed"], str):
                g[u][v][k]["maxspeed"] = int(g[u][v][k]["maxspeed"])

            elif isinstance(g[u][v][k]["maxspeed"], list):
                g[u][v][k]["maxspeed"] = [int(elem) for elem in g[u][v][k]["maxspeed"]]
        
        # Treating "length" attribute
        if "length" in g[u][v][k]:
            if isinstance(g[u][v][k]["length"], str):
                g[u][v][k]["length"] = float(g[u][v][k]["length"])

            elif isinstance(g[u][v][k]["length"], list):
                g[u][v][k]["length"] = [float(elem) for elem in g[u][v][k]["length"]]
        
        # Treating "lanes" attribute
        if "lanes" in g[u][v][k]:
            if isinstance(g[u][v][k]["lanes"], str):
                g[u][v][k]["lanes"] = int(g[u][v][k]["lanes"])

            elif isinstance(g[u][v][k]["lanes"], list):
                g[u][v][k]["lanes"] = [int(elem) for elem in g[u][v][k]["lanes"]]


def fill_attributes(g, default_vals):

    def filter_highways(g):
        for u, v, k in list(g.edges(keys = True)):
            if "highway" in g[u][v][k] and isinstance(g[u][v][k]["highway"], str) and\
            (g[u][v][k]["highway"] == "busway" or g[u][v][k]["highway"] == "living_street"):
                g.remove_edge(u, v, k)

    filter_highways(g)

    def dict_highway_vals(attribute_list):
        # Gives for each attribute name in attribute_list and 
        # highway type the values associated to this attribute for this highway type
        dict_data = {}
        for u, v, k in g.edges(keys = True):
            if "highway" in g[u][v][k]:
                highway_type = g[u][v][k]["highway"]
                for attr_name in attribute_list:
                    if attr_name in g[u][v][k]:
                        if not isinstance(g[u][v][k][attr_name], list) and\
                            isinstance(highway_type, str):
                            ls_items = dict_data.get((attr_name, highway_type), [])
                            ls_items.append(g[u][v][k][attr_name])
                            dict_data[(attr_name, highway_type)] = ls_items

                        if isinstance(g[u][v][k][attr_name], list) and\
                            isinstance(g[u][v][k]["highway"], list) and\
                            len(g[u][v][k][attr_name]) == len(g[u][v][k]["highway"]):
                            for i in range(len(g[u][v][k][attr_name])):
                                ls_items = dict_data.get((attr_name, highway_type[i]), [])
                                ls_items.append(g[u][v][k][attr_name][i])
                                dict_data[(attr_name, highway_type[i])] = ls_items
        return dict_data

    # Grouping the values of each attribute (lanes, maxspeed) by the highway type   
    dict_data = dict_highway_vals(["lanes", "maxspeed"])    
    
    for attr_name, highway_type in dict_data: 
        dict_data[(attr_name, highway_type)] = median(dict_data[(attr_name, highway_type)])
    
    for u, v, k in g.edges(keys = True):
        if "geometry" not in g[u][v][k]:
            coordinates = [(g.nodes[u]["x"], g.nodes[u]["y"]),
                           (g.nodes[v]["x"], g.nodes[v]["y"])]
            g[u][v][k]["geometry"] = LineString(coordinates)

        if "length" not in g[u][v][k]:
                g[u][v][k]["length"] = g[u][v][k]["geometry"].length

        if "highway" in g[u][v][k] and isinstance(g[u][v][k]["highway"], str):
            if "lanes" not in g[u][v][k]:
                g[u][v][k]["lanes"] = dict_data.get(("lanes", g[u][v][k]["highway"]),
                                                    default_vals["lanes"])

            if "maxspeed" not in g[u][v][k]:
                g[u][v][k]["maxspeed"] = dict_data.get(("maxspeed", g[u][v][k]["highway"]),
                                                        default_vals["maxspeed"])
        else:
            if "lanes" not in g[u][v][k]:
                g[u][v][k]["lanes"] = default_vals["lanes"]

            if "maxspeed" not in g[u][v][k]:
                g[u][v][k]["maxspeed"] = default_vals["maxspeed"]



def pre_process_attributes(g):
    # Convert strings to integers
    convert_attr_str_to_int(g)

    # Fille the lanes
    # fill_lanes_directional(g)

    # Convert the attributes "maxspeed", "length" and "lanes"
    fill_attributes(g, 
                    {"lanes":1, 
                     "maxspeed":30})    
    
  

def get_osm_map(area_name, 
                path_file = None):
    """
    Load a graph from disk if it exists, otherwise download it from OSM,
    save it, and return the monkey-patched NetworkX graph.
    """
    print("Downloading OSMnx graph…")
    g = ox.graph_from_place(area_name, network_type="drive", retain_all=True)
    pre_process_attributes(g)
    if path_file is not None:
        with open(path_file, 'wb') as f:
            pickle.dump(g, f, pickle.HIGHEST_PROTOCOL) 
    return g