
import networkx as nx
import osmnx as ox

import json

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from geopy.point import Point

import pandas as pd

import unidecode

from shapely.geometry import LineString, Point, MultiLineString
from shapely.ops import linemerge, substring

import pickle
import sys


AGGLOMERATION_AREA_NAMES = ["Bailly, Yvelines, France", 
                            "Bièvres, Yvelines, France", 
                            "Bois d'Arcy, Yvelines, France", 
                            "Bougival, Yvelines, France", 
                            "Buc, Yvelines, France", 
                            "Châteaufort, Yvelines, France", 
                            "Fontenay-Le-Fleury, Yvelines, France", 
                            "Jouy-En-Josas, Yvelines, France", 
                            "La Celle Saint-Cloud, Yvelines, France", 
                            "Le Chesnay-Rocquencourt, Yvelines, France", 
                            "Les Loges-En-Josas, Yvelines, France", 
                            "Noisy-Le-Roi, Yvelines, France", 
                            "Rennemoulin, Yvelines, France", 
                            "Saint-Cyr-l'École, Yvelines, France", 
                            "Toussus-Le-Noble, Yvelines, France", 
                            "Vélizy-Villacoublay, Yvelines, France", 
                            "Versailles, Yvelines, France", 
                            "Viroflay, Yvelines, France"]


def get_osm_map(area_name, 
                path_file = None):
    """
    Load a graph from disk if it exists, otherwise download it from OSM,
    save it, and return the monkey-patched NetworkX graph.
    """
    print("Downloading OSMnx graph…")
    g = ox.graph_from_place(area_name, network_type="drive", retain_all=True)
    fill_lanes_directional(g)
    if path_file is not None:
        with open(path_file, 'wb') as f:
            pickle.dump(g, f, pickle.HIGHEST_PROTOCOL) 
    return g


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


def load_interest_point(graph, points_filename):
    """
    Updates an OSMnx street-network graph with interest point data and cache the
    added points. For each point, a new vertex is created and the
    arcs (forward and backard) are splitted at it. Geometries are preserved
    for display purpose.

    Point geocoding is performed using Nominatim service.

    Parameters
    ----------
    graph: networkx.MultiDiGraph
        An OSMnx graph in WGS 84 (long/lat) coordinates that will be modified
        in-place.
    points_filename: a JSON file containing interest points addresses in the form:
    ```json
    [
      {
        "nom": "name",
        "adresse": "address",
        "ville": "city name"
      },    
    ```

    Returns
    -------
    list[tuple[int, str]]
        A list of ``(node_id, point_name)`` for every point added
    """

    def normalize_address(chaine):
        return unidecode.unidecode(chaine).strip().upper()

    with open(points_filename, "r", encoding="utf-8") as f:
        relais = json.load(f)

    nodes = list()
    geolocator = Nominatim(user_agent="UrbanLogistic (nassim.haddam@uvsq.fr)")
    total = len(relais)

    for idx, point in enumerate(relais, 1):
        #dynamic_print(f"Geocoding {idx}/{total}", None)
        print(f"Geocoding {idx}/{total}")

        full_address = point['adresse'] + ", " + point['ville']
        try:
            
            geocode = RateLimiter(
                geolocator.geocode,
                min_delay_seconds=1,
                max_retries=2,
                error_wait_seconds=2
            )
            location = geocode(full_address, timeout=10)
            
            if location is None:
                #dynamic_print(None, f"[GEOCODING FAILED] {full_address}")
                print(None, f"[GEOCODING FAILED] {full_address}")
            else:
                lon, lat = location.longitude, location.latitude

                u, v, k = ox.distance.nearest_edges(graph, lon, lat)
                e_data = graph[u][v][k]

                edge_geom = e_data.get(
                    "geometry",
                    LineString([(graph.nodes[u]["x"], graph.nodes[u]["y"]),
                                (graph.nodes[v]["x"], graph.nodes[v]["y"])])
                )

                if isinstance(edge_geom, MultiLineString):
                    edge_geom = linemerge(edge_geom)

                p_snap = edge_geom.interpolate(edge_geom.project(Point(lon, lat)))
                d_proj = edge_geom.project(p_snap)
                d_total = edge_geom.length

                geom_uw = substring(edge_geom, 0, d_proj, normalized=False)
                geom_wv = substring(edge_geom, d_proj, d_total, normalized=False)

                w_id = max(graph.nodes) + 1
                point_name = normalize_address(point["nom"])
                """graph.add_node(
                    w_id,
                    x=p_snap.x,
                    y=p_snap.y,
                    relay_name=point_name,
                    relay_type=relay_type
                )"""
                graph.add_node(
                    w_id,
                    x=p_snap.x,
                    y=p_snap.y,
                    relay_name=point_name
                )

                base_attrs = {k_: v_ for k_, v_ in e_data.items()
                              if k_ not in ("geometry", "length")}

                graph.remove_edge(u, v, k)
                graph.add_edge(u, w_id, **base_attrs,
                               geometry=geom_uw, length=geom_uw.length)
                graph.add_edge(w_id, v, **base_attrs,
                               geometry=geom_wv, length=geom_wv.length)

                if graph.has_edge(v, u):
                    for k_rev, d_rev in list(graph[v][u].items()):
                        geom_rev = d_rev.get(
                            "geometry",
                            LineString([(graph.nodes[v]["x"], graph.nodes[v]["y"]),
                                        (graph.nodes[u]["x"], graph.nodes[u]["y"])])
                        )
                        if isinstance(geom_rev, MultiLineString):
                            geom_rev = linemerge(geom_rev)

                        d_tot_rev = geom_rev.length
                        d_proj_rev = d_tot_rev - d_proj
                        geom_vw = substring(geom_rev, 0, d_proj_rev, normalized=False)
                        geom_wu = substring(geom_rev, d_proj_rev, d_tot_rev, normalized=False)

                        base_rev = {k_: v_ for k_, v_ in d_rev.items()
                                    if k_ not in ("geometry", "length")}

                        graph.remove_edge(v, u, k_rev)
                        graph.add_edge(v, w_id, **base_rev,
                                       geometry=geom_vw, length=geom_vw.length)
                        graph.add_edge(w_id, u, **base_rev,
                                       geometry=geom_wu, length=geom_wu.length)

                nodes.append((w_id, point_name))

        except Exception as e:
            #dynamic_print(None, f"[UNKONWN ERROR] {full_address} | {e}")
            print(f"[UNKONWN ERROR] {full_address} | {e}")

    """with open(nodes_filename, "w", encoding="utf-8") as f:
        json.dump(nodes, f)  # les node_id restent bien au format int dans le json
    """
    return nodes, True
