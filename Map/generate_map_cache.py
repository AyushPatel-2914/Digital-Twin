import numpy as np
import pickle
import map_data
import os
import heapq
import itertools

def build_weighted_graph_3d(nodes: dict, edges: list) -> dict:
    """
    Builds a weighted graph where edge weights are 3D Euclidean distances.
    """
    graph = {name: [] for name in nodes}
    for edge in edges:
        if len(edge) < 2:
            continue
            
        n1_name, n2_name = edge[0], edge[1]
        
        if n1_name not in nodes or n2_name not in nodes:
            print(f"Warning: Edge nodes not found in NODES dict: {n1_name}, {n2_name}")
            continue

        p1 = nodes[n1_name] # Now np.array([x, y, z])
        p2 = nodes[n2_name] # Now np.array([x, y, z])
        
        # linalg.norm automatically handles 3D vectors
        distance = np.linalg.norm(p1 - p2)
        
        # Add bidirectional connection
        graph[n1_name].append((n2_name, distance))
        graph[n2_name].append((n1_name, distance))
    return graph

def a_star_pathfinding_3d(graph: dict, nodes_coords: dict, start_name: str, goal_name: str) -> list:
    """
    Finds the shortest path between two nodes using A* in 3D.
    Heuristic (h) uses 3D straight-line distance.
    """
    if start_name not in graph or goal_name not in graph:
        return []
        
    # Heuristic function: 3D Euclidean distance to goal
    def h(name):
        return np.linalg.norm(nodes_coords[name] - nodes_coords[goal_name])

    # open_set stores (f_score, node_name)
    open_set = [(h(start_name), start_name)]
    came_from = {}
    
    g_score = {name: float('inf') for name in graph}
    g_score[start_name] = 0
    
    f_score = {name: float('inf') for name in graph}
    f_score[start_name] = h(start_name)
    
    while open_set:
        # Pop node with the lowest f_score
        _, current_name = heapq.heappop(open_set)

        if current_name == goal_name:
            path_names = []
            temp = current_name
            while temp in came_from:
                path_names.append(temp)
                temp = came_from[temp]
            path_names.append(start_name)
            return list(reversed(path_names))

        for neighbor_name, weight in graph[current_name]:
            tentative_g_score = g_score[current_name] + weight
            
            if tentative_g_score < g_score[neighbor_name]:
                came_from[neighbor_name] = current_name
                g_score[neighbor_name] = tentative_g_score
                f_score[neighbor_name] = tentative_g_score + h(neighbor_name)
                
                # Push to open set if not already there with a better score
                heapq.heappush(open_set, (f_score[neighbor_name], neighbor_name))
                
    return []

def main():
    print("Building 3D road graph from map_data...")
    if not hasattr(map_data, 'NODES') or not hasattr(map_data, 'EDGES'):
        print("Error: map_data.py is missing NODES or EDGES.")
        return

    # Use the 3D graph builder
    road_graph = build_weighted_graph_3d(map_data.NODES, map_data.EDGES)
    
    print("Pre-calculating optimal 3D routes between zones...")
    route_cache = {}
    
    fuel_zones = getattr(map_data, 'FUEL_ZONES', [])
    all_targets = list(set(map_data.LOAD_ZONES + map_data.DUMP_ZONES + fuel_zones))
    
    count = 0
    total_pairs = len(all_targets) * (len(all_targets) - 1)
    
    for start, end in itertools.permutations(all_targets, 2):
        # Pass NODES to pathfinding to allow for the 3D heuristic calculation
        path = a_star_pathfinding_3d(road_graph, map_data.NODES, start, end)
        if path:
            route_cache[(start, end)] = path
        
        count += 1
        if count % 100 == 0:
            print(f"  Cached {count}/{total_pairs} routes...", end='\r')
            
    print(f"\nCached {len(route_cache)} valid 3D routes.")

    output_file = 'map_cache.pkl'
    print(f"Saving 3D graph and route cache to {output_file}...")
    try:
        with open(output_file, 'wb') as f:
            pickle.dump({
                'road_graph': road_graph,
                'route_cache': route_cache
            }, f)
        print("Done! map_cache.pkl regenerated successfully for 3D.")
    except Exception as e:
        print(f"Error saving pickle file: {e}")

if __name__ == "__main__":
    main()