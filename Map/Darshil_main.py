import pygame
import simpy
import math
import random
import pickle
import os
import sys
import map_data
import heapq
import csv
import json

# ==============================
# RESOURCE PATH HELPER
# ==============================
def resource_path(filename):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    return filename

# ==============================
# LOAD SETTINGS FROM JSON
# ==============================
try:
    with open(resource_path("settings.json"), "r") as f:
        config = json.load(f)
    print("Loaded settings.json successfully.")
except Exception as e:
    print(f"settings.json not found or invalid ({e}), using defaults.")
    config = {}

# ==============================
# CONFIGURATION & SETTINGS
# ==============================
NUM_TRUCKS          = config.get("NUM_TRUCKS", 30)
NUM_TOWERS          = config.get("NUM_TOWERS", 4)
SPEED_TRUCK         = config.get("SPEED_TRUCK", 12.0)
SPEED_TOWER         = config.get("SPEED_TOWER", 5.0)

MOVE_POWER          = config.get("MOVE_POWER", 4)
IDLE_POWER          = config.get("IDLE_POWER", 3)

SIM_STEP            = config.get("SIM_STEP", 1)
TIME_MULTIPLIER     = config.get("TIME_MULTIPLIER", 1)

BATTERY_VOLTAGE_MAX = config.get("BATTERY_VOLTAGE_MAX", 26.5)
BATTERY_VOLTAGE_MIN = config.get("BATTERY_VOLTAGE_MIN", 22.0)
BATTERY_AH          = config.get("BATTERY_AH", 150)

WIDTH               = config.get("WINDOW_WIDTH", 1200)
HEIGHT              = config.get("WINDOW_HEIGHT", 900)
GRAPH_HEIGHT        = config.get("GRAPH_HEIGHT", 200)

METERS_TO_PIXELS    = config.get("METERS_TO_PIXELS", 6.0)
PIXELS_TO_METERS    = 1.0 / METERS_TO_PIXELS
POINTS_PER_SEGMENT  = config.get("POINTS_PER_SEGMENT", 20)

BATTERY_SCALE_FACTOR = config.get("BATTERY_SCALE_FACTOR", 10)
BATTERY_CAPACITY_WH  = (BATTERY_VOLTAGE_MAX * BATTERY_AH) / BATTERY_SCALE_FACTOR

P0          = config.get("P0", 5)
PMAX        = config.get("PMAX", 250)
D0          = config.get("D0", 80)
PATH_LOSS   = config.get("PATH_LOSS", 2.5)

REPOSITION_INTERVAL = config.get("REPOSITION_INTERVAL", 60)

# Communication radius: computed or overridden from JSON
_comm_override = config.get("COMM_RADIUS_OVERRIDE", None)
COMM_RADIUS = _comm_override if _comm_override else D0 * (PMAX / P0) ** (1 / PATH_LOSS)

TOWER_HUB_GOALS = config.get("TOWER_HUB_GOALS", ['main_hub', 'e_hub', 'sw_hub', 'fw_hub', 'n_hub', 's_hub'])

# Colors
WHITE       = (255, 255, 255)
BLACK       = (0, 0, 0)
GRAY        = (100, 100, 100)
ROAD_COLOR  = (180, 180, 180)

_tc = config.get("TOWER_COLOR", [220, 30, 30])
TOWER_BASE_COLOR = tuple(_tc)

_tkc = config.get("TRUCK_COLOR", [0, 60, 180])
TRUCK_COLOR = tuple(_tkc)

# ==============================
# GENERATE DISTINCT TOWER COLORS
# ==============================
def generate_tower_colors(n):
    colors = []
    for i in range(n):
        r = 220
        g = int(30 + (i / max(n - 1, 1)) * 180)
        b = int(30 + (i / max(n - 1, 1)) * 80)
        colors.append((r, g, b))
    return colors

# ==============================
# DATA LOADING
# ==============================
print("Loading map data and waypoints...")
try:
    with open(resource_path('map_cache.pkl'), 'rb') as f:
        road_graph = pickle.load(f)['road_graph']
    with open(resource_path('waypoints.pkl'), 'rb') as f:
        waypoints_map = pickle.load(f)
except FileNotFoundError:
    print("ERROR: map_cache.pkl or waypoints.pkl not found!")
    input("Press Enter to exit...")
    exit()

# ==============================
# HELPERS
# ==============================
def a_star_pathfinding(graph, start_name, goal_name):
    open_set = [(0, start_name)]
    came_from = {}
    g_score = {name: float('inf') for name in graph}
    g_score[start_name] = 0
    while open_set:
        _, current_name = heapq.heappop(open_set)
        if current_name == goal_name:
            path = []
            while current_name in came_from:
                path.append(current_name)
                current_name = came_from[current_name]
            path.append(start_name)
            return list(reversed(path))
        for neighbor, weight in graph[current_name]:
            tentative_g = g_score[current_name] + weight
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current_name
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g, neighbor))
    return []

def get_path_waypoints(route_nodes):
    final_waypoints = []
    if not route_nodes:
        return []
    for i in range(len(route_nodes) - 1):
        s_start, s_end = route_nodes[i], route_nodes[i + 1]
        for chain_tuple, wps in waypoints_map.items():
            try:
                idx = chain_tuple.index(s_start)
                if idx + 1 < len(chain_tuple) and chain_tuple[idx + 1] == s_end:
                    final_waypoints.extend(wps[idx * POINTS_PER_SEGMENT: (idx + 1) * POINTS_PER_SEGMENT])
                    break
                idx = chain_tuple.index(s_end)
                if idx + 1 < len(chain_tuple) and chain_tuple[idx + 1] == s_start:
                    segment = wps[idx * POINTS_PER_SEGMENT: (idx + 1) * POINTS_PER_SEGMENT + 1]
                    final_waypoints.extend(segment[::-1][:-1])
                    break
            except ValueError:
                pass
    if final_waypoints and route_nodes:
        final_waypoints.append(map_data.NODES[route_nodes[-1]])
    return final_waypoints

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def tx_power(d):
    if d == 0:
        return 0
    return P0 * (d / D0) ** PATH_LOSS

def navigate_map(env, entity):
    """Smooth SimPy navigation process for any entity with pos, speed, allowed_goals."""
    current_node = random.choice(entity.allowed_goals)
    entity.pos[0], entity.pos[1] = map_data.NODES[current_node][0], map_data.NODES[current_node][1]
    while True:
        target_node = random.choice(entity.allowed_goals)
        while target_node == current_node:
            target_node = random.choice(entity.allowed_goals)
        route = a_star_pathfinding(road_graph, current_node, target_node)
        waypoints = get_path_waypoints(route)
        current_node = target_node
        while waypoints:
            target_wp = waypoints[0]
            dx = target_wp[0] - entity.pos[0]
            dy = target_wp[1] - entity.pos[1]
            dist = math.sqrt(dx * dx + dy * dy)
            step_distance = entity.speed * SIM_STEP
            if dist <= step_distance:
                entity.pos[0], entity.pos[1] = target_wp[0], target_wp[1]
                waypoints.pop(0)
            else:
                entity.pos[0] += (dx / dist) * step_distance
                entity.pos[1] += (dy / dist) * step_distance
            yield env.timeout(SIM_STEP)

# ==============================
# K-MEDIAN HELPERS
# ==============================

def truck_to_nearest_nodes(trucks):
    """
    For each truck, find the nearest road node in map_data.NODES by Euclidean distance.
    Returns a list of node names (one per truck).
    """
    node_names = list(map_data.NODES.keys())
    nearest_nodes = []
    for truck in trucks:
        best_node = None
        best_dist = float('inf')
        for node_name in node_names:
            d = distance(truck.pos, map_data.NODES[node_name])
            if d < best_dist:
                best_dist = d
                best_node = node_name
        nearest_nodes.append(best_node)
    return nearest_nodes


def greedy_k_median(truck_nodes, k):
    """
    Greedy k-median on the road graph using Euclidean distances between node coords.
    Iteratively selects the candidate node that minimises the total distance from
    every truck-node to its nearest already-selected median.

    truck_nodes : list of node names (may contain duplicates)
    k           : number of medians to select
    Returns     : list of k node names
    """
    all_node_names = list(map_data.NODES.keys())

    # Deduplicate truck nodes but keep all as demand points
    demand_points = truck_nodes  # keep all (including duplicates) as demand

    if k >= len(all_node_names):
        return all_node_names[:k]

    selected = []

    # Precompute coords for fast access
    coords = {n: map_data.NODES[n] for n in all_node_names}
    demand_coords = [coords[n] for n in demand_points]

    # Track min distance from each demand point to the nearest selected median
    min_dists = [float('inf')] * len(demand_points)

    for _ in range(k):
        best_node = None
        best_total = float('inf')

        for candidate in all_node_names:
            if candidate in selected:
                continue
            c_coord = coords[candidate]
            total = 0.0
            for j, dc in enumerate(demand_coords):
                d = distance(c_coord, dc)
                total += min(min_dists[j], d)
            if total < best_total:
                best_total = total
                best_node = candidate

        if best_node is None:
            break

        selected.append(best_node)

        # Update min_dists with the newly added median
        best_coord = coords[best_node]
        for j, dc in enumerate(demand_coords):
            d = distance(best_coord, dc)
            if d < min_dists[j]:
                min_dists[j] = d

    return selected

# ==============================
# EUCLIDEAN k-MEDIAN (LOCAL SEARCH)
# ==============================

def compute_kmedian_cost(open_nodes, truck_nodes):
    """
    Total Euclidean distance:
    truck -> nearest facility node
    """

    total_cost = 0

    for tnode in truck_nodes:

        tcoord = map_data.NODES[tnode]

        min_dist = float('inf')

        for fnode in open_nodes:

            fcoord = map_data.NODES[fnode]

            d = distance(tcoord, fcoord)

            if d < min_dist:
                min_dist = d

        total_cost += min_dist

    return total_cost


def local_search_k_median_nodes(truck_nodes, k):

    all_nodes = list(map_data.NODES.keys())

    # Random initial facilities
    facilities = random.sample(all_nodes, k)

    current_cost = compute_kmedian_cost(
        facilities,
        truck_nodes
    )

    iteration = 0
    max_iterations = 30

    while iteration < max_iterations:

        improved = False

        for i in range(k):

            for candidate in all_nodes:

                if candidate in facilities:
                    continue

                new_facilities = facilities.copy()

                new_facilities[i] = candidate

                new_cost = compute_kmedian_cost(
                    new_facilities,
                    truck_nodes
                )

                if new_cost < current_cost:

                    facilities = new_facilities
                    current_cost = new_cost

                    improved = True

                    break

            if improved:
                break

        if not improved:
            break

        iteration += 1

    return facilities

def kmedian_navigation(env, tower, towers_dict, trucks):
    """
    SimPy process for a single tower.
    Every REPOSITION_INTERVAL seconds:
      - The shared k-median computation is triggered (only tower "0" computes it;
        all towers read from a shared dict to avoid redundant computation).
      - Each tower moves along the road network to its assigned k-median node.
    Movement is smooth: waypoint-by-waypoint, mirroring navigate_map logic.
    """
    # Stagger startup so towers don't all try to path-find from (0,0)
    tower_index = int(tower.id)

    # Place tower on a valid starting node immediately
    start_node = TOWER_HUB_GOALS[tower_index % len(TOWER_HUB_GOALS)]
    tower.pos[0], tower.pos[1] = map_data.NODES[start_node][0], map_data.NODES[start_node][1]
    tower.current_node = start_node

    # Small stagger so all towers don't recompute simultaneously
    yield env.timeout(tower_index * SIM_STEP)

    while True:
        # ---- Wait until next reposition interval ----
        # Tower 0 is the "coordinator": writes shared targets into towers_dict
        if tower.id == "0":
            truck_nodes = truck_to_nearest_nodes(trucks)
            # Guard: if fewer trucks than towers, pad with random nodes
            if len(truck_nodes) < NUM_TOWERS:
                all_nodes = list(map_data.NODES.keys())
                while len(truck_nodes) < NUM_TOWERS:
                    truck_nodes.append(random.choice(all_nodes))

            medians = local_search_k_median_nodes(
                        truck_nodes,
                        NUM_TOWERS
                    )

            # Assign median[i] → tower with id str(i)
            for i, tid in enumerate(sorted(towers_dict.keys())):
                towers_dict[tid].target_node = medians[i] if i < len(medians) else tower.current_node

        # Give coordinator a tick to write targets before followers read
        yield env.timeout(SIM_STEP)

        # ---- Move toward assigned target node (if it changed) ----
        target = getattr(tower, 'target_node', tower.current_node)

        if target != tower.current_node:
            route = a_star_pathfinding(road_graph, tower.current_node, target)
            waypoints = get_path_waypoints(route)

            while waypoints:
                # Check if target has been updated mid-travel (abort early)
                new_target = getattr(tower, 'target_node', target)
                if new_target != target:
                    # New assignment arrived; break and re-evaluate next cycle
                    break

                target_wp = waypoints[0]
                dx = target_wp[0] - tower.pos[0]
                dy = target_wp[1] - tower.pos[1]
                dist_to_wp = math.sqrt(dx * dx + dy * dy)
                step_distance = tower.speed * SIM_STEP

                if dist_to_wp <= step_distance:
                    tower.pos[0], tower.pos[1] = target_wp[0], target_wp[1]
                    waypoints.pop(0)
                else:
                    tower.pos[0] += (dx / dist_to_wp) * step_distance
                    tower.pos[1] += (dy / dist_to_wp) * step_distance

                yield env.timeout(SIM_STEP)

            # Snap to exact target coord and update current node
            if not waypoints or getattr(tower, 'target_node', target) == target:
                tower.pos[0], tower.pos[1] = map_data.NODES[target][0], map_data.NODES[target][1]
                tower.current_node = target

        # ---- Sleep until the next reposition ----
        yield env.timeout(max(SIM_STEP, REPOSITION_INTERVAL - SIM_STEP))


# ==============================
# ENTITIES
# ==============================
class Truck:
    def __init__(self, env, i):
        self.id = f"T{i}"
        self.pos = [0.0, 0.0]
        self.speed = SPEED_TRUCK
        self.allowed_goals = map_data.LOAD_ZONES + map_data.DUMP_ZONES
        self.connected_tower = None
        env.process(navigate_map(env, self))

class Tower:
    def __init__(self, env, tid, color, towers_dict, trucks):
        self.id = tid
        self.pos = [0.0, 0.0]
        self.speed = SPEED_TOWER
        self.allowed_goals = TOWER_HUB_GOALS      # kept for compatibility
        self.energy = BATTERY_CAPACITY_WH
        self.pct = 100.0
        self.history = []
        self.color = color
        self.dot_color = TOWER_BASE_COLOR
        self.current_node = TOWER_HUB_GOALS[0]    # updated during navigation
        self.target_node  = TOWER_HUB_GOALS[0]    # set by k-median coordinator
        env.process(kmedian_navigation(env, self, towers_dict, trucks))

# ==============================
# DRAWING HELPERS
# ==============================
def grid_to_screen(pos_m, scale, pan):
    return (
        int(pos_m[0] * METERS_TO_PIXELS * scale + pan[0]),
        int(pos_m[1] * METERS_TO_PIXELS * scale + pan[1])
    )

def draw_battery_graph(screen, towers, font, graph_rect):
    pygame.draw.rect(screen, (30, 30, 30), graph_rect)
    pygame.draw.rect(screen, WHITE, graph_rect, 2)

    for pct_mark in [0, 25, 50, 75, 100]:
        y = graph_rect.bottom - (pct_mark / 100.0) * graph_rect.height
        pygame.draw.line(screen, (60, 60, 60), (graph_rect.left, int(y)), (graph_rect.right, int(y)), 1)
        label = font.render(f"{pct_mark}%", True, (150, 150, 150))
        screen.blit(label, (graph_rect.left - 36, int(y) - 7))

    tower_list = list(towers.values())
    for tower in tower_list:
        if len(tower.history) < 2:
            continue
        max_pts = len(tower.history)
        points = []
        for i, pct in enumerate(tower.history):
            x = graph_rect.x + (i / max_pts) * graph_rect.width
            y = graph_rect.bottom - (pct / 100.0) * graph_rect.height
            points.append((int(x), int(y)))
        pygame.draw.lines(screen, tower.color, False, points, 2)
        lx, ly = points[-1]
        label = font.render(f"{tower.id}: {tower.pct:.1f}%", True, tower.color)
        screen.blit(label, (min(lx + 5, graph_rect.right - 80), max(ly - 10, graph_rect.top + 2)))

def draw_hud(screen, font, towers, trucks, sim_time):
    hud_x, hud_y = 10, 10
    lines = [
        f"Sim Time: {sim_time/60:.1f} min",
        f"Trucks: {NUM_TRUCKS}   Towers: {NUM_TOWERS}",
        f"Comm Radius: {COMM_RADIUS:.0f} m",
        f"Reposition Interval: {REPOSITION_INTERVAL}s",
    ]
    for i, t in enumerate(towers.values()):
        conn = sum(1 for tr in trucks if tr.connected_tower == t.id)
        lines.append(f"Tower {t.id}: {t.pct:.1f}%  trucks={conn}  →{t.target_node}")

    for i, line in enumerate(lines):
        surf = font.render(line, True, WHITE)
        bg = pygame.Surface((surf.get_width() + 6, surf.get_height() + 2), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 140))
        screen.blit(bg, (hud_x - 3, hud_y + i * 18 - 1))
        screen.blit(surf, (hud_x, hud_y + i * 18))

# ==============================
# MAIN SIMULATION
# ==============================
def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Mining Network Simulation — {NUM_TOWERS} Towers | {NUM_TRUCKS} Trucks")
    clock = pygame.time.Clock()
    font      = pygame.font.SysFont("Consolas", 13)
    font_big  = pygame.font.SysFont("Consolas", 15, bold=True)

    all_nodes = list(map_data.NODES.values())
    min_x = min(p[0] for p in all_nodes)
    max_x = max(p[0] for p in all_nodes)
    min_y = min(p[1] for p in all_nodes)
    max_y = max(p[1] for p in all_nodes)
    map_area_height = HEIGHT - GRAPH_HEIGHT
    scale = min(
        (WIDTH - 60) / ((max_x - min_x) * METERS_TO_PIXELS),
        (map_area_height - 60) / ((max_y - min_y) * METERS_TO_PIXELS)
    )
    pan = [
        30 - (min_x * METERS_TO_PIXELS * scale),
        30 - (min_y * METERS_TO_PIXELS * scale)
    ]

    # Build entities — trucks first so towers can reference them
    env = simpy.Environment()
    tower_colors = generate_tower_colors(NUM_TOWERS)
    tower_ids = [str(i) for i in range(NUM_TOWERS)]

    trucks = [Truck(env, i) for i in range(NUM_TRUCKS)]

    # towers_dict is passed into each Tower so kmedian_navigation can write targets
    towers = {}
    for i, tid in enumerate(tower_ids):
        towers[tid] = Tower(env, tid, tower_colors[i], towers, trucks)

    comm_radius_px = int(COMM_RADIUS * METERS_TO_PIXELS * scale)

    # ---- CSV SETUP ----
    base_headers = [
        "timestamp_minute", "tower_id",
        "tower_pos_x_m", "tower_pos_y_m",
        "move_power_w", "idle_power_w",
        "mesh_comm_power_w", "truck_comm_power_w",
        "total_power_w", "battery_voltage_v",
        "battery_pct", "current_draw_a",
        "ah_used_this_step", "energy_remaining_wh",
        "connected_truck_ids", "connected_tower_ids"
    ]
    truck_headers = []
    for i in range(NUM_TRUCKS):
        truck_headers += [
            f"truck{i}_x", f"truck{i}_y",
            *[f"truck{i}_dist_to_tower{tid}" for tid in tower_ids],
            f"truck{i}_connected_tower"
        ]
    headers = base_headers + truck_headers

    csv_files, csv_writers = {}, {}
    for tid in tower_ids:
        f = open(f"tower_{tid}_data.csv", mode='w', newline='')
        writer = csv.writer(f)
        writer.writerow(headers)
        csv_files[tid] = f
        csv_writers[tid] = writer

    graph_rect = pygame.Rect(50, HEIGHT - GRAPH_HEIGHT + 20, WIDTH - 100, GRAPH_HEIGHT - 40)

    running = True
    sim_running = True

    road_lines = []
    for wps in waypoints_map.values():
        if len(wps) > 1:
            road_lines.append([grid_to_screen(p, scale, pan) for p in wps])

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    sim_running = not sim_running

        # ---- Simulation Step ----
        if sim_running:
            env.run(until=env.now + SIM_STEP)

            for t in trucks:
                dists = [(tid, distance(t.pos, tow.pos)) for tid, tow in towers.items()]
                nearest_tid, min_d = min(dists, key=lambda x: x[1])
                t.connected_tower = nearest_tid if min_d <= COMM_RADIUS else None

            for tid, tower in towers.items():
                if tower.pct <= 0:
                    continue

                mesh_p = sum(
                    tx_power(distance(tower.pos, towers[oid].pos))
                    for oid in towers
                    if oid != tid and distance(tower.pos, towers[oid].pos) <= COMM_RADIUS
                )
                truck_p = sum(
                    tx_power(distance(tower.pos, t.pos))
                    for t in trucks if t.connected_tower == tid
                )
                total_p = MOVE_POWER + IDLE_POWER + mesh_p + truck_p

                v_curr = BATTERY_VOLTAGE_MIN + (tower.pct / 100) * (BATTERY_VOLTAGE_MAX - BATTERY_VOLTAGE_MIN)
                i_curr = total_p / v_curr if v_curr > 0 else 0
                ah_step = i_curr * (SIM_STEP / 3600) * TIME_MULTIPLIER

                tower.energy -= (total_p * (SIM_STEP / 3600) * TIME_MULTIPLIER)
                tower.pct = max(0.0, (tower.energy / BATTERY_CAPACITY_WH) * 100)
                tower.history.append(tower.pct)

                conn_towers = [
                    oid for oid in towers
                    if oid != tid and distance(tower.pos, towers[oid].pos) <= COMM_RADIUS
                ]
                conn_trucks = [t.id for t in trucks if t.connected_tower == tid]

                row = [
                    round(env.now / 60, 3), tid,
                    round(tower.pos[0], 2), round(tower.pos[1], 2),
                    MOVE_POWER, IDLE_POWER,
                    round(mesh_p, 3), round(truck_p, 3),
                    round(total_p, 3), round(v_curr, 3),
                    round(tower.pct, 2), round(i_curr, 4),
                    round(ah_step, 6), round(tower.energy, 4),
                    ";".join(conn_trucks), ";".join(conn_towers)
                ]
                for t in trucks:
                    row.append(round(t.pos[0], 2))
                    row.append(round(t.pos[1], 2))
                    for oid in tower_ids:
                        row.append(round(distance(t.pos, towers[oid].pos), 2))
                    row.append(t.connected_tower if t.connected_tower else "None")

                csv_writers[tid].writerow(row)

            if all(t.pct <= 0 for t in towers.values()):
                sim_running = False
                print("All towers depleted. Simulation ended.")

        # ---- DRAWING ----
        screen.fill((20, 20, 30))

        for pts in road_lines:
            if len(pts) > 1:
                pygame.draw.lines(screen, ROAD_COLOR, False, pts, 1)

        g_to_s = lambda pos_m: grid_to_screen(pos_m, scale, pan)

        for tid, tower in towers.items():
            if tower.pct <= 0:
                continue
            t_px = g_to_s(tower.pos)
            for t in trucks:
                if t.connected_tower == tid:
                    pygame.draw.line(screen, (60, 60, 80), t_px, g_to_s(t.pos), 1)

        for tid, tower in towers.items():
            if tower.pct <= 0:
                continue
            t_px = g_to_s(tower.pos)
            pygame.draw.circle(screen, TOWER_BASE_COLOR, t_px, comm_radius_px, 1)
            pygame.draw.circle(screen, TOWER_BASE_COLOR, t_px, 9)
            pygame.draw.circle(screen, WHITE, t_px, 9, 1)
            lbl = font.render(tower.id, True, WHITE)
            screen.blit(lbl, (t_px[0] + 11, t_px[1] - 8))
            pct_lbl = font.render(f"{tower.pct:.0f}%", True, tower.color)
            screen.blit(pct_lbl, (t_px[0] + 11, t_px[1] + 4))

            # Draw a small marker at the current k-median target
            if hasattr(tower, 'target_node') and tower.target_node in map_data.NODES:
                tgt_px = g_to_s(map_data.NODES[tower.target_node])
                pygame.draw.circle(screen, tower.color, tgt_px, 5, 2)
                pygame.draw.line(screen, tower.color, t_px, tgt_px, 1)

        for t in trucks:
            t_px = g_to_s(t.pos)
            pygame.draw.circle(screen, TRUCK_COLOR, t_px, 5)
            pygame.draw.circle(screen, (150, 180, 255), t_px, 5, 1)

        pygame.draw.rect(screen, (10, 10, 15), (0, HEIGHT - GRAPH_HEIGHT, WIDTH, GRAPH_HEIGHT))
        draw_battery_graph(screen, towers, font, graph_rect)

        gtitle = font_big.render("Tower Battery History", True, WHITE)
        screen.blit(gtitle, (graph_rect.centerx - gtitle.get_width() // 2, HEIGHT - GRAPH_HEIGHT + 3))

        draw_hud(screen, font, towers, trucks, env.now)

        if not sim_running:
            pause_surf = font_big.render("PAUSED — SPACE to resume | ESC to quit", True, (255, 220, 50))
            screen.blit(pause_surf, (WIDTH // 2 - pause_surf.get_width() // 2, HEIGHT // 2 - 16))

        pygame.display.flip()
        clock.tick(60)

    for f in csv_files.values():
        f.close()
    pygame.quit()
    print("Simulation finished. CSV files saved.")

if __name__ == '__main__':
    run_simulation()