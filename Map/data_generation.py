import pygame
import simpy
import math
import random
import pickle
import os
import map_data
import heapq
import csv

# ==============================
# CONFIGURATION & SETTINGS
# ==============================
WIDTH, HEIGHT = 1200, 900
GRAPH_HEIGHT = 200 

WHITE, BLACK, GRAY = (255, 255, 255), (0, 0, 0), (100, 100, 100)
PURPLE_NODE, ORANGE, GREEN, RED = (150, 0, 150), (255, 165, 0), (0, 200, 0), (200, 0, 0)
TRUCK_COLOR = (0, 255, 255) 
TOWER_COLORS = {"A": (255, 50, 50), "B": (50, 255, 50), "C": (50, 50, 255)}

METERS_TO_PIXELS = 6.0
PIXELS_TO_METERS = 1.0 / METERS_TO_PIXELS
POINTS_PER_SEGMENT = 20

BATTERY_VOLTAGE_MAX = 26.5
BATTERY_VOLTAGE_MIN = 22.0
BATTERY_AH = 150
BATTERY_SCALE_FACTOR = 10
BATTERY_CAPACITY_WH = (BATTERY_VOLTAGE_MAX * BATTERY_AH) / BATTERY_SCALE_FACTOR

MOVE_POWER = 4
IDLE_POWER = 3
P0, PMAX, D0, PATH_LOSS = 5, 250, 80, 2.5
COMM_RADIUS = D0 * (PMAX / P0) ** (1 / PATH_LOSS)

NUM_TRUCKS = 6
SPEED_TOWER = 5.0
SPEED_TRUCK = 12.0
SIM_STEP = 1 
TIME_MULTIPLIER = 1 

# ==============================
# DATA LOADING & HELPERS (Implicitly same as your original)
# ==============================
try:
    with open('map_cache.pkl', 'rb') as f:
        road_graph = pickle.load(f)['road_graph']
    with open('waypoints.pkl', 'rb') as f:
        waypoints_map = pickle.load(f)
except FileNotFoundError:
    print("ERROR: map_cache.pkl or waypoints.pkl not found!")
    exit()

def a_star_pathfinding(graph, start_name, goal_name):
    open_set = [(0, start_name)]
    came_from, g_score = {}, {name: float('inf') for name in graph}
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
    if not route_nodes: return []
    for i in range(len(route_nodes) - 1):
        s_start, s_end = route_nodes[i], route_nodes[i+1]
        for chain_tuple, wps in waypoints_map.items():
            try:
                idx = chain_tuple.index(s_start)
                if idx + 1 < len(chain_tuple) and chain_tuple[idx+1] == s_end:
                    final_waypoints.extend(wps[idx*POINTS_PER_SEGMENT : (idx+1)*POINTS_PER_SEGMENT])
                    break
                idx = chain_tuple.index(s_end)
                if idx + 1 < len(chain_tuple) and chain_tuple[idx+1] == s_start:
                    segment = wps[idx*POINTS_PER_SEGMENT : (idx+1)*POINTS_PER_SEGMENT+1]
                    final_waypoints.extend(segment[::-1][:-1])
                    break
            except ValueError: pass
    if final_waypoints and route_nodes:
        final_waypoints.append(map_data.NODES[route_nodes[-1]])
    return final_waypoints

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def tx_power(d):
    if d == 0: return 0
    return P0 * (d/D0)**PATH_LOSS

def navigate_map(env, entity):
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
            dx, dy = target_wp[0] - entity.pos[0], target_wp[1] - entity.pos[1]
            dist = math.sqrt(dx*dx + dy*dy)
            step_distance = entity.speed * SIM_STEP
            if dist <= step_distance:
                entity.pos[0], entity.pos[1] = target_wp[0], target_wp[1]
                waypoints.pop(0)
            else:
                entity.pos[0] += (dx/dist) * step_distance
                entity.pos[1] += (dy/dist) * step_distance
            yield env.timeout(SIM_STEP)

class Truck:
    def __init__(self, env, i):
        self.id = f"T{i}"
        self.pos = [0.0, 0.0]
        self.speed = SPEED_TRUCK
        self.allowed_goals = map_data.LOAD_ZONES + map_data.DUMP_ZONES
        self.connected_tower = None
        env.process(navigate_map(env, self))

class Tower:
    def __init__(self, env, tid):
        self.id = tid
        self.pos = [0.0, 0.0]
        self.speed = SPEED_TOWER
        self.allowed_goals = ['main_hub', 'e_hub', 'sw_hub', 'fw_hub', 'n_hub', 's_hub']
        self.energy = BATTERY_CAPACITY_WH
        self.pct = 100.0
        self.history = []
        env.process(navigate_map(env, self))

def grid_to_screen(pos_m, scale, pan):
    return (int(pos_m[0] * METERS_TO_PIXELS * scale + pan[0]), 
            int(pos_m[1] * METERS_TO_PIXELS * scale + pan[1]))

def draw_battery_graph(screen, towers, font):
    graph_rect = pygame.Rect(50, HEIGHT - GRAPH_HEIGHT + 20, WIDTH - 100, GRAPH_HEIGHT - 40)
    pygame.draw.rect(screen, (30, 30, 30), graph_rect)
    pygame.draw.rect(screen, WHITE, graph_rect, 2)
    for tower in towers.values():
        if len(tower.history) > 1:
            points = []
            max_pts = len(tower.history)
            for i, pct in enumerate(tower.history):
                x = graph_rect.x + (i / max_pts) * graph_rect.width
                y = graph_rect.bottom - (pct / 100.0) * graph_rect.height
                points.append((int(x), int(y)))
            pygame.draw.lines(screen, TOWER_COLORS[tower.id], False, points, 2)
            last_x, last_y = points[-1]
            pct_text = font.render(f"{tower.id}: {pct:.1f}%", True, TOWER_COLORS[tower.id])
            screen.blit(pct_text, (last_x + 5, last_y - 10))

# ==============================
# MAIN SIMULATION
# ==============================
def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 14)

    all_nodes = list(map_data.NODES.values())
    min_x, max_x = min(p[0] for p in all_nodes), max(p[0] for p in all_nodes)
    min_y, max_y = min(p[1] for p in all_nodes), max(p[1] for p in all_nodes)
    scale = min((WIDTH - 40) / ((max_x - min_x) * METERS_TO_PIXELS), ((HEIGHT - GRAPH_HEIGHT) - 40) / ((max_y - min_y) * METERS_TO_PIXELS))
    pan = [20 - (min_x * METERS_TO_PIXELS * scale), 20 - (min_y * METERS_TO_PIXELS * scale)]

    env = simpy.Environment()
    tower_ids = ["A", "B", "C"]
    towers = {tid: Tower(env, tid) for tid in tower_ids}
    trucks = [Truck(env, i) for i in range(NUM_TRUCKS)]

    # --- MULTI-CSV SETUP ---
    headers = [
        "timestamp_minute", "tower_identifier", "tower_position_x_meter", "tower_position_y_meter",
        "tower_motion_power_watt", "tower_idle_power_watt", "tower_to_tower_mesh_comm_power_watt",
        "tower_to_truck_comm_power_watt", "total_power_consumption_watt", "battery_voltage_volt",
        "battery_percentage_remaining", "current_draw_ampere", "ampere_hour_used_this_step",
        "battery_energy_remaining_wh", "connected_trucks_ids", "connected_tower_ids"
    ]
    for i in range(NUM_TRUCKS):
        headers += [f"truck{i}_x", f"truck{i}_y", f"truck{i}_distance_to_A", 
                    f"truck{i}_distance_to_B", f"truck{i}_distance_to_C", f"truck{i}_connected_tower"]

    # Open 3 files and 3 writers
    csv_files = {}
    csv_writers = {}
    for tid in tower_ids:
        f = open(f"tower_{tid}_data.csv", mode='w', newline='')
        writer = csv.writer(f)
        writer.writerow(headers)
        csv_files[tid] = f
        csv_writers[tid] = writer

    running = True
    sim_running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        if sim_running:
            env.run(until=env.now + SIM_STEP)

            # 1. Update truck connections
            for t in trucks:
                dists = [(tid, distance(t.pos, tow.pos)) for tid, tow in towers.items()]
                nearest_tid, min_d = min(dists, key=lambda x: x[1])
                t.connected_tower = nearest_tid if min_d <= COMM_RADIUS else None

            # 2. Process each tower and write to its specific CSV
            for tid, tower in towers.items():
                if tower.pct <= 0: continue

                # Mesh & Truck connectivity for logging
                mesh_power = 0
                conn_towers = [oid for oid in towers if oid != tid and distance(tower.pos, towers[oid].pos) <= COMM_RADIUS]
                for oid in conn_towers: mesh_power += tx_power(distance(tower.pos, towers[oid].pos))

                truck_power = 0
                conn_trucks = [t.id for t in trucks if t.connected_tower == tid]
                for t in trucks:
                    if t.connected_tower == tid: truck_power += tx_power(distance(tower.pos, t.pos))

                total_p = MOVE_POWER + IDLE_POWER + mesh_power + truck_power
                
                v_curr = BATTERY_VOLTAGE_MIN + (tower.pct/100)*(BATTERY_VOLTAGE_MAX - BATTERY_VOLTAGE_MIN)
                i_curr = total_p / v_curr
                ah_step = i_curr * (SIM_STEP / 3600) * TIME_MULTIPLIER
                
                tower.energy -= (total_p * (SIM_STEP / 3600) * TIME_MULTIPLIER)
                tower.pct = max(0, (tower.energy / BATTERY_CAPACITY_WH) * 100)
                tower.history.append(tower.pct)

                # Prepare Data Row
                row = [
                    round(env.now / 60, 3), tid, round(tower.pos[0], 2), round(tower.pos[1], 2),
                    MOVE_POWER, IDLE_POWER, round(mesh_power, 2), round(truck_power, 2),
                    round(total_p, 2), round(v_curr, 2), round(tower.pct, 2),
                    round(i_curr, 4), round(ah_step, 6), round(tower.energy, 4),
                    ";".join(conn_trucks), ";".join(conn_towers)
                ]
                for t in trucks:
                    row += [round(t.pos[0], 2), round(t.pos[1], 2), 
                            round(distance(t.pos, towers["A"].pos), 2), 
                            round(distance(t.pos, towers["B"].pos), 2), 
                            round(distance(t.pos, towers["C"].pos), 2), 
                            t.connected_tower or "None"]
                
                # WRITE TO SPECIFIC TOWER FILE
                csv_writers[tid].writerow(row)

            if all(t.pct <= 0 for t in towers.values()):
                sim_running = False

        # --- Graphics (Simplified loop for speed) ---
        screen.fill(WHITE)
        g_to_s = lambda pos_m: grid_to_screen(pos_m, scale, pan)
        for wps in waypoints_map.values():
            if len(wps) > 1: pygame.draw.lines(screen, GRAY, False, [g_to_s(p) for p in wps], 2)
        for tid, tower in towers.items():
            if tower.pct > 0:
                p_px = g_to_s(tower.pos)
                pygame.draw.circle(screen, TOWER_COLORS[tid], p_px, int(COMM_RADIUS * METERS_TO_PIXELS * scale), 1)
                pygame.draw.circle(screen, TOWER_COLORS[tid], p_px, 8)
        for t in trucks:
            pygame.draw.circle(screen, TRUCK_COLOR, g_to_s(t.pos), 5)
        
        pygame.draw.rect(screen, BLACK, (0, HEIGHT - GRAPH_HEIGHT, WIDTH, GRAPH_HEIGHT))
        draw_battery_graph(screen, towers, font)
        pygame.display.flip()
        clock.tick(60)

    # Cleanup files
    for f in csv_files.values(): f.close()
    pygame.quit()

if __name__ == '__main__':
    run_simulation()