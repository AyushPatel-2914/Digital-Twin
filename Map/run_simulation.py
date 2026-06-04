import pygame
import simpy
import math
import random
import pickle
import os
import map_data
import heapq

# ==============================
# CONFIGURATION & SETTINGS
# ==============================
WIDTH, HEIGHT = 1200, 900
GRAPH_HEIGHT = 200 # Space reserved at the bottom for the battery graph

WHITE, BLACK, GRAY = (255, 255, 255), (0, 0, 0), (100, 100, 100)
PURPLE_NODE, ORANGE, GREEN, RED = (150, 0, 150), (255, 165, 0), (0, 200, 0), (200, 0, 0)
TRUCK_COLOR = (0, 255, 255) # Cyan
TOWER_COLORS = {"A": (255, 50, 50), "B": (50, 255, 50), "C": (50, 50, 255)}

METERS_TO_PIXELS = 6.0
PIXELS_TO_METERS = 1.0 / METERS_TO_PIXELS
POINTS_PER_SEGMENT = 20

# --- SIMULATION & RF PARAMS ---
BATTERY_VOLTAGE_MAX = 26.5
BATTERY_VOLTAGE_MIN = 22.0
BATTERY_AH = 150

# ---> CRITICAL FIX FOR SPEED:
# We divide the real capacity by 50 so it drains in minutes instead of hours!
BATTERY_SCALE_FACTOR = 10
BATTERY_CAPACITY_WH = (BATTERY_VOLTAGE_MAX * BATTERY_AH) / BATTERY_SCALE_FACTOR

MOVE_POWER = 4
IDLE_POWER = 3

P0 = 5
PMAX = 250
D0 = 80
PATH_LOSS = 2.5
COMM_RADIUS = D0 * (PMAX / P0) ** (1 / PATH_LOSS)

NUM_TRUCKS = 6
SPEED_TOWER = 5.0
SPEED_TRUCK = 12.0

SIM_STEP = 1 # 1 simulated second per frame
TIME_MULTIPLIER = 1 # Multiplies energy drain to make the graph move faster

# ==============================
# DATA LOADING
# ==============================
print("Loading map data and waypoints...")
try:
    with open('map_cache.pkl', 'rb') as f:
        road_graph = pickle.load(f)['road_graph']
    with open('waypoints.pkl', 'rb') as f:
        waypoints_map = pickle.load(f)
except FileNotFoundError:
    print("ERROR: map_cache.pkl or waypoints.pkl not found!")
    exit()

# ==============================
# PATHFINDING HELPERS
# ==============================
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

# ==============================
# ENTITY MOVEMENT GENERATOR
# ==============================
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

# ==============================
# CLASSES
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
    def __init__(self, env, tid):
        self.id = tid
        self.pos = [0.0, 0.0]
        self.speed = SPEED_TOWER
        self.allowed_goals = ['main_hub', 'e_hub', 'sw_hub', 'fw_hub', 'n_hub', 's_hub']
        self.energy = BATTERY_CAPACITY_WH
        self.pct = 100.0
        self.history = []
        env.process(navigate_map(env, self))

# ==============================
# DRAWING HELPERS
# ==============================
def grid_to_screen(pos_m, scale, pan):
    return (int(pos_m[0] * METERS_TO_PIXELS * scale + pan[0]), 
            int(pos_m[1] * METERS_TO_PIXELS * scale + pan[1]))

def draw_battery_graph(screen, towers, font):
    graph_rect = pygame.Rect(50, HEIGHT - GRAPH_HEIGHT + 20, WIDTH - 100, GRAPH_HEIGHT - 40)
    pygame.draw.rect(screen, (30, 30, 30), graph_rect) # Graph background
    pygame.draw.rect(screen, WHITE, graph_rect, 2)     # Graph border

    # Labels
    title = font.render("Live Battery Percentage (%)", True, WHITE)
    screen.blit(title, (graph_rect.x, graph_rect.y - 20))
    screen.blit(font.render("100%", True, GRAY), (graph_rect.x - 40, graph_rect.y))
    screen.blit(font.render("0%", True, GRAY), (graph_rect.x - 30, graph_rect.bottom - 15))

    # Draw lines
    for tower in towers.values():
        if len(tower.history) > 1:
            points = []
            max_pts = len(tower.history)
            for i, pct in enumerate(tower.history):
                x = graph_rect.x + (i / max_pts) * graph_rect.width
                y = graph_rect.bottom - (pct / 100.0) * graph_rect.height
                points.append((int(x), int(y)))
            pygame.draw.lines(screen, TOWER_COLORS[tower.id], False, points, 2)
            
            # Label the current percentage
            last_x, last_y = points[-1]
            pct_text = font.render(f"{tower.id}: {pct:.1f}%", True, TOWER_COLORS[tower.id])
            screen.blit(pct_text, (last_x + 5, last_y - 10))

# ==============================
# MAIN SIMULATION LOOP
# ==============================
def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Dynamic Mining Network Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 14)

    # Pre-render static map to a surface to save massive CPU cycles
    all_nodes = list(map_data.NODES.values())
    min_x, max_x = min(p[0] for p in all_nodes), max(p[0] for p in all_nodes)
    min_y, max_y = min(p[1] for p in all_nodes), max(p[1] for p in all_nodes)
    map_w, map_h = max_x - min_x, max_y - min_y
    scale = min((WIDTH - 40) / (map_w * METERS_TO_PIXELS), ((HEIGHT - GRAPH_HEIGHT) - 40) / (map_h * METERS_TO_PIXELS))
    pan = [20 - (min_x * METERS_TO_PIXELS * scale), 20 - (min_y * METERS_TO_PIXELS * scale)]

    # Setup SimPy
    env = simpy.Environment()
    towers = {tid: Tower(env, tid) for tid in ["A", "B", "C"]}
    trucks = [Truck(env, i) for i in range(NUM_TRUCKS)]

    running = True
    sim_running = True

    while running:
        # 1. Pygame Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if sim_running:
            # 2. Advance SimPy
            env.run(until=env.now + SIM_STEP)

            # 3. Calculate Communications & Battery Drain
            for t in trucks:
                dists = [(tid, distance(t.pos, tow.pos)) for tid, tow in towers.items()]
                nearest_tid, min_d = min(dists, key=lambda x: x[1])
                t.connected_tower = nearest_tid if min_d <= COMM_RADIUS else None

            for tid, tower in towers.items():
                if tower.pct <= 0:
                    continue # Dead battery

                # Mesh power to other towers
                mesh_power = 0
                for oid, otower in towers.items():
                    if oid != tid:
                        d = distance(tower.pos, otower.pos)
                        if d <= COMM_RADIUS: mesh_power += tx_power(d)

                # Comm power to trucks
                truck_power = 0
                for t in trucks:
                    if t.connected_tower == tid:
                        d = distance(tower.pos, t.pos)
                        truck_power += tx_power(d)

                total_power = MOVE_POWER + IDLE_POWER + mesh_power + truck_power
                
                # Drain energy (scaled up by TIME_MULTIPLIER for visual speed)
                energy_used = total_power * (SIM_STEP / 3600) * TIME_MULTIPLIER 
                tower.energy -= energy_used
                tower.pct = max(0, (tower.energy / BATTERY_CAPACITY_WH) * 100)
                tower.history.append(tower.pct)

                if tower.pct <= 0:
                    print(f"Tower {tid} battery depleted!")

            # Stop condition
            if all(t.pct <= 0 for t in towers.values()):
                print("All tower batteries depleted. Simulation paused.")
                sim_running = False

        # 4. Drawing
        screen.fill(WHITE)
        g_to_s = lambda pos_m: grid_to_screen(pos_m, scale, pan)

        # Draw Roads
        for wps in waypoints_map.values():
            if len(wps) > 1:
                pygame.draw.lines(screen, GRAY, False, [g_to_s(p) for p in wps], 2)
        
        # Draw Map Nodes
        for name, pos_m in map_data.NODES.items():
            color = GREEN if name in map_data.LOAD_ZONES else RED if name in map_data.DUMP_ZONES else PURPLE_NODE
            pygame.draw.circle(screen, color, g_to_s(pos_m), 3)

        # Draw Comm Links (Faint Lines)
        for t in trucks:
            if t.connected_tower and towers[t.connected_tower].pct > 0:
                pygame.draw.line(screen, (200, 200, 200), g_to_s(t.pos), g_to_s(towers[t.connected_tower].pos), 1)
        
        for t1 in towers.values():
            for t2 in towers.values():
                if t1.id != t2.id and distance(t1.pos, t2.pos) <= COMM_RADIUS and t1.pct > 0 and t2.pct > 0:
                    pygame.draw.line(screen, (255, 200, 200), g_to_s(t1.pos), g_to_s(t2.pos), 1)

        # Draw Towers & Radiuses
        for tower in towers.values():
            if tower.pct > 0:
                pos_px = g_to_s(tower.pos)
                pygame.draw.circle(screen, TOWER_COLORS[tower.id], pos_px, int(COMM_RADIUS * METERS_TO_PIXELS * scale), 1)
                pygame.draw.circle(screen, TOWER_COLORS[tower.id], pos_px, 8)
                screen.blit(font.render(tower.id, True, BLACK), (pos_px[0]+10, pos_px[1]+10))

        # Draw Trucks
        for t in trucks:
            pygame.draw.circle(screen, TRUCK_COLOR, g_to_s(t.pos), 5)
            pygame.draw.circle(screen, BLACK, g_to_s(t.pos), 5, 1)

        # Draw UI
        pygame.draw.rect(screen, BLACK, (0, HEIGHT - GRAPH_HEIGHT, WIDTH, GRAPH_HEIGHT))
        draw_battery_graph(screen, towers, font)

        sim_time_text = font.render(f"Sim Time: {env.now:.1f}s | Real Time Multiplier: {TIME_MULTIPLIER}x", True, BLACK)
        screen.blit(sim_time_text, (10, 10))

        pygame.display.flip()
        clock.tick(60) # Run at 60 Frames Per Second

    pygame.quit()

if __name__ == '__main__':
    run_simulation()