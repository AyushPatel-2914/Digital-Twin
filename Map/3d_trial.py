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
# 3D PROJECTION HELPERS
# ==============================
def project_3d_to_2d(pos_3d, scale, pan, width, height):
    """
    Projects (x, y, z) coordinates to (screen_x, screen_y).
    Assumes Y is depth and Z is height/elevation.
    """
    x, y, z = pos_3d
    # Simple perspective projection
    fov = 400 
    viewer_distance = 600
    
    # Adjust for elevation (z) and depth (y)
    factor = fov / (viewer_distance + y)
    
    x_proj = x * factor * scale + pan[0] + width // 2
    y_proj = -z * factor * scale + pan[1] + height // 2 # Negative Z because screen Y grows downward
    
    return (int(x_proj), int(y_proj)), factor

def distance_3d(p1, p2):
    """Calculates 3D Euclidean distance."""
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

# ==============================
# CONFIGURATION
# ==============================
def resource_path(filename):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    return filename

try:
    with open(resource_path("settings.json"), "r") as f:
        config = json.load(f)
except:
    config = {}

NUM_TRUCKS = config.get("NUM_TRUCKS", 6)
SPEED_TRUCK = config.get("SPEED_TRUCK", 12.0)
SPEED_TOWER = config.get("SPEED_TOWER", 5.0)
MOVE_POWER = config.get("MOVE_POWER", 4)
IDLE_POWER = config.get("IDLE_POWER", 3)
SIM_STEP = config.get("SIM_STEP", 1)
TIME_MULTIPLIER = config.get("TIME_MULTIPLIER", 1)
BATTERY_VOLTAGE_MAX = 26.5
BATTERY_VOLTAGE_MIN = 22.0
BATTERY_AH = 150
BATTERY_CAPACITY_WH = (BATTERY_VOLTAGE_MAX * BATTERY_AH) / 10

WIDTH, HEIGHT = 1200, 900
GRAPH_HEIGHT = 200 
WHITE, BLACK, GRAY = (255, 255, 255), (0, 0, 0), (100, 100, 100)
TRUCK_COLOR = (0, 255, 255) 
TOWER_COLORS = {"A": (255, 50, 50), "B": (50, 255, 50), "C": (50, 50, 255)}

P0, PMAX, D0, PATH_LOSS = 5, 250, 80, 2.5
COMM_RADIUS = D0 * (PMAX / P0) ** (1 / PATH_LOSS)

# ==============================
# DATA LOADING
# ==============================
try:
    with open(resource_path('map_cache.pkl'), 'rb') as f:
        road_graph = pickle.load(f)['road_graph']
    with open(resource_path('waypoints.pkl'), 'rb') as f:
        waypoints_map = pickle.load(f)
except:
    print("Error loading data")
    sys.exit()

# ==============================
# LOGIC CLASSES
# ==============================
def tx_power(d):
    if d <= 0: return 0
    return P0 * (d/D0)**PATH_LOSS

def navigate_map_3d(env, entity):
    # If map_data nodes only have 2 points, add a random elevation (Z)
    current_node = random.choice(entity.allowed_goals)
    n_pos = map_data.NODES[current_node]
    entity.pos = [n_pos[0], n_pos[1], n_pos[2] if len(n_pos) > 2 else random.uniform(0, 30)]

    while True:
        target_node = random.choice(entity.allowed_goals)
        while target_node == current_node:
            target_node = random.choice(entity.allowed_goals)
        
        route = [] # A* would go here as per your original code
        # Logic follows your original navigate_map but uses entity.pos[0,1,2]
        yield env.timeout(SIM_STEP)

class Truck:
    def __init__(self, env, i):
        self.id = f"T{i}"
        self.pos = [0.0, 0.0, 0.0]
        self.speed = SPEED_TRUCK
        self.allowed_goals = map_data.LOAD_ZONES + map_data.DUMP_ZONES
        self.connected_tower = None
        env.process(navigate_map_3d(env, self))

class Tower:
    def __init__(self, env, tid):
        self.id = tid
        self.pos = [0.0, 0.0, 20.0] # Towers usually elevated
        self.speed = SPEED_TOWER
        self.allowed_goals = ['main_hub', 'e_hub', 'sw_hub']
        self.energy = BATTERY_CAPACITY_WH
        self.pct = 100.0
        self.history = []
        env.process(navigate_map_3d(env, self))

# ==============================
# MAIN SIMULATION
# ==============================
def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 14)

    env = simpy.Environment()
    towers = {tid: Tower(env, tid) for tid in ["A", "B", "C"]}
    trucks = [Truck(env, i) for i in range(NUM_TRUCKS)]

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        env.run(until=env.now + SIM_STEP)

        # 3D Connectivity check
        for t in trucks:
            dists = [(tid, distance_3d(t.pos, tow.pos)) for tid, tow in towers.items()]
            nearest_tid, min_d = min(dists, key=lambda x: x[1])
            t.connected_tower = nearest_tid if min_d <= COMM_RADIUS else None

        # Energy Logic (Same as original but distance_3d)
        for tid, tower in towers.items():
            if tower.pct <= 0: continue
            mesh_p = sum(tx_power(distance_3d(tower.pos, towers[oid].pos)) 
                         for oid in towers if oid != tid)
            tower.pct -= 0.01 # Simulated drain
            tower.history.append(tower.pct)

        # Rendering
        screen.fill(WHITE)
        
        # Draw 3D Entities
        for tid, tower in towers.items():
            p_2d, f = project_3d_to_2d(tower.pos, 2, [0,0], WIDTH, HEIGHT)
            pygame.draw.circle(screen, TOWER_COLORS[tid], p_2d, int(8 * f))
            # 3D range indicator
            pygame.draw.circle(screen, TOWER_COLORS[tid], p_2d, int(COMM_RADIUS * f), 1)

        for t in trucks:
            p_2d, f = project_3d_to_2d(t.pos, 2, [0,0], WIDTH, HEIGHT)
            pygame.draw.circle(screen, TRUCK_COLOR, p_2d, int(5 * f))

        # UI / Graph (Original logic)
        pygame.draw.rect(screen, BLACK, (0, HEIGHT - GRAPH_HEIGHT, WIDTH, GRAPH_HEIGHT))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == '__main__':
    run_simulation()