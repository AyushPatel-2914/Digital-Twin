"""
simulation_3d_physics.py
========================
Integrates:
  • File 1 (simulation.py)   — full truck physics: MPC controller, Kalman
    filter, K-turn maneuvers, Dispatcher, global planner, traffic weights,
    waypoint path stitching, op-state machine.
  • File 2 (simulation_3d.py) — 3-D isometric rendering, SimPy-based tower
    k-median placement, battery drain, comm-link visualisation, battery graph.

Architecture
------------
  Trucks  → driven each frame by Car.update_op_state() + Car.move()
             (no SimPy coroutine for trucks).
  Towers  → still run inside SimPy via kmedian_navigation(); the SimPy env
             is stepped forward once per render frame.
  Camera  → isometric world_to_screen used everywhere; Car positions at z=0
             unless the map supplies elevation data.

Controls
--------
  SPACE / P   — pause / resume
  TAB         — cycle selected truck (shows its planned path)
  SHIFT+0…5   — sim speed 0.5x … 5x
  SCROLL      — zoom
  MMB drag    — pan
  R           — reset view (auto-fit)
  ESC         — quit
"""

# ═══════════════════════════════════════════════════════════════════════════════
#  STANDARD IMPORTS
# ═══════════════════════════════════════════════════════════════════════════════
import sys, os, json, math, random, heapq, pickle
import pygame
import simpy
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
#  PROJECT IMPORTS  (same as the two original files)
# ═══════════════════════════════════════════════════════════════════════════════
from Map import map_loader as map_data          # NODES, LOAD_ZONES, DUMP_ZONES …
from config import (                             # simulation constants
    WIDTH, HEIGHT, PADDING, METERS_TO_PIXELS,
    ZOOM_FACTOR, POINTS_PER_SEGMENT,
    SPEED_MS_EMPTY, WHITE,
)
from utils import Path, KalmanFilter
from car import Car
from dispatcher import Dispatcher
from graphics import draw_active_path           # path ribbon renderer
from tooltip_overlay import get_hovered_entity, draw_tooltip
from Algorithm.planner_registry import (
    load_local_planner,
    DEFAULT_GLOBAL_PLANNER,
    DEFAULT_LOCAL_PLANNER,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  3-D MAP DATA  (from simulation_3d.py)
# ═══════════════════════════════════════════════════════════════════════════════
from Map import MAP_3d_data as MD

NODES_3D      = MD.NODES           # key -> np.array([x, y, z])
ROAD_GRAPH_3D = MD.ROAD_GRAPH      # key -> [(neighbour, weight), …]
VIS_CHAINS    = MD.VISUAL_CHAINS

# Flat lookup arrays for fast nearest-node search (XY only, matching 2-D logic)
_NODE_KEYS = list(NODES_3D.keys())
_NODE_XY   = np.array([[float(NODES_3D[k][0]), float(NODES_3D[k][1])]
                        for k in _NODE_KEYS])


# ═══════════════════════════════════════════════════════════════════════════════
#  SETTINGS  (settings_3d.json — identical to File 2, with physics extras)
# ═══════════════════════════════════════════════════════════════════════════════
def resource_path(f):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, f)
    return f

_cfg_path = resource_path("settings_3d.json")
try:
    with open(_cfg_path, "r", encoding="utf-8") as _fh:
        CFG = json.load(_fh)
    print(f"[OK]  Loaded {_cfg_path}")
except FileNotFoundError:
    CFG = {}
    print(f"[WARN] {_cfg_path} not found — using built-in defaults.")
except json.JSONDecodeError as _e:
    CFG = {}
    print(f"[ERR]  Bad JSON in settings_3d.json ({_e}) — using defaults.")

def cfg(key, default):
    return CFG.get(key, default)

# ── tower / comm parameters ───────────────────────────────────────────────────
NUM_TOWERS          = int(cfg("NUM_TOWERS",            2))
SPEED_TOWER         = float(cfg("SPEED_TOWER",         3.5))
SIM_STEP            = float(cfg("SIM_STEP",            1.0))
TIME_MULTIPLIER     = float(cfg("TIME_MULTIPLIER",     1.0))

MOVE_POWER          = float(cfg("MOVE_POWER",          4.0))
IDLE_POWER          = float(cfg("IDLE_POWER",          3.0))

BATT_VMAX           = float(cfg("BATTERY_VOLTAGE_MAX", 26.5))
BATT_VMIN           = float(cfg("BATTERY_VOLTAGE_MIN", 22.0))
BATT_AH             = float(cfg("BATTERY_AH",         150.0))
BATT_SCALE          = float(cfg("BATTERY_SCALE_FACTOR", 10.0))
BATT_CAP_WH         = (BATT_VMAX * BATT_AH) / BATT_SCALE

P0                  = float(cfg("P0",                  5.0))
PMAX                = float(cfg("PMAX",              250.0))
D0                  = float(cfg("D0",                 80.0))
PATH_LOSS           = float(cfg("PATH_LOSS",           2.5))

_comm_ov            = cfg("COMM_RADIUS_OVERRIDE", None)
COMM_RADIUS         = float(_comm_ov) if _comm_ov is not None \
                      else D0 * (PMAX / P0) ** (1.0 / PATH_LOSS)

REPOSITION_INTERVAL = float(cfg("REPOSITION_INTERVAL", 60.0))
KMEDIAN_MAX_ITER    = int(cfg("KMEDIAN_MAX_ITER",      30))

TOWER_HUB_GOALS = cfg("TOWER_HUB_GOALS",
                       ["main_hub","e_hub","sw_hub","fw_hub","n_hub","s_hub"])

# ── display ───────────────────────────────────────────────────────────────────
GRAPH_HEIGHT = int(cfg("GRAPH_HEIGHT",    200))
MAP_H        = HEIGHT - GRAPH_HEIGHT

ISO_ANG = math.radians(float(cfg("ISO_ANGLE_DEG", 30)))
ISO_SX  = float(cfg("ISO_SCALE_X", 1.00))
ISO_SY  = float(cfg("ISO_SCALE_Y", 0.55))
Z_SCALE = float(cfg("Z_SCALE",     0.18))

SHOW_LINKS = bool(cfg("SHOW_COMM_LINKS",    True))
SHOW_GRAPH = bool(cfg("SHOW_BATTERY_GRAPH", True))
SHOW_HUD   = bool(cfg("SHOW_HUD",           True))

_tc  = cfg("TOWER_COLOR",      [220, 30,  30])
_tkc = cfg("TRUCK_COLOR",      [  0, 60, 180])
_bg  = cfg("BACKGROUND_COLOR", [ 15, 18,  25])
TOWER_BASE_COLOR = tuple(int(v) for v in _tc)
TRUCK_COLOR_3D   = tuple(int(v) for v in _tkc)
BG_COLOR         = tuple(int(v) for v in _bg)
WHITE_3D         = (255, 255, 255)

print(f"\n[SIM]  Towers={NUM_TOWERS}  CommR={COMM_RADIUS:.1f}m  "
      f"BattCap={BATT_CAP_WH:.1f}Wh  Reposition every {REPOSITION_INTERVAL}s")


# ═══════════════════════════════════════════════════════════════════════════════
#  ISOMETRIC CAMERA
# ═══════════════════════════════════════════════════════════════════════════════
def _compute_autofit():
    pad = 60
    pts = []
    for n in NODES_3D.values():
        x, y, z = float(n[0]), float(n[1]), float(n[2])
        sx =  (x - y) * math.cos(ISO_ANG) * ISO_SX
        sy =  (x + y) * math.sin(ISO_ANG) * ISO_SY - z * Z_SCALE
        pts.append((sx, sy))
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    rw = max(xs) - min(xs);   rh = max(ys) - min(ys)
    if rw == 0 or rh == 0:
        return 1.0, WIDTH // 2, MAP_H // 2
    scale = min((WIDTH - 2*pad) / rw, (MAP_H - 2*pad) / rh)
    ox = WIDTH  // 2 - ((min(xs) + max(xs)) / 2) * scale
    oy = MAP_H  // 2 - ((min(ys) + max(ys)) / 2) * scale
    return scale, ox, oy

_AUTO_SCALE, _AUTO_OX, _AUTO_OY = _compute_autofit()
view = {"scale": _AUTO_SCALE, "ox": _AUTO_OX, "oy": _AUTO_OY}

def world_to_screen(pos3, vs=None):
    """3-D isometric projection → pixel coords."""
    if vs is None:
        vs = view
    x, y = float(pos3[0]), float(pos3[1])
    z    = float(pos3[2]) if len(pos3) > 2 else 0.0
    s    = vs["scale"]
    sx   = (x - y) * math.cos(ISO_ANG) * ISO_SX * s
    sy   = (x + y) * math.sin(ISO_ANG) * ISO_SY * s - z * Z_SCALE * s
    return int(sx + vs["ox"]), int(sy + vs["oy"])

def car_to_screen(car):
    """Project a Car's 2-D position through isometric transform (z=0)."""
    return world_to_screen([car.x_m, car.y_m, 0.0])


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG LOADERS  (unchanged from File 1)
# ═══════════════════════════════════════════════════════════════════════════════
def _load_json_file(path, label):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {label}: {e}")
    return None

def load_mine_config():
    default = {"truck_count": 8, "coal_capacities": {}}
    config  = {"truck_count": 5, "coal_capacities": {}}
    loaded  = _load_json_file("Map/mine_config.json", "mine config")
    if loaded:
        config["truck_count"]    = loaded.get("truck_count", 5)
        config["coal_capacities"]= loaded.get("coal_capacities", {})
        print(f"Loaded mine config: {config['truck_count']} trucks.")
    else:
        if os.path.exists("truck.txt"):
            try:
                with open("truck.txt") as f:
                    config["truck_count"] = int(f.read().strip())
            except ValueError:
                pass
        print(f"Using default mine config ({config['truck_count']} trucks).")
    return config

def load_algorithm_config():
    config  = {"global_planner": DEFAULT_GLOBAL_PLANNER,
               "local_planner":  DEFAULT_LOCAL_PLANNER}
    loaded  = _load_json_file("algorithm_config.json", "algorithm config")
    if loaded:
        config["global_planner"] = loaded.get("global_planner", DEFAULT_GLOBAL_PLANNER)
        config["local_planner"]  = loaded.get("local_planner",  DEFAULT_LOCAL_PLANNER)
        print(f"Algorithm config: global='{config['global_planner']}' "
              f"local='{config['local_planner']}'.")
    else:
        print(f"Using default algorithm config.")
    return config


# ═══════════════════════════════════════════════════════════════════════════════
#  WAYPOINT STITCHING  (unchanged from File 1)
# ═══════════════════════════════════════════════════════════════════════════════
def get_path_from_nodes(route_node_names, waypoints_map):
    final_waypoints = []
    if not route_node_names:
        return []
    for i in range(len(route_node_names) - 1):
        seg_start, seg_end = route_node_names[i], route_node_names[i + 1]
        found = False
        for chain_tuple, waypoints in waypoints_map.items():
            try:
                idx = chain_tuple.index(seg_start)
                if idx + 1 < len(chain_tuple) and chain_tuple[idx + 1] == seg_end:
                    s = idx * POINTS_PER_SEGMENT
                    e = (idx + 1) * POINTS_PER_SEGMENT
                    final_waypoints.extend(waypoints[s:e])
                    found = True
                    break
                idx = chain_tuple.index(seg_end)
                if idx + 1 < len(chain_tuple) and chain_tuple[idx + 1] == seg_start:
                    s = idx * POINTS_PER_SEGMENT
                    e = (idx + 1) * POINTS_PER_SEGMENT
                    seg = waypoints[s:e + 1]
                    final_waypoints.extend(seg[::-1][:-1])
                    found = True
                    break
            except ValueError:
                continue
        if not found:
            pass
    if final_waypoints and route_node_names[-1] in map_data.NODES:
        final_waypoints.append(map_data.NODES[route_node_names[-1]])
    return final_waypoints


# ═══════════════════════════════════════════════════════════════════════════════
#  K-MEDIAN ALGORITHM  (unchanged from File 2)
# ═══════════════════════════════════════════════════════════════════════════════
def dist_xy(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

def tx_power(d):
    return 0.0 if d == 0 else P0 * (d / D0) ** PATH_LOSS

def nearest_node_to(pos):
    diffs = _NODE_XY - np.array([float(pos[0]), float(pos[1])])
    return _NODE_KEYS[int(np.argmin((diffs ** 2).sum(axis=1)))]

def truck_to_nearest_nodes(cars):
    """Map each Car to its nearest 3-D road node (XY only)."""
    nearest = []
    for car in cars:
        diffs = _NODE_XY - np.array([float(car.x_m), float(car.y_m)])
        nearest.append(_NODE_KEYS[int(np.argmin((diffs ** 2).sum(axis=1)))])
    return nearest

def compute_kmedian_cost(facility_nodes, demand_nodes):
    total = 0.0
    for dn in demand_nodes:
        dc = NODES_3D[dn]
        total += min(dist_xy(dc, NODES_3D[fn]) for fn in facility_nodes)
    return total

def local_search_k_median(truck_nodes, k, max_iterations=None):
    if max_iterations is None:
        max_iterations = KMEDIAN_MAX_ITER
    all_nodes = _NODE_KEYS
    if k >= len(all_nodes):
        return list(all_nodes[:k])
    facilities   = random.sample(all_nodes, k)
    current_cost = compute_kmedian_cost(facilities, truck_nodes)
    for _ in range(max_iterations):
        improved = False
        for i in range(k):
            for candidate in all_nodes:
                if candidate in facilities:
                    continue
                new_fac  = facilities.copy()
                new_fac[i] = candidate
                new_cost = compute_kmedian_cost(new_fac, truck_nodes)
                if new_cost < current_cost:
                    facilities   = new_fac
                    current_cost = new_cost
                    improved     = True
                    break
            if improved:
                break
        if not improved:
            break
    return facilities


# ═══════════════════════════════════════════════════════════════════════════════
#  A*  (for tower navigation on the 3-D road graph)
# ═══════════════════════════════════════════════════════════════════════════════
def a_star_3d(graph, start, goal):
    if start == goal:
        return [start]
    open_set  = [(0, start)]
    came_from = {}
    g_cost    = {start: 0.0}
    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal:
            path = []
            while cur in came_from:
                path.append(cur); cur = came_from[cur]
            path.append(start)
            return list(reversed(path))
        for nb, w in graph.get(cur, []):
            tg = g_cost.get(cur, 1e18) + w
            if tg < g_cost.get(nb, 1e18):
                came_from[nb] = cur
                g_cost[nb]    = tg
                heapq.heappush(open_set, (tg, nb))
    return []


# ═══════════════════════════════════════════════════════════════════════════════
#  TOWER ENTITY  (unchanged from File 2)
# ═══════════════════════════════════════════════════════════════════════════════
class Tower:
    def __init__(self, env, tid, graph_color, towers_dict, cars_ref):
        self.id           = tid
        self.pos          = [0.0, 0.0, 0.0]
        self.speed        = SPEED_TOWER
        self.energy       = BATT_CAP_WH
        self.pct          = 100.0
        self.history      = []
        self.graph_color  = graph_color
        self.current_node = None
        self.target_node  = None
        self.is_moving    = False
        # cars_ref is a live list of Car objects (used by coordinator)
        self._cars_ref    = cars_ref
        env.process(self._navigate(env, towers_dict))

    def _navigate(self, env, towers_dict):
        """SimPy coroutine: k-median coordinator + tower navigation."""
        tower_index = int(self.id)
        valid_hubs  = [g for g in TOWER_HUB_GOALS if g in NODES_3D]
        if not valid_hubs:
            valid_hubs = [_NODE_KEYS[0]]
        start_node  = valid_hubs[tower_index % len(valid_hubs)]

        p = NODES_3D[start_node].astype(float)
        self.pos[:] = [p[0], p[1], p[2]]
        self.current_node = start_node
        self.target_node  = start_node

        # Stagger startup
        yield env.timeout(tower_index * SIM_STEP)

        while True:
            # ── Coordinator (tower "0") runs k-median ─────────────────
            if self.id == "0":
                cars        = self._cars_ref
                truck_nodes = truck_to_nearest_nodes(cars)
                k           = NUM_TOWERS
                while len(truck_nodes) < k:
                    truck_nodes.append(random.choice(_NODE_KEYS))

                print(f"[K-MEDIAN] t={env.now:.0f}s  "
                      f"demand={len(truck_nodes)}  k={k} …")
                medians = local_search_k_median(truck_nodes, k)
                print(f"[K-MEDIAN] done → {medians}")

                for i, tid in enumerate(sorted(towers_dict.keys(),
                                               key=lambda x: int(x))):
                    towers_dict[tid].target_node = (
                        medians[i] if i < len(medians) else self.current_node
                    )

            yield env.timeout(SIM_STEP)

            # ── Navigate toward assigned target ───────────────────────
            target = self.target_node
            if target != self.current_node and target in NODES_3D:
                route = a_star_3d(ROAD_GRAPH_3D, self.current_node, target)

                if route and len(route) >= 2:
                    for seg_i in range(len(route) - 1):
                        if self.target_node != target:
                            break
                        p_from  = NODES_3D[route[seg_i]].astype(float)
                        p_to    = NODES_3D[route[seg_i + 1]].astype(float)
                        seg_len = float(np.linalg.norm(p_to - p_from))
                        if seg_len == 0:
                            continue
                        t_dist = 0.0
                        while t_dist < seg_len:
                            if self.target_node != target:
                                break
                            t_dist = min(t_dist + self.speed * SIM_STEP,
                                         seg_len)
                            alpha  = t_dist / seg_len
                            ip     = p_from + alpha * (p_to - p_from)
                            self.pos[:] = [float(ip[0]), float(ip[1]),
                                           float(ip[2])]
                            self.is_moving = True
                            yield env.timeout(SIM_STEP)
                        self.current_node = route[seg_i + 1]

                    if self.target_node == target:
                        p = NODES_3D[target].astype(float)
                        self.pos[:] = [p[0], p[1], p[2]]
                        self.current_node = target
                        self.is_moving    = False

            sleep_time = max(SIM_STEP, REPOSITION_INTERVAL - SIM_STEP)
            self.is_moving = False
            yield env.timeout(sleep_time)

def tower_graph_colors(n):
    return [
        (230, int(40 + (i / max(n - 1, 1)) * 200),
             int(40 + (i / max(n - 1, 1)) * 100))
        for i in range(n)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
#  ROAD RENDERING HELPERS  (from File 2, unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
_zvals = [float(n[2]) for n in NODES_3D.values()]
_z_min, _z_max = min(_zvals), max(_zvals)
_z_rng = max(_z_max - _z_min, 1.0)

def elevation_color(z):
    t = (float(z) - _z_min) / _z_rng
    return (int(60 + t * 80), int(100 + t * 130), int(60 + t * 50))

def build_road_segments(vs):
    segs = []
    for chain in VIS_CHAINS:
        valid = [n for n in chain if n in NODES_3D]
        if len(valid) < 2:
            continue
        pts = [world_to_screen(NODES_3D[n], vs) for n in valid]
        zs  = [float(NODES_3D[n][2]) for n in valid]
        segs.append((pts, elevation_color(sum(zs) / len(zs))))
    return segs

def battery_color(pct):
    if pct > 60:
        return (int(255 * (1 - (pct - 60) / 40)), 220, 40)
    elif pct > 20:
        return (255, int(70 + 150 * (pct - 20) / 40), 20)
    return (255, 40, 40)

def draw_battery_graph(screen, towers, font, graph_rect):
    pygame.draw.rect(screen, (25, 25, 35), graph_rect)
    pygame.draw.rect(screen, (80, 80, 80), graph_rect, 1)
    for pct in (0, 25, 50, 75, 100):
        gy = graph_rect.bottom - (pct / 100) * graph_rect.height
        pygame.draw.line(screen, (50, 50, 65),
                         (graph_rect.left, int(gy)),
                         (graph_rect.right, int(gy)), 1)
        screen.blit(font.render(f"{pct}%", True, (110, 110, 130)),
                    (graph_rect.left - 34, int(gy) - 7))
    for tw in towers.values():
        if len(tw.history) < 2:
            continue
        n   = len(tw.history)
        pts = [(int(graph_rect.x + (i / n) * graph_rect.width),
                int(graph_rect.bottom - (p / 100) * graph_rect.height))
               for i, p in enumerate(tw.history)]
        pygame.draw.lines(screen, tw.graph_color, False, pts, 2)
        lx, ly = pts[-1]
        screen.blit(font.render(f"{tw.id}:{tw.pct:.0f}%", True, tw.graph_color),
                    (min(lx + 4, graph_rect.right - 70),
                     max(ly - 10, graph_rect.top + 2)))

def draw_elevation_bar(screen, font):
    bx, by, bw, bh = WIDTH - 40, 40, 12, 120
    for i in range(bh):
        t = 1 - i / bh
        pygame.draw.line(screen,
                         (int(20 + t * 80), int(60 + t * 120), int(20 + t * 40)),
                         (bx, by + i), (bx + bw, by + i))
    pygame.draw.rect(screen, (120, 120, 120), (bx, by, bw, bh), 1)
    hi = font.render(f"{_z_max:.0f}m", True, (200, 230, 180))
    lo = font.render(f"{_z_min:.0f}m", True, (140, 180, 120))
    screen.blit(hi, (bx - hi.get_width() - 2, by - 2))
    screen.blit(lo, (bx - lo.get_width() - 2, by + bh - 8))


# ═══════════════════════════════════════════════════════════════════════════════
#  TRUCK DRAWING IN ISO VIEW
# ═══════════════════════════════════════════════════════════════════════════════
def draw_truck_iso(screen, car, is_selected, font_small):
    """
    Draw a Car at its isometric screen position.
    The heading arrow is projected into the iso plane so orientation is visible.
    """
    sp = car_to_screen(car)
    # Skip off-screen trucks
    if not (0 <= sp[0] < WIDTH and 0 <= sp[1] < MAP_H + 60):
        return

    # Body colour: blue when empty, orange when loaded
    loaded = car.current_mass_kg > 1000  # rough threshold; tune as needed
    body_col = (200, 110, 0) if loaded else TRUCK_COLOR_3D

    # Shadow
    pygame.draw.circle(screen, (0, 20, 60), (sp[0] + 2, sp[1] + 2), 7)
    # Body
    pygame.draw.circle(screen, body_col, sp, 7)
    # Highlight ring
    ring_col = (255, 220, 80) if is_selected else (120, 160, 255)
    pygame.draw.circle(screen, ring_col, sp, 7, 2 if is_selected else 1)

    # Heading arrow projected into iso
    arrow_len = 14
    world_dx = math.cos(car.angle) * arrow_len
    world_dy = math.sin(car.angle) * arrow_len
    tip = world_to_screen([car.x_m + world_dx, car.y_m + world_dy, 0.0])
    pygame.draw.line(screen, ring_col, sp, tip, 2)

    # Label
    if is_selected:
        lbl = font_small.render(
            f"T{car.id} {car.speed_ms * 3.6:.0f}km/h", True, (255, 255, 180))
        screen.blit(lbl, (sp[0] + 9, sp[1] - 14))


def draw_path_iso(screen, path, car):
    """
    Draw the planned waypoint path for the selected car in iso projection.
    """
    # FIX: Changed path.waypoints to path.wp to match your utils.Path class structure
    if path is None or len(path.wp) < 2:
        return
    pts = [world_to_screen([wp[0], wp[1], 0.0]) for wp in path.wp]
    
    # Clip to map area
    pts = [(x, y) for x, y in pts if 0 <= x < WIDTH and 0 <= y < MAP_H + 60]
    if len(pts) >= 2:
        pygame.draw.lines(screen, (255, 220, 80), False, pts, 1)

# ═══════════════════════════════════════════════════════════════════════════════
#  HUD
# ═══════════════════════════════════════════════════════════════════════════════
def draw_hud_combined(screen, font, cars, towers, sel_idx, sim_speed,
                      paused, sim_time, fontB):
    sel = cars[sel_idx]
    pcts    = [tw.pct for tw in towers.values()]
    avg_pct = sum(pcts) / max(len(pcts), 1)
    moving  = sum(1 for tw in towers.values() if tw.is_moving)

    lines = [
        f"Truck {sel.id} | {sel.speed_ms * 3.6:.1f} km/h | "
        f"{sel.current_mass_kg:.0f} kg | {sel.op_state}  "
        f"(TAB to switch)",
        f"Trucks: {len(cars)}  |  Towers: {NUM_TOWERS}  "
        f"avg batt: {avg_pct:.1f}%  moving towers: {moving}",
        f"Sim speed: {sim_speed}x  |  {'PAUSED' if paused else 'Running'}  "
        f"|  t={sim_time:.0f}s",
        "SPACE=pause  TAB=truck  SHIFT+0–5=speed  SCROLL=zoom  R=reset  ESC=quit",
    ]
    for i, line in enumerate(lines):
        surf = font.render(line, True, WHITE_3D)
        bg   = pygame.Surface((surf.get_width() + 6, surf.get_height() + 2),
                               pygame.SRCALPHA)
        bg.fill((0, 0, 0, 155))
        screen.blit(bg,   (8, 8 + i * 18 - 1))
        screen.blit(surf, (11, 8 + i * 18))

    # Per-tower status rows
    y0 = 8 + len(lines) * 18 + 4
    for tw in towers.values():
        conn = sum(1 for c in cars
                   if math.hypot(c.x_m - tw.pos[0],
                                 c.y_m - tw.pos[1]) <= COMM_RADIUS)
        mv  = "→" if tw.is_moving else "·"
        tgt = tw.target_node if tw.target_node else "—"
        row = (f"  [{mv}] Tower {tw.id}  "
               f"{tw.current_node}→{tgt}  "
               f"{tw.pct:.0f}%  trucks:{conn}")
        surf = font.render(row, True, tw.graph_color)
        bg   = pygame.Surface((surf.get_width() + 6, surf.get_height() + 2),
                               pygame.SRCALPHA)
        bg.fill((0, 0, 0, 130))
        screen.blit(bg,   (8, y0 - 1))
        screen.blit(surf, (11, y0))
        y0 += 17


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption(
        "Mining Sim — 3D Iso View + Full Truck Physics + K-Median Towers")
    clock     = pygame.time.Clock()
    font      = pygame.font.SysFont("Consolas", 13)
    fontB     = pygame.font.SysFont("Consolas", 15, bold=True)
    font_hud  = pygame.font.SysFont("Consolas", 14)

    # ── Load configs (File 1 logic) ───────────────────────────────────
    mine_cfg  = load_mine_config()
    algo_cfg  = load_algorithm_config()
    truck_count      = mine_cfg["truck_count"]
    coal_capacities  = mine_cfg["coal_capacities"]
    global_planner_name = algo_cfg["global_planner"]
    local_planner_name  = algo_cfg["local_planner"]

    try:
        local_planner = load_local_planner(local_planner_name)
    except Exception as e:
        print(f"Local planner '{local_planner_name}' failed ({e}). Falling back.")
        local_planner = load_local_planner(DEFAULT_LOCAL_PLANNER)

    print(f"[SIM] Starting with {truck_count} trucks.")

    # ── Load pre-calculated waypoints & route cache (File 1) ─────────
    wp_file    = "Map/waypoints.pkl"
    cache_file = "Map/map_cache.pkl"

    if not os.path.exists(wp_file):
        print(f"Error: '{wp_file}' not found."); return
    with open(wp_file, "rb") as f:
        waypoints_map = pickle.load(f)

    if not os.path.exists(cache_file):
        print(f"Error: '{cache_file}' not found."); return
    with open(cache_file, "rb") as f:
        cache_data  = pickle.load(f)
        road_graph  = cache_data.get("road_graph",  {})
        route_cache = cache_data.get("route_cache", {})

    # ── Spur patching (File 1, unchanged) ────────────────────────────
    all_terminals = set(map_data.LOAD_ZONES + map_data.DUMP_ZONES)
    incoming_map  = {}
    outgoing_counts = {}
    for start_node, edges in road_graph.items():
        outgoing_counts[start_node] = len(edges)
        for target, weight in edges:
            incoming_map.setdefault(target, []).append(start_node)

    def trace_back_and_patch(current_node, visited):
        if current_node in visited: return
        visited.add(current_node)
        parents = incoming_map.get(current_node, [])
        if not parents: return
        parent = parents[0]
        if current_node not in road_graph:
            road_graph[current_node] = []
        if not any(t == parent for t, _ in road_graph[current_node]):
            p1 = map_data.NODES[current_node]
            p2 = map_data.NODES[parent]
            dist = float(np.linalg.norm(p1 - p2))
            road_graph[current_node].append((parent, dist))
            print(f"Patched spur: {current_node} -> {parent}")
        pin  = len(incoming_map.get(parent, []))
        pout = outgoing_counts.get(parent, 0)
        if not ("hub" in parent or (pin > 1 and pout > 1)):
            trace_back_and_patch(parent, visited)

    for zone in all_terminals:
        trace_back_and_patch(zone, set())

    # ── Dispatcher + Car creation (File 1) ───────────────────────────
    dispatcher = Dispatcher(
        road_graph,
        coal_capacities=coal_capacities,
        global_planner_name=global_planner_name,
    )
    cars = []
    kfs  = []

    start_nodes = list(map_data.DUMP_ZONES)
    random.shuffle(start_nodes)

    for i in range(truck_count):
        sn      = start_nodes[i % len(start_nodes)]
        new_car = Car(i + 1, 0, 0, 0)
        new_car.current_node_name = sn
        if sn in map_data.NODES:
            new_car.x_m = map_data.NODES[sn][0]
            new_car.y_m = map_data.NODES[sn][1]
        cars.append(new_car)
        kfs.append(KalmanFilter(dt=1.0 / 60.0,
                                start_x=new_car.x_m, start_y=new_car.y_m))

    print("[SIM] Running initial global optimisation…")
    dispatcher.update_global_plan(cars)

    for idx, car in enumerate(cars):
        target = dispatcher.assign_task(car)
        car.target_node_name = target
        route  = local_planner.compute_route(
            road_graph, car.current_node_name, target, cache=route_cache)
        if not route: continue
        wp = get_path_from_nodes(route, waypoints_map)
        if not wp or len(wp) < 2: continue
        pos   = wp[0]
        angle = math.atan2(wp[1][1] - wp[0][1], wp[1][0] - wp[0][0])
        car.x_m, car.y_m, car.angle = pos[0], pos[1], angle
        car.path             = Path(wp)
        car.op_state         = "GOING_TO_ENDPOINT"
        car.desired_speed_ms = SPEED_MS_EMPTY
        kfs[idx] = KalmanFilter(dt=1.0 / 60.0, start_x=pos[0], start_y=pos[1])

    # Initial MPC run
    for car in cars:
        car.run_mpc([])

    # ── SimPy env + Towers (File 2) ───────────────────────────────────
    env       = simpy.Environment()
    tcolors   = tower_graph_colors(NUM_TOWERS)
    tower_ids = [str(i) for i in range(NUM_TOWERS)]
    towers    = {}
    for i, tid in enumerate(tower_ids):
        towers[tid] = Tower(env, tid, tcolors[i], towers, cars)

    # ── View / pan state ─────────────────────────────────────────────
    road_segs       = build_road_segments(view)
    graph_rect      = pygame.Rect(46, HEIGHT - GRAPH_HEIGHT + 18,
                                   WIDTH - 92, GRAPH_HEIGHT - 36)
    dragging        = False
    drag_start      = (0, 0)
    view_start_snap = (_AUTO_OX, _AUTO_OY)
    selected_car_idx = 0
    paused          = False
    sim_speed       = 1.0

    def _comm_px():
        a = world_to_screen([0, 0, 0])
        b = world_to_screen([COMM_RADIUS, 0, 0])
        return max(4, abs(b[0] - a[0]))
    comm_px = _comm_px()

    # ── Physics timers ────────────────────────────────────────────────
    mpc_timer            = 0.0
    traffic_update_timer = 0.0
    global_opt_timer     = 0.0
    MPC_INTERVAL         = 0.1
    TRAFFIC_UPDATE_INTERVAL = 1.0
    GLOBAL_OPT_INTERVAL  = 30.0

    # ── SimPy stepping state ──────────────────────────────────────────
    simpy_accum   = 0.0   # fractional sim-seconds accumulated
    sim_clock_s   = 0.0   # total simulated seconds

    # ═════════════════════════════════════════════════════════════════
    #  MAIN LOOP
    # ═════════════════════════════════════════════════════════════════
    running = True
    while running:
        frame_dt = clock.tick(60) / 1000.0
        if frame_dt == 0:
            continue

        sim_dt = 0.0 if paused else frame_dt * sim_speed

        # ── Event handling ────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_SPACE, pygame.K_p):
                    paused = not paused
                elif event.key == pygame.K_r:
                    view["scale"] = _AUTO_SCALE
                    view["ox"]    = _AUTO_OX
                    view["oy"]    = _AUTO_OY
                    road_segs = build_road_segments(view)
                    comm_px   = _comm_px()
                elif event.key == pygame.K_TAB:
                    selected_car_idx = (selected_car_idx + 1) % len(cars)
                elif event.mod & pygame.KMOD_SHIFT:
                    speed_map = {pygame.K_0: 0.5, pygame.K_1: 1.0,
                                 pygame.K_2: 2.0, pygame.K_3: 3.0,
                                 pygame.K_4: 4.0, pygame.K_5: 5.0}
                    if event.key in speed_map:
                        sim_speed = speed_map[event.key]
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 2:
                dragging        = True
                drag_start      = event.pos
                view_start_snap = (view["ox"], view["oy"])
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 2:
                dragging = False
            elif event.type == pygame.MOUSEMOTION and dragging:
                view["ox"] = view_start_snap[0] + event.pos[0] - drag_start[0]
                view["oy"] = view_start_snap[1] + event.pos[1] - drag_start[1]
                road_segs  = build_road_segments(view)
            elif event.type == pygame.MOUSEWHEEL:
                f = 1.1 if event.y > 0 else 0.9
                view["scale"] = max(0.05, min(8.0, view["scale"] * f))
                road_segs = build_road_segments(view)
                comm_px   = _comm_px()

        # ── Simulation update ─────────────────────────────────────────
        if sim_dt > 0:
            sim_clock_s          += sim_dt
            mpc_timer            += sim_dt
            traffic_update_timer += sim_dt
            global_opt_timer     += sim_dt

            # -- SimPy tower advancement --
            simpy_accum += sim_dt
            while simpy_accum >= SIM_STEP:
                env.run(until=env.now + SIM_STEP)
                simpy_accum -= SIM_STEP

                # Battery drain per SimPy step
                for tid, tw in towers.items():
                    if tw.pct <= 0:
                        continue
                    mesh_p = sum(
                        tx_power(dist_xy(tw.pos, otw.pos))
                        for oid, otw in towers.items()
                        if oid != tid and dist_xy(tw.pos, otw.pos) <= COMM_RADIUS
                    )
                    truck_p = sum(
                        tx_power(dist_xy(tw.pos, [c.x_m, c.y_m, 0.0]))
                        for c in cars
                        if math.hypot(c.x_m - tw.pos[0],
                                      c.y_m - tw.pos[1]) <= COMM_RADIUS
                    )
                    total_p = (IDLE_POWER
                               + (MOVE_POWER if tw.is_moving else 0.0)
                               + mesh_p + truck_p)
                    drain_noise = np.random.normal(loc=0.0, scale=0.5)
                    total_p = max(0.1, total_p + drain_noise)
                    tw.energy -= total_p * (SIM_STEP / 3600.0) * TIME_MULTIPLIER
                    tw.pct     = max(0.0, (tw.energy / BATT_CAP_WH) * 100.0)
                    tw.history.append(tw.pct)

            # -- Traffic update (1 Hz) --
            if traffic_update_timer >= TRAFFIC_UPDATE_INTERVAL:
                traffic_update_timer = 0.0
                dispatcher.update_traffic_weights(cars)

            # -- Global optimisation (0.033 Hz) --
            if global_opt_timer >= GLOBAL_OPT_INTERVAL:
                global_opt_timer = 0.0
                dispatcher.update_global_plan(cars)

            # -- MPC update (10 Hz) --
            if mpc_timer >= MPC_INTERVAL:
                mpc_timer          = 0.0
                all_trajectories   = [c.planned_trajectory for c in cars]
                for i, car in enumerate(cars):
                    if car.op_state == "TURNING_AROUND":
                        continue
                    others = [all_trajectories[j]
                              for j in range(len(cars)) if j != i]
                    car.run_mpc(others, other_cars=cars)

            # -- Physics update (60 Hz) --
            for idx, car in enumerate(cars):
                kf        = kfs[idx]
                est_pos_m = np.array([kf.x[0], kf.x[2]])

                if car.path and car.op_state != "TURNING_AROUND":
                    car.s_path_m = car.path.project(est_pos_m, car.s_path_m)

                direction, base_speed = car.update_op_state(sim_dt, dispatcher)
                car.desired_speed_ms  = base_speed

                if car.needs_new_path:
                    car.needs_new_path    = False
                    car.current_node_name = car.target_node_name
                    new_target            = dispatcher.assign_task(car)
                    car.target_node_name  = new_target
                    print(f"Car {car.id}: routing "
                          f"{car.current_node_name} → {new_target}")

                    route_nodes = local_planner.compute_route(
                        dispatcher.get_graph(),
                        car.current_node_name,
                        new_target,
                        cache=route_cache,
                    )
                    if route_nodes:
                        wp_m = get_path_from_nodes(route_nodes, waypoints_map)
                        if wp_m and len(wp_m) >= 2:
                            car.path = Path(wp_m)
                            car.s_path_m = 0.0
                            car.turn_target_angle = math.atan2(
                                wp_m[1][1] - wp_m[0][1],
                                wp_m[1][0] - wp_m[0][0])
                            car.current_mpc_control = np.zeros(2)
                            car.mpc.prev_u = np.zeros_like(car.mpc.prev_u)
                            print(f"Car {car.id}: K-turn to "
                                  f"{math.degrees(car.turn_target_angle):.1f}°")
                        else:
                            print(f"Car {car.id}: path too short!")
                    else:
                        print(f"Car {car.id}: no route found!")

                # Physics / manoeuvre
                if car.op_state == "TURNING_AROUND":
                    done = car.execute_turn_step(sim_dt)
                    if done:
                        if car.path:
                            car.s_path_m = car.path.project(
                                np.array([car.x_m, car.y_m]), 0.0)
                        kf.x[0] = car.x_m;  kf.x[1] = 0.0
                        kf.x[2] = car.y_m;  kf.x[3] = 0.0
                        car.run_mpc([], other_cars=cars)
                        print(f"Car {car.id}: K-turn complete → {car.op_state}")
                else:
                    car.move(sim_dt)

                # Kalman filter
                accel_vec = np.array([
                    car.accel_ms2 * math.cos(car.angle),
                    car.accel_ms2 * math.sin(car.angle)])
                kf.predict(u=accel_vec)
                kf.update(z=car.get_noisy_measurement())

        # ── Rendering ─────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        # Roads
        for pts, col in road_segs:
            if len(pts) >= 2:
                dc = tuple(max(0, c - 30) for c in col)
                pygame.draw.lines(screen, dc, False, pts, 1)

        # Comm links: tower → connected trucks
        if SHOW_LINKS:
            for tid, tw in towers.items():
                if tw.pct <= 0: continue
                tp = world_to_screen(tw.pos)
                for car in cars:
                    if math.hypot(car.x_m - tw.pos[0],
                                  car.y_m - tw.pos[1]) <= COMM_RADIUS:
                        pygame.draw.line(screen, (50, 50, 90),
                                         tp, car_to_screen(car), 1)

        # Tower target movement lines
        for tw in towers.values():
            if tw.target_node and tw.target_node in NODES_3D and tw.is_moving:
                pygame.draw.line(screen, (100, 100, 30),
                                 world_to_screen(tw.pos),
                                 world_to_screen(NODES_3D[tw.target_node]), 1)

        # Selected truck path
        sel_car = cars[selected_car_idx]
        if sel_car.path:
            draw_path_iso(screen, sel_car.path, sel_car)

        # Draw all trucks (iso projection, physics-driven positions)
        for idx, car in enumerate(cars):
            draw_truck_iso(screen, car,
                           is_selected=(idx == selected_car_idx), font_small=font)

        # Towers
        for tid, tw in towers.items():
            if tw.pct <= 0: continue
            sp   = world_to_screen(tw.pos)
            bcol = battery_color(tw.pct)

            if tw.target_node and tw.target_node in NODES_3D:
                tgt_sp = world_to_screen(NODES_3D[tw.target_node])
                pygame.draw.circle(screen, tw.graph_color, tgt_sp, 5, 2)

            ex = int(comm_px * math.cos(ISO_ANG) * ISO_SX)
            ey = int(comm_px * math.sin(ISO_ANG) * ISO_SY)
            if ex > 3 and ey > 3:
                pygame.draw.ellipse(screen, bcol,
                    pygame.Rect(sp[0] - ex, sp[1] - ey, 2 * ex, 2 * ey), 1)

            pygame.draw.circle(screen, (60, 0, 0),  (sp[0] + 2, sp[1] + 2), 9)
            pygame.draw.circle(screen, bcol,          sp, 9)
            pygame.draw.circle(screen, WHITE_3D,      sp, 9, 1)
            if tw.is_moving:
                pygame.draw.circle(screen, WHITE_3D, sp, 3)
            screen.blit(font.render(tw.id, True, WHITE_3D),
                        (sp[0] + 12, sp[1] - 8))
            screen.blit(font.render(f"{tw.pct:.0f}%", True, bcol),
                        (sp[0] + 12, sp[1] + 4))

        # Zone markers
        for zn in map_data.LOAD_ZONES:
            if zn in map_data.NODES:
                p2d = map_data.NODES[zn]
                sp  = world_to_screen([p2d[0], p2d[1], 0.0])
                if 0 <= sp[0] < WIDTH and 0 <= sp[1] < MAP_H:
                    pygame.draw.circle(screen, (0, 200, 80), sp, 5)
        for zn in map_data.DUMP_ZONES:
            if zn in map_data.NODES:
                p2d = map_data.NODES[zn]
                sp  = world_to_screen([p2d[0], p2d[1], 0.0])
                if 0 <= sp[0] < WIDTH and 0 <= sp[1] < MAP_H:
                    pygame.draw.circle(screen, (255, 180, 0), sp, 5)

        draw_elevation_bar(screen, font)

        # Battery graph panel
        if SHOW_GRAPH:
            pygame.draw.rect(screen, (8, 8, 14),
                             (0, HEIGHT - GRAPH_HEIGHT, WIDTH, GRAPH_HEIGHT))
            draw_battery_graph(screen, towers, font, graph_rect)
            pcts    = [tw.pct for tw in towers.values()]
            std_now = float(np.std(pcts)) if pcts else 0.0
            ts = fontB.render(
                f"Tower Battery (%)  —  K-Median (local search)  "
                f"|  Std-dev: {std_now:.2f}%",
                True, (180, 180, 200))
            screen.blit(ts, (graph_rect.centerx - ts.get_width() // 2,
                             HEIGHT - GRAPH_HEIGHT + 2))

        # Legend
        pygame.draw.circle(screen, (0, 200, 80),
                           (WIDTH - 120, HEIGHT - GRAPH_HEIGHT - 22), 5)
        screen.blit(font.render("Load zone", True, (180, 230, 180)),
                    (WIDTH - 110, HEIGHT - GRAPH_HEIGHT - 28))
        pygame.draw.circle(screen, (255, 180, 0),
                           (WIDTH - 120, HEIGHT - GRAPH_HEIGHT - 8), 5)
        screen.blit(font.render("Dump zone", True, (255, 210, 130)),
                    (WIDTH - 110, HEIGHT - GRAPH_HEIGHT - 14))

        pygame.draw.line(screen, (60, 60, 70),
                         (0, HEIGHT - GRAPH_HEIGHT),
                         (WIDTH, HEIGHT - GRAPH_HEIGHT), 1)

        # HUD overlay
        if SHOW_HUD:
            draw_hud_combined(screen, font_hud, cars, towers,
                              selected_car_idx, sim_speed,
                              paused, sim_clock_s, fontB)

        if paused:
            ps = fontB.render(
                "⏸  PAUSED — SPACE to resume | R=reset view | ESC=quit",
                True, (255, 230, 60))
            screen.blit(ps, (WIDTH // 2 - ps.get_width() // 2, MAP_H // 2 - 14))

        pygame.display.flip()

    pygame.quit()
    print("[DONE] Simulation ended.")


if __name__ == "__main__":
    run_simulation()