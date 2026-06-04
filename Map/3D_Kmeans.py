"""
simulation_3d.py  —  3-D Mining Sim with 2-D K-Median Algorithm
================================================================
The tower-placement algorithm is a direct port of the 2-D local-search
k-median from the reference code, adapted for 3-D nodes (XY used for
distance; Z used only for rendering).

Algorithm (identical logic to the 2-D reference)
-------------------------------------------------
  truck_to_nearest_nodes()
      Maps every truck to the road node closest to it (XY Euclidean).

  local_search_k_median_nodes()
      1. Random initial set of k facility nodes.
      2. Iteratively: for each facility, try swapping it with every
         other node; keep the swap if it lowers total assignment cost.
      3. Repeat until no swap improves cost, or max_iterations hit.
      Returns k node-keys that minimise sum(truck → nearest facility).

  kmedian_navigation()  [SimPy coroutine, one per tower]
      • Tower "0" is the coordinator: every REPOSITION_INTERVAL seconds
        it runs the k-median, then writes target_node into every tower.
      • All towers independently navigate via A* toward their assigned
        target node, re-checking for target changes mid-route.
      • Smooth waypoint-by-waypoint movement (same as trucks).

No CSV output — simulation only.

Controls
--------
  SPACE       — pause / resume
  ESC         — quit
  SCROLL      — zoom in / out
  MIDDLE-DRAG — pan
  R           — reset view (auto-fit)
"""

import sys, os, json, math, random, heapq
import pygame
import simpy
import numpy as np


# ══════════════════════════════════════════════════════════════════════
#  RESOURCE PATH
# ══════════════════════════════════════════════════════════════════════
def resource_path(f):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, f)
    return f


# ══════════════════════════════════════════════════════════════════════
#  CONFIG  (settings_3d.json)
# ══════════════════════════════════════════════════════════════════════
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
    print(f"[ERR]  Bad JSON ({_e}) — using built-in defaults.")


def cfg(key, default):
    return CFG.get(key, default)


# ── simulation parameters ─────────────────────────────────────────────
NUM_TRUCKS          = int(cfg("NUM_TRUCKS",           20))
NUM_TOWERS          = int(cfg("NUM_TOWERS",            8))
SPEED_TRUCK         = float(cfg("SPEED_TRUCK",        1.2))
SPEED_TOWER         = float(cfg("SPEED_TOWER",         0.5))
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

# k-median: how often (sim-seconds) coordinator reruns the algorithm
REPOSITION_INTERVAL = float(cfg("REPOSITION_INTERVAL", 60.0))
# max local-search iterations per reposition
KMEDIAN_MAX_ITER    = int(cfg("KMEDIAN_MAX_ITER",      30))

TOWER_HUB_GOALS     = cfg("TOWER_HUB_GOALS",
                           ["main_hub","e_hub","sw_hub","fw_hub","n_hub","s_hub"])

# ── display parameters ────────────────────────────────────────────────
WIDTH        = int(cfg("WINDOW_WIDTH",   1400))
HEIGHT       = int(cfg("WINDOW_HEIGHT",   900))
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
TRUCK_COLOR      = tuple(int(v) for v in _tkc)
BG_COLOR         = tuple(int(v) for v in _bg)
WHITE = (255, 255, 255)

print(f"\n[SIM]  Towers={NUM_TOWERS}  Trucks={NUM_TRUCKS}  "
      f"CommR={COMM_RADIUS:.1f}m  BattCap={BATT_CAP_WH:.1f}Wh  "
      f"Reposition every {REPOSITION_INTERVAL}s\n")


# ══════════════════════════════════════════════════════════════════════
#  MAP DATA
# ══════════════════════════════════════════════════════════════════════
import MAP_3d_data as MD

NODES      = MD.NODES        # dict: key -> np.array([x, y, z])
ROAD_GRAPH = MD.ROAD_GRAPH   # dict: key -> [(neighbour, weight), ...]
LOAD_ZONES = MD.LOAD_ZONES
DUMP_ZONES = MD.DUMP_ZONES
VIS_CHAINS = MD.VISUAL_CHAINS

# Precompute for fast nearest-node lookups (XY only, matching 2-D logic)
_NODE_KEYS = list(NODES.keys())
_NODE_XY   = np.array([[float(NODES[k][0]), float(NODES[k][1])]
                        for k in _NODE_KEYS])          # (N, 2)


# ══════════════════════════════════════════════════════════════════════
#  AUTO-FIT ISOMETRIC VIEW
# ══════════════════════════════════════════════════════════════════════
def _compute_autofit():
    pad = 60
    pts = []
    for n in NODES.values():
        x, y, z = float(n[0]), float(n[1]), float(n[2])
        sx = (x - y) * math.cos(ISO_ANG) * ISO_SX
        sy = (x + y) * math.sin(ISO_ANG) * ISO_SY - z * Z_SCALE
        pts.append((sx, sy))
    xs = [p[0] for p in pts];  ys = [p[1] for p in pts]
    rw = max(xs) - min(xs);    rh = max(ys) - min(ys)
    if rw == 0 or rh == 0:
        return 1.0, WIDTH // 2, MAP_H // 2
    scale = min((WIDTH - 2*pad) / rw, (MAP_H - 2*pad) / rh)
    ox = WIDTH  // 2 - ((min(xs)+max(xs))/2) * scale
    oy = MAP_H  // 2 - ((min(ys)+max(ys))/2) * scale
    return scale, ox, oy

_AUTO_SCALE, _AUTO_OX, _AUTO_OY = _compute_autofit()
print(f"[VIEW] auto-fit  scale={_AUTO_SCALE:.4f}  "
      f"ox={_AUTO_OX:.1f}  oy={_AUTO_OY:.1f}")

view = {"scale": _AUTO_SCALE, "ox": _AUTO_OX, "oy": _AUTO_OY}

def world_to_screen(pos3, vs=None):
    if vs is None:
        vs = view
    x, y, z = float(pos3[0]), float(pos3[1]), float(pos3[2])
    s  = vs["scale"]
    sx = (x - y) * math.cos(ISO_ANG) * ISO_SX * s
    sy = (x + y) * math.sin(ISO_ANG) * ISO_SY * s - z * Z_SCALE * s
    return int(sx + vs["ox"]), int(sy + vs["oy"])


# ══════════════════════════════════════════════════════════════════════
#  A*  (road-graph pathfinding)
# ══════════════════════════════════════════════════════════════════════
def a_star(graph, start, goal):
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


# ══════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════
def dist_xy(a, b):
    """2-D Euclidean distance using only X and Y (same as the 2-D reference)."""
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

def tx_power(d):
    return 0.0 if d == 0 else P0 * (d / D0) ** PATH_LOSS

def nearest_node_to(pos):
    """Return road-node key nearest to pos (XY only)."""
    diffs = _NODE_XY - np.array([float(pos[0]), float(pos[1])])
    return _NODE_KEYS[int(np.argmin((diffs**2).sum(axis=1)))]


# ══════════════════════════════════════════════════════════════════════
#  K-MEDIAN ALGORITHM  (direct port from 2-D reference)
# ══════════════════════════════════════════════════════════════════════

def truck_to_nearest_nodes(trucks):
    """
    For each truck, find the nearest road node by XY Euclidean distance.
    Returns a list of node keys, one per truck (may contain duplicates).
    Mirrors the 2-D reference exactly.
    """
    nearest = []
    for truck in trucks:
        px, py = float(truck.pos[0]), float(truck.pos[1])
        diffs   = _NODE_XY - np.array([px, py])
        idx     = int(np.argmin((diffs**2).sum(axis=1)))
        nearest.append(_NODE_KEYS[idx])
    return nearest


def compute_kmedian_cost(facility_nodes, demand_nodes):
    """
    Total XY Euclidean distance: each demand node -> its nearest facility.
    Mirrors the 2-D reference compute_kmedian_cost().
    """
    total = 0.0
    for dnode in demand_nodes:
        dcoord = NODES[dnode]
        min_d  = min(dist_xy(dcoord, NODES[fn]) for fn in facility_nodes)
        total += min_d
    return total


def local_search_k_median(truck_nodes, k, max_iterations=None):
    """
    Local-search k-median on road nodes.

    Identical logic to local_search_k_median_nodes() in the 2-D reference:
      1. Random initial set of k facility nodes.
      2. For each facility i, try every other node as a swap candidate;
         accept the first swap that strictly lowers cost.
      3. Repeat until no improvement or max_iterations reached.

    Uses XY Euclidean distance (Z ignored), consistent with 2-D reference.
    truck_nodes : list of node keys (demand points, duplicates allowed)
    k           : number of facilities (= NUM_TOWERS)
    Returns     : list of k node keys
    """
    if max_iterations is None:
        max_iterations = KMEDIAN_MAX_ITER

    all_nodes = _NODE_KEYS

    # Guard: fewer nodes than towers → just return what we have
    if k >= len(all_nodes):
        return list(all_nodes[:k])

    # 1. Random initialisation
    facilities    = random.sample(all_nodes, k)
    current_cost  = compute_kmedian_cost(facilities, truck_nodes)

    # 2. Local search
    for _iteration in range(max_iterations):
        improved = False

        for i in range(k):
            for candidate in all_nodes:
                if candidate in facilities:
                    continue
                new_fac  = facilities.copy()
                new_fac[i] = candidate
                new_cost = compute_kmedian_cost(new_fac, truck_nodes)
                if new_cost < current_cost:
                    facilities    = new_fac
                    current_cost  = new_cost
                    improved      = True
                    break           # restart outer loop (same as 2-D ref)
            if improved:
                break

        if not improved:
            break

    return facilities


# ══════════════════════════════════════════════════════════════════════
#  TOWER NAVIGATION  (SimPy coroutine — mirrors kmedian_navigation 2-D)
# ══════════════════════════════════════════════════════════════════════

def kmedian_navigation(env, tower, towers_dict, trucks):
    """
    SimPy coroutine for one tower.

    • Tower "0" is the coordinator: every REPOSITION_INTERVAL seconds it
      runs local_search_k_median() and writes target_node into every tower.
    • All towers navigate via A* toward their current target_node.
    • If target_node changes mid-route, the tower aborts and re-routes.

    Smooth sub-step movement: position is interpolated every SIM_STEP,
    identical to how trucks move.
    """
    tower_index = int(tower.id)

    # Place tower on a valid hub node at startup
    valid_hubs  = [g for g in TOWER_HUB_GOALS if g in NODES]
    if not valid_hubs:
        valid_hubs = [_NODE_KEYS[0]]
    start_node  = valid_hubs[tower_index % len(valid_hubs)]

    p = NODES[start_node].astype(float)
    tower.pos[:] = [p[0], p[1], p[2]]
    tower.current_node = start_node
    tower.target_node  = start_node

    # Stagger startup so towers don't all recompute simultaneously
    yield env.timeout(tower_index * SIM_STEP)

    while True:
        # ── Coordinator (tower "0") runs the k-median ─────────────────
        if tower.id == "0":
            truck_nodes = truck_to_nearest_nodes(trucks)

            # Guard: pad if fewer trucks than towers
            while len(truck_nodes) < NUM_TOWERS:
                truck_nodes.append(random.choice(_NODE_KEYS))

            print(f"[K-MEDIAN] t={env.now:.0f}s  running local-search "
                  f"(k={NUM_TOWERS}, demand={len(truck_nodes)}) …")
            medians = local_search_k_median(truck_nodes, NUM_TOWERS)
            print(f"[K-MEDIAN] done → {medians}")

            # Assign median[i] to tower with id str(i)
            for i, tid in enumerate(sorted(towers_dict.keys(),
                                            key=lambda x: int(x))):
                towers_dict[tid].target_node = (medians[i]
                                                if i < len(medians)
                                                else tower.current_node)

        # Give coordinator one tick to write targets before followers read
        yield env.timeout(SIM_STEP)

        # ── Navigate toward assigned target ───────────────────────────
        target = tower.target_node

        if target != tower.current_node and target in NODES:
            route = a_star(ROAD_GRAPH, tower.current_node, target)

            if route and len(route) >= 2:
                for seg_i in range(len(route) - 1):
                    # Abort if a new target arrived
                    if tower.target_node != target:
                        break

                    p_from  = NODES[route[seg_i]].astype(float)
                    p_to    = NODES[route[seg_i + 1]].astype(float)
                    seg_len = float(np.linalg.norm(p_to - p_from))
                    if seg_len == 0:
                        continue

                    t_dist = 0.0
                    while t_dist < seg_len:
                        # Check for new target mid-segment
                        if tower.target_node != target:
                            break

                        t_dist = min(t_dist + tower.speed * SIM_STEP, seg_len)
                        alpha  = t_dist / seg_len
                        ip     = p_from + alpha * (p_to - p_from)
                        tower.pos[:] = [float(ip[0]), float(ip[1]),
                                        float(ip[2])]
                        tower.is_moving = True
                        yield env.timeout(SIM_STEP)

                    tower.current_node = route[seg_i + 1]

                # Snap to exact target position if we finished the route
                if tower.target_node == target:
                    p = NODES[target].astype(float)
                    tower.pos[:] = [p[0], p[1], p[2]]
                    tower.current_node = target
                    tower.is_moving    = False

        # ── Sleep until next reposition interval ──────────────────────
        sleep_time = max(SIM_STEP, REPOSITION_INTERVAL - SIM_STEP)
        tower.is_moving = False
        yield env.timeout(sleep_time)


# ══════════════════════════════════════════════════════════════════════
#  TRUCK NAVIGATION  (SimPy coroutine — unchanged)
# ══════════════════════════════════════════════════════════════════════

def navigate_truck(env, truck):
    goals = [g for g in (LOAD_ZONES + DUMP_ZONES) if g in NODES]
    cur   = random.choice(goals)
    p     = NODES[cur].astype(float)
    truck.pos[:] = [p[0], p[1], p[2]]

    while True:
        tgt = random.choice(goals)
        while tgt == cur:
            tgt = random.choice(goals)

        route = a_star(ROAD_GRAPH, cur, tgt)
        if not route or len(route) < 2:
            cur = tgt
            continue

        for i in range(len(route) - 1):
            p_from  = NODES[route[i]].astype(float)
            p_to    = NODES[route[i+1]].astype(float)
            seg_len = float(np.linalg.norm(p_to - p_from))
            if seg_len == 0:
                continue
            t_dist = 0.0
            while t_dist < seg_len:
                t_dist = min(t_dist + truck.speed * SIM_STEP, seg_len)
                alpha  = t_dist / seg_len
                ip     = p_from + alpha * (p_to - p_from)
                truck.pos[:] = [float(ip[0]), float(ip[1]), float(ip[2])]
                yield env.timeout(SIM_STEP)
        cur = tgt


# ══════════════════════════════════════════════════════════════════════
#  ENTITIES
# ══════════════════════════════════════════════════════════════════════

class Truck:
    def __init__(self, env, idx):
        self.id              = f"T{idx}"
        self.pos             = [0.0, 0.0, 0.0]
        self.speed           = SPEED_TRUCK
        self.connected_tower = None
        env.process(navigate_truck(env, self))


class Tower:
    def __init__(self, env, tid, graph_color, towers_dict, trucks):
        self.id           = tid
        self.pos          = [0.0, 0.0, 0.0]
        self.speed        = SPEED_TOWER
        self.energy       = BATT_CAP_WH
        self.pct          = 100.0
        self.history      = []
        self.graph_color  = graph_color
        self.current_node = None       # updated by navigation coroutine
        self.target_node  = None       # written by coordinator (tower "0")
        self.is_moving    = False
        env.process(kmedian_navigation(env, self, towers_dict, trucks))


def tower_graph_colors(n):
    return [(230, int(40 + (i/max(n-1,1))*200), int(40 + (i/max(n-1,1))*100))
            for i in range(n)]


# ══════════════════════════════════════════════════════════════════════
#  ELEVATION + ROAD RENDERING
# ══════════════════════════════════════════════════════════════════════
_zvals = [float(n[2]) for n in NODES.values()]
_z_min, _z_max = min(_zvals), max(_zvals)
_z_rng = max(_z_max - _z_min, 1.0)

def elevation_color(z):
    t = (float(z) - _z_min) / _z_rng
    return (int(60+t*80), int(100+t*130), int(60+t*50))

def build_road_segments(vs):
    segs = []
    for chain in VIS_CHAINS:
        valid = [n for n in chain if n in NODES]
        if len(valid) < 2:
            continue
        pts = [world_to_screen(NODES[n], vs) for n in valid]
        zs  = [float(NODES[n][2]) for n in valid]
        segs.append((pts, elevation_color(sum(zs)/len(zs))))
    return segs


# ══════════════════════════════════════════════════════════════════════
#  BATTERY COLOUR  green → yellow → red
# ══════════════════════════════════════════════════════════════════════
def battery_color(pct):
    if pct > 60:
        return (int(255*(1-(pct-60)/40)), 220, 40)
    elif pct > 20:
        return (255, int(70 + 150*(pct-20)/40), 20)
    return (255, 40, 40)


# ══════════════════════════════════════════════════════════════════════
#  DRAW HELPERS
# ══════════════════════════════════════════════════════════════════════
def draw_battery_graph(screen, towers, font, graph_rect):
    pygame.draw.rect(screen, (25, 25, 35), graph_rect)
    pygame.draw.rect(screen, (80, 80, 80), graph_rect, 1)
    for pct in (0, 25, 50, 75, 100):
        gy = graph_rect.bottom - (pct/100)*graph_rect.height
        pygame.draw.line(screen, (50,50,65),
                         (graph_rect.left, int(gy)),
                         (graph_rect.right, int(gy)), 1)
        screen.blit(font.render(f"{pct}%", True, (110,110,130)),
                    (graph_rect.left-34, int(gy)-7))
    for tw in towers.values():
        if len(tw.history) < 2:
            continue
        n   = len(tw.history)
        pts = [(int(graph_rect.x + (i/n)*graph_rect.width),
                int(graph_rect.bottom - (p/100)*graph_rect.height))
               for i, p in enumerate(tw.history)]
        pygame.draw.lines(screen, tw.graph_color, False, pts, 2)
        lx, ly = pts[-1]
        screen.blit(font.render(f"{tw.id}:{tw.pct:.0f}%", True, tw.graph_color),
                    (min(lx+4, graph_rect.right-70),
                     max(ly-10, graph_rect.top+2)))


def draw_hud(screen, font, towers, trucks, now):
    pcts       = [tw.pct for tw in towers.values()]
    avg_pct    = sum(pcts)/max(len(pcts), 1)
    std_pct    = float(np.std(pcts)) if len(pcts) > 1 else 0.0
    total_conn = sum(1 for t in trucks if t.connected_tower is not None)
    moving     = sum(1 for tw in towers.values() if tw.is_moving)

    lines = [
        f"Sim: {now:.0f}s ({now/60:.1f} min)  |  "
        f"Towers: {NUM_TOWERS}   Trucks: {NUM_TRUCKS}   CommR: {COMM_RADIUS:.0f}m",
        f"Connected trucks: {total_conn}/{NUM_TRUCKS}  |  "
        f"Avg battery: {avg_pct:.1f}%   Std-dev: {std_pct:.2f}%  |  "
        f"Reposition every {REPOSITION_INTERVAL:.0f}s",
        f"Moving towers: {moving}/{NUM_TOWERS}   |   "
        f"Algorithm: Local-Search K-Median (max {KMEDIAN_MAX_ITER} iter)",
        "SPACE=pause   ESC=quit   R=reset view   SCROLL=zoom   MMB-drag=pan",
    ]
    for tw in towers.values():
        conn = sum(1 for t in trucks if t.connected_tower == tw.id)
        mv   = "→ moving" if tw.is_moving else "· idle  "
        tgt  = tw.target_node if tw.target_node else "—"
        lines.append(
            f"  [{mv}] Tower {tw.id}  "
            f"cur:{tw.current_node} → tgt:{tgt}  |  "
            f"{tw.pct:.1f}%  {tw.energy:.0f}Wh  trucks:{conn}")

    for i, line in enumerate(lines):
        surf = font.render(line, True, WHITE)
        bg   = pygame.Surface((surf.get_width()+6, surf.get_height()+2),
                               pygame.SRCALPHA)
        bg.fill((0, 0, 0, 155))
        screen.blit(bg,   (8, 8+i*18-1))
        screen.blit(surf, (11, 8+i*18))


def draw_elevation_bar(screen, font):
    bx, by, bw, bh = WIDTH-40, 40, 12, 120
    for i in range(bh):
        t = 1 - i/bh
        pygame.draw.line(screen,
                         (int(20+t*80), int(60+t*120), int(20+t*40)),
                         (bx, by+i), (bx+bw, by+i))
    pygame.draw.rect(screen, (120,120,120), (bx,by,bw,bh), 1)
    hi = font.render(f"{_z_max:.0f}m", True, (200,230,180))
    lo = font.render(f"{_z_min:.0f}m", True, (140,180,120))
    screen.blit(hi, (bx-hi.get_width()-2, by-2))
    screen.blit(lo, (bx-lo.get_width()-2, by+bh-8))


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(
        f"3D Mining — {NUM_TOWERS} Towers (Local-Search K-Median) | {NUM_TRUCKS} Trucks")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont("Consolas", 13)
    fontB = pygame.font.SysFont("Consolas", 15, bold=True)

    # ── build entities ────────────────────────────────────────────────
    env        = simpy.Environment()
    tcolors    = tower_graph_colors(NUM_TOWERS)
    tower_ids  = [str(i) for i in range(NUM_TOWERS)]

    # trucks first so towers can reference them in kmedian_navigation
    trucks = [Truck(env, i) for i in range(NUM_TRUCKS)]

    # towers_dict passed into each Tower so coordinator can write targets
    towers = {}
    for i, tid in enumerate(tower_ids):
        towers[tid] = Tower(env, tid, tcolors[i], towers, trucks)

    print(f"[SIM] {NUM_TOWERS} towers + {NUM_TRUCKS} trucks ready.\n")

    # ── layout ────────────────────────────────────────────────────────
    graph_rect = pygame.Rect(46, HEIGHT-GRAPH_HEIGHT+18,
                             WIDTH-92, GRAPH_HEIGHT-36)
    road_segs  = build_road_segments(view)

    running     = True
    sim_running = True
    dragging    = False
    drag_start  = (0, 0)
    view_start  = (_AUTO_OX, _AUTO_OY)

    def _comm_px():
        a = world_to_screen([0, 0, 0])
        b = world_to_screen([COMM_RADIUS, 0, 0])
        return max(4, abs(b[0]-a[0]))
    comm_px = _comm_px()

    # ── main loop ─────────────────────────────────────────────────────
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    sim_running = not sim_running
                elif ev.key == pygame.K_r:
                    view["scale"] = _AUTO_SCALE
                    view["ox"]    = _AUTO_OX
                    view["oy"]    = _AUTO_OY
                    road_segs = build_road_segments(view)
                    comm_px   = _comm_px()
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 2:
                dragging   = True
                drag_start = ev.pos
                view_start = (view["ox"], view["oy"])
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 2:
                dragging = False
            elif ev.type == pygame.MOUSEMOTION and dragging:
                view["ox"] = view_start[0] + ev.pos[0] - drag_start[0]
                view["oy"] = view_start[1] + ev.pos[1] - drag_start[1]
                road_segs  = build_road_segments(view)
            elif ev.type == pygame.MOUSEWHEEL:
                f = 1.1 if ev.y > 0 else 0.9
                view["scale"] = max(0.05, min(8.0, view["scale"]*f))
                road_segs = build_road_segments(view)
                comm_px   = _comm_px()

        # ── simulation tick ───────────────────────────────────────────
        if sim_running:
            env.run(until=env.now + SIM_STEP)
            now = env.now

            # Connect each truck to its nearest tower within COMM_RADIUS
            for t in trucks:
                best_tid, best_d = None, float('inf')
                for tid, tw in towers.items():
                    d = dist_xy(t.pos, tw.pos)
                    if d < best_d:
                        best_d, best_tid = d, tid
                t.connected_tower = best_tid if best_d <= COMM_RADIUS else None

            # Battery & power update per tower
            for tid, tw in towers.items():
                if tw.pct <= 0:
                    continue

                mesh_p = sum(
                    tx_power(dist_xy(tw.pos, otw.pos))
                    for oid, otw in towers.items()
                    if oid != tid and dist_xy(tw.pos, otw.pos) <= COMM_RADIUS
                )
                truck_p = sum(
                    tx_power(dist_xy(tw.pos, t.pos))
                    for t in trucks if t.connected_tower == tid
                )
                motion_p = MOVE_POWER if tw.is_moving else 0.0
                total_p  = IDLE_POWER + motion_p + mesh_p + truck_p

                tw.energy -= total_p * (SIM_STEP / 3600.0) * TIME_MULTIPLIER
                tw.pct     = max(0.0, (tw.energy / BATT_CAP_WH) * 100.0)
                tw.history.append(tw.pct)

            if all(tw.pct <= 0 for tw in towers.values()):
                sim_running = False
                print("[SIM] All towers depleted.")

        # ── render ────────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        # roads
        for pts, col in road_segs:
            if len(pts) >= 2:
                dc = tuple(max(0, c-30) for c in col)
                pygame.draw.lines(screen, dc, False, pts, 1)

        w2s = world_to_screen

        # comm links: tower → connected trucks
        if SHOW_LINKS:
            for tid, tw in towers.items():
                if tw.pct <= 0:
                    continue
                tp = w2s(tw.pos)
                for t in trucks:
                    if t.connected_tower == tid:
                        pygame.draw.line(screen, (50,50,90), tp, w2s(t.pos), 1)

        # target lines: tower → its k-median target node
        for tw in towers.values():
            if tw.target_node and tw.target_node in NODES and tw.is_moving:
                pygame.draw.line(screen, (100, 100, 30),
                                 w2s(tw.pos), w2s(NODES[tw.target_node]), 1)

        # towers
        for tid, tw in towers.items():
            if tw.pct <= 0:
                continue
            sp   = w2s(tw.pos)
            bcol = battery_color(tw.pct)

            # draw k-median target marker
            if tw.target_node and tw.target_node in NODES:
                tgt_sp = w2s(NODES[tw.target_node])
                pygame.draw.circle(screen, tw.graph_color, tgt_sp, 5, 2)

            # comm-radius ellipse
            ex = int(comm_px * math.cos(ISO_ANG) * ISO_SX)
            ey = int(comm_px * math.sin(ISO_ANG) * ISO_SY)
            if ex > 3 and ey > 3:
                pygame.draw.ellipse(screen, bcol,
                    pygame.Rect(sp[0]-ex, sp[1]-ey, 2*ex, 2*ey), 1)

            # body
            pygame.draw.circle(screen, (60,0,0),   (sp[0]+2, sp[1]+2), 9)
            pygame.draw.circle(screen, bcol,         sp, 9)
            pygame.draw.circle(screen, WHITE,         sp, 9, 1)
            if tw.is_moving:
                pygame.draw.circle(screen, WHITE, sp, 3)   # moving dot

            screen.blit(font.render(tw.id, True, WHITE), (sp[0]+12, sp[1]-8))
            screen.blit(font.render(f"{tw.pct:.0f}%", True, bcol),
                        (sp[0]+12, sp[1]+4))

        # trucks
        for t in trucks:
            sp = w2s(t.pos)
            pygame.draw.circle(screen, (0,20,60),    (sp[0]+1, sp[1]+1), 5)
            pygame.draw.circle(screen, TRUCK_COLOR,   sp, 5)
            pygame.draw.circle(screen, (120,160,255), sp, 5, 1)

        # zone markers
        for zn in LOAD_ZONES:
            if zn in NODES:
                sp = w2s(NODES[zn])
                if 0 <= sp[0] < WIDTH and 0 <= sp[1] < MAP_H:
                    pygame.draw.circle(screen, (0,200,80), sp, 4)
        for zn in DUMP_ZONES:
            if zn in NODES:
                sp = w2s(NODES[zn])
                if 0 <= sp[0] < WIDTH and 0 <= sp[1] < MAP_H:
                    pygame.draw.circle(screen, (255,180,0), sp, 4)

        draw_elevation_bar(screen, font)

        # battery graph panel
        if SHOW_GRAPH:
            pygame.draw.rect(screen, (8,8,14),
                             (0, HEIGHT-GRAPH_HEIGHT, WIDTH, GRAPH_HEIGHT))
            draw_battery_graph(screen, towers, font, graph_rect)
            pcts    = [tw.pct for tw in towers.values()]
            std_now = float(np.std(pcts)) if pcts else 0.0
            ts = fontB.render(
                f"Tower Battery (%)  —  K-Median (local search)  "
                f"|  Std-dev: {std_now:.2f}%",
                True, (180,180,200))
            screen.blit(ts, (graph_rect.centerx - ts.get_width()//2,
                             HEIGHT-GRAPH_HEIGHT+2))

        if SHOW_HUD:
            draw_hud(screen, font, towers, trucks, env.now)

        # legend
        pygame.draw.circle(screen, (0,200,80),
                           (WIDTH-120, HEIGHT-GRAPH_HEIGHT-22), 5)
        screen.blit(font.render("Load zone", True, (180,230,180)),
                    (WIDTH-110, HEIGHT-GRAPH_HEIGHT-28))
        pygame.draw.circle(screen, (255,180,0),
                           (WIDTH-120, HEIGHT-GRAPH_HEIGHT-8), 5)
        screen.blit(font.render("Dump zone", True, (255,210,130)),
                    (WIDTH-110, HEIGHT-GRAPH_HEIGHT-14))

        pygame.draw.line(screen, (60,60,70),
                         (0, HEIGHT-GRAPH_HEIGHT),
                         (WIDTH, HEIGHT-GRAPH_HEIGHT), 1)

        if not sim_running:
            ps = fontB.render(
                "⏸  PAUSED — SPACE to resume | R = reset view | ESC = quit",
                True, (255,230,60))
            screen.blit(ps, (WIDTH//2 - ps.get_width()//2, MAP_H//2-14))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print("[DONE] Simulation ended.")


if __name__ == "__main__":
    run_simulation()