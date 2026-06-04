"""
simulation_3d.py
================
Full mining-simulation on the 3-D map, rendered in isometric projection.

Controls
--------
  SPACE       — pause / resume
  ESC         — quit
  SCROLL      — zoom in / out
  MIDDLE-DRAG — pan the view
  R           — reset view

CSV output
----------
  One file per tower:  tower_<ID>_log.csv
  Every simulation step appends one fully-detailed row.
"""

import sys, os, json, math, random, heapq, csv
import pygame
import simpy
import numpy as np


# ─────────────────────────────────────────────────────────────────────
#  RESOURCE-PATH HELPER
# ─────────────────────────────────────────────────────────────────────
def resource_path(f):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, f)
    return f


# ═════════════════════════════════════════════════════════════════════
#  LOAD settings_3d.json
# ═════════════════════════════════════════════════════════════════════
_cfg_path = resource_path("settings_3d.json")
try:
    with open(_cfg_path, "r", encoding="utf-8") as _fh:
        CFG = json.load(_fh)
    print(f"[OK]  Loaded {_cfg_path}")
except FileNotFoundError:
    print(f"[WARN] {_cfg_path} not found — using all built-in defaults.")
    CFG = {}
except json.JSONDecodeError as _e:
    print(f"[ERR]  {_cfg_path} has invalid JSON ({_e}) — using all built-in defaults.")
    CFG = {}


def cfg(key, default):
    """Return CFG[key] if present, else default (with a note on first use)."""
    if key not in CFG:
        print(f"  [CFG] '{key}' missing in JSON → default={default!r}")
    return CFG.get(key, default)


# ─── simulation parameters ────────────────────────────────────────────
NUM_TRUCKS      = int(cfg("NUM_TRUCKS",       200))
NUM_TOWERS      = int(cfg("NUM_TOWERS",       50))
SPEED_TRUCK     = float(cfg("SPEED_TRUCK",    12.0))
SPEED_TOWER     = float(cfg("SPEED_TOWER",     5.0))
MOVE_POWER      = float(cfg("MOVE_POWER",      4.0))
IDLE_POWER      = float(cfg("IDLE_POWER",      3.0))
SIM_STEP        = float(cfg("SIM_STEP",        1.0))
TIME_MULTIPLIER = float(cfg("TIME_MULTIPLIER", 1.0))

BATT_VMAX   = float(cfg("BATTERY_VOLTAGE_MAX",  26.5))
BATT_VMIN   = float(cfg("BATTERY_VOLTAGE_MIN",  22.0))
BATT_AH     = float(cfg("BATTERY_AH",          150.0))
BATT_SCALE  = float(cfg("BATTERY_SCALE_FACTOR",  10.0))
BATT_CAP_WH = (BATT_VMAX * BATT_AH) / BATT_SCALE

P0        = float(cfg("P0",        5.0))
PMAX      = float(cfg("PMAX",    250.0))
D0        = float(cfg("D0",       80.0))
PATH_LOSS = float(cfg("PATH_LOSS", 2.5))

_comm_ov    = cfg("COMM_RADIUS_OVERRIDE", None)
COMM_RADIUS = float(_comm_ov) if _comm_ov is not None \
              else D0 * (PMAX / P0) ** (1.0 / PATH_LOSS)

TOWER_HUB_GOALS = cfg("TOWER_HUB_GOALS",
                       ["main_hub","e_hub","sw_hub","fw_hub","n_hub","s_hub"])

CSV_FLUSH_N = int(cfg("CSV_FLUSH_EVERY_N_STEPS", 100))

# ─── display parameters ───────────────────────────────────────────────
WIDTH        = int(cfg("WINDOW_WIDTH",   1400))
HEIGHT       = int(cfg("WINDOW_HEIGHT",   950))
GRAPH_HEIGHT = int(cfg("GRAPH_HEIGHT",    200))
MAP_H        = HEIGHT - GRAPH_HEIGHT

ISO_ANG    = math.radians(float(cfg("ISO_ANGLE_DEG", 30)))
ISO_SX     = float(cfg("ISO_SCALE_X",   1.00))
ISO_SY     = float(cfg("ISO_SCALE_Y",   0.55))
Z_SCALE    = float(cfg("Z_SCALE",       0.18))
BASE_SCALE = float(cfg("MAP_SCALE",     0.75))
BASE_OX    = float(cfg("MAP_OFFSET_X",  700.0))
BASE_OY    = float(cfg("MAP_OFFSET_Y",  420.0))

SHOW_LINKS = bool(cfg("SHOW_COMM_LINKS",    True))
SHOW_GRAPH = bool(cfg("SHOW_BATTERY_GRAPH", True))
SHOW_HUD   = bool(cfg("SHOW_HUD",           True))

_tc  = cfg("TOWER_COLOR",      [220, 30,  30])
_tkc = cfg("TRUCK_COLOR",      [  0, 60, 180])
_bg  = cfg("BACKGROUND_COLOR", [ 15, 18,  25])

TOWER_BASE_COLOR = tuple(int(v) for v in _tc)
TRUCK_COLOR      = tuple(int(v) for v in _tkc)
BG_COLOR         = tuple(int(v) for v in _bg)
WHITE  = (255, 255, 255)

print(f"\n[SIM] Towers={NUM_TOWERS}  Trucks={NUM_TRUCKS}  "
      f"CommR={COMM_RADIUS:.1f}m  BattCap={BATT_CAP_WH:.1f}Wh\n")


# ═════════════════════════════════════════════════════════════════════
#  MAP DATA  (import the companion module)
# ═════════════════════════════════════════════════════════════════════
import MAP_3d_data as MD

NODES      = MD.NODES
EDGES      = MD.EDGES
ROAD_GRAPH = MD.ROAD_GRAPH
LOAD_ZONES = MD.LOAD_ZONES
DUMP_ZONES = MD.DUMP_ZONES
FUEL_ZONES = MD.FUEL_ZONES
VIS_CHAINS = MD.VISUAL_CHAINS


# ═════════════════════════════════════════════════════════════════════
#  ISOMETRIC PROJECTION
# ═════════════════════════════════════════════════════════════════════
view = {"scale": BASE_SCALE, "ox": BASE_OX, "oy": BASE_OY}

def world_to_screen(pos3, vs=None):
    if vs is None:
        vs = view
    x, y, z = float(pos3[0]), float(pos3[1]), float(pos3[2])
    s  = vs["scale"]
    sx = (x - y) * math.cos(ISO_ANG) * ISO_SX * s
    sy = (x + y) * math.sin(ISO_ANG) * ISO_SY * s - z * Z_SCALE * s
    return int(sx + vs["ox"]), int(sy + vs["oy"])


# ═════════════════════════════════════════════════════════════════════
#  A* PATH-FINDING
# ═════════════════════════════════════════════════════════════════════
def a_star(graph, start, goal):
    open_set  = [(0, start)]
    came_from = {}
    g_cost    = {n: float('inf') for n in graph}
    g_cost[start] = 0
    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal:
            path = []
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            return list(reversed(path))
        for nb, w in graph.get(cur, []):
            tg = g_cost[cur] + w
            if tg < g_cost.get(nb, float('inf')):
                came_from[nb] = cur
                g_cost[nb]    = tg
                heapq.heappush(open_set, (tg, nb))
    return []


# ═════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════
def dist2(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

def dist3(a, b):
    return float(np.linalg.norm(
        np.array(a, dtype=float) - np.array(b, dtype=float)))

def tx_power(d):
    return 0.0 if d == 0 else P0 * (d / D0) ** PATH_LOSS


# ═════════════════════════════════════════════════════════════════════
#  NAVIGATION (SimPy process)
# ═════════════════════════════════════════════════════════════════════
def navigate(env, entity):
    cur = random.choice(entity.goals)
    p   = NODES[cur]
    entity.pos[:] = [p[0], p[1], p[2]]
    while True:
        tgt = random.choice(entity.goals)
        while tgt == cur:
            tgt = random.choice(entity.goals)
        route = a_star(ROAD_GRAPH, cur, tgt)
        if not route or len(route) < 2:
            cur = tgt
            continue
        for i in range(len(route) - 1):
            p_from  = NODES[route[i]].astype(float)
            p_to    = NODES[route[i + 1]].astype(float)
            seg_len = float(np.linalg.norm(p_to - p_from))
            if seg_len == 0:
                continue
            step_d = entity.speed * SIM_STEP
            t = 0.0
            while t < seg_len:
                t     = min(t + step_d, seg_len)
                alpha = t / seg_len
                ip    = p_from + alpha * (p_to - p_from)
                entity.pos[:] = [float(ip[0]), float(ip[1]), float(ip[2])]
                yield env.timeout(SIM_STEP)
        cur = tgt


# ═════════════════════════════════════════════════════════════════════
#  ENTITIES
# ═════════════════════════════════════════════════════════════════════
class Truck:
    def __init__(self, env, idx):
        self.id              = f"T{idx}"
        self.pos             = [0.0, 0.0, 0.0]
        self.speed           = SPEED_TRUCK
        self.goals           = LOAD_ZONES + DUMP_ZONES
        self.connected_tower = None
        env.process(navigate(env, self))


class Tower:
    def __init__(self, env, tid, graph_color):
        self.id          = tid
        self.pos         = [0.0, 0.0, 0.0]
        self.speed       = SPEED_TOWER
        self.goals       = [g for g in TOWER_HUB_GOALS if g in NODES]
        self.energy      = BATT_CAP_WH
        self.pct         = 100.0
        self.history     = []
        self.graph_color = graph_color
        env.process(navigate(env, self))


# ─── graph-line colours (all red family) ──────────────────────────────
def tower_graph_colors(n):
    cols = []
    for i in range(n):
        g = int(40 + (i / max(n - 1, 1)) * 200)
        b = int(40 + (i / max(n - 1, 1)) * 100)
        cols.append((230, g, b))
    return cols


# ═════════════════════════════════════════════════════════════════════
#  CSV HEADER  (same for every tower file)
# ═════════════════════════════════════════════════════════════════════
def build_csv_header(num_trucks):
    hdr = [
        # ── timestamp ──────────────────────────────────────
        "timestamp_s",
        "timestamp_min",
        # ── tower identity & position ──────────────────────
        "tower_id",
        "tower_x_m",
        "tower_y_m",
        "tower_z_m",
        # ── battery / electrical ───────────────────────────
        "battery_pct",
        "battery_voltage_V",
        "battery_energy_remaining_Wh",
        "battery_capacity_Wh",
        "current_draw_A",
        "ah_consumed_this_step",
        "ah_consumed_cumulative",
        # ── power breakdown (W) ────────────────────────────
        "power_move_W",
        "power_idle_W",
        "power_mesh_comm_W",
        "power_truck_comm_W",
        "power_total_W",
        # ── connected towers summary ───────────────────────
        "num_connected_towers",
        "connected_tower_ids",           # "1;2"
        "connected_tower_x_coords_m",    # "x1;x2"
        "connected_tower_y_coords_m",
        "connected_tower_z_coords_m",
        "connected_tower_distances_2d_m",
        "connected_tower_distances_3d_m",
        "mesh_tx_power_per_tower_W",     # per-link TX power
        # ── connected trucks summary ───────────────────────
        "num_connected_trucks",
        "connected_truck_ids",           # "T0;T5"
        "connected_truck_x_coords_m",
        "connected_truck_y_coords_m",
        "connected_truck_z_coords_m",
        "connected_truck_distances_2d_m",
        "connected_truck_distances_3d_m",
        "truck_tx_power_per_truck_W",
    ]
    # ── per-truck snapshot ─────────────────────────────────
    for i in range(num_trucks):
        hdr += [
            f"truck{i}_id",
            f"truck{i}_x_m",
            f"truck{i}_y_m",
            f"truck{i}_z_m",
            f"truck{i}_dist2d_to_tower_m",
            f"truck{i}_dist3d_to_tower_m",
            f"truck{i}_connected_tower",
        ]
    return hdr


# ═════════════════════════════════════════════════════════════════════
#  ELEVATION COLOUR  (for road rendering)
# ═════════════════════════════════════════════════════════════════════
_zvals = [float(n[2]) for n in NODES.values()]
_z_min = min(_zvals);  _z_max = max(_zvals)
_z_rng = max(_z_max - _z_min, 1.0)

def elevation_color(z):
    t = (float(z) - _z_min) / _z_rng
    return (int(60 + t*80), int(100 + t*130), int(60 + t*50))

def build_road_segments(vs):
    segs = []
    for chain in VIS_CHAINS:
        valid = [n for n in chain if n in NODES]
        if len(valid) < 2:
            continue
        pts = [world_to_screen(NODES[n], vs) for n in valid]
        zs  = [float(NODES[n][2]) for n in valid]
        col = elevation_color(sum(zs) / len(zs))
        segs.append((pts, col))
    return segs


# ═════════════════════════════════════════════════════════════════════
#  DRAW HELPERS
# ═════════════════════════════════════════════════════════════════════
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


def draw_hud(screen, font, towers, trucks, now):
    lines = [
        f"Sim: {now:.0f}s  ({now/60:.1f} min) | "
        f"Towers:{NUM_TOWERS}  Trucks:{NUM_TRUCKS}  CommR:{COMM_RADIUS:.0f}m",
        "SPACE=pause  ESC=quit  R=reset  SCROLL=zoom  MMB-drag=pan",
    ]
    for tw in towers.values():
        cn = sum(1 for t in trucks if t.connected_tower == tw.id)
        lines.append(
            f"  Tower {tw.id}: {tw.pct:.1f}%  "
            f"| {tw.energy:.1f} Wh  | connected trucks: {cn}")
    for i, line in enumerate(lines):
        surf = font.render(line, True, WHITE)
        bg   = pygame.Surface((surf.get_width() + 6, surf.get_height() + 2),
                              pygame.SRCALPHA)
        bg.fill((0, 0, 0, 155))
        screen.blit(bg,   (8, 8 + i * 18 - 1))
        screen.blit(surf, (11, 8 + i * 18))


def draw_elevation_bar(screen, font):
    bx, by, bw, bh = WIDTH - 40, 40, 12, 120
    for i in range(bh):
        t = 1 - i / bh
        pygame.draw.line(screen,
                         (int(20 + t*80), int(60 + t*120), int(20 + t*40)),
                         (bx, by + i), (bx + bw, by + i))
    pygame.draw.rect(screen, (120, 120, 120), (bx, by, bw, bh), 1)
    hi = font.render(f"{_z_max:.0f}m", True, (200, 230, 180))
    lo = font.render(f"{_z_min:.0f}m", True, (140, 180, 120))
    screen.blit(hi, (bx - hi.get_width() - 2, by - 2))
    screen.blit(lo, (bx - lo.get_width() - 2, by + bh - 8))


# ═════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════
def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(
        f"3D Mining Sim — {NUM_TOWERS} Towers | {NUM_TRUCKS} Trucks")
    clock = pygame.time.Clock()
    font  = pygame.font.SysFont("Consolas", 13)
    fontB = pygame.font.SysFont("Consolas", 15, bold=True)

    # ── entities ──────────────────────────────────────────────────────
    env       = simpy.Environment()
    tcolors   = tower_graph_colors(NUM_TOWERS)
    tower_ids = [str(i) for i in range(NUM_TOWERS)]
    towers    = {tid: Tower(env, tid, tcolors[i])
                 for i, tid in enumerate(tower_ids)}
    trucks    = [Truck(env, i) for i in range(NUM_TRUCKS)]

    # ── open one CSV per tower ────────────────────────────────────────
    hdr = build_csv_header(NUM_TRUCKS)
    csv_files, csv_writers = {}, {}
    for tid in tower_ids:
        fname = f"tower_{tid}_log.csv"
        fh    = open(fname, "w", newline="", encoding="utf-8")
        wr    = csv.writer(fh)
        wr.writerow(hdr)
        csv_files[tid]   = fh
        csv_writers[tid] = wr
        print(f"[CSV] {fname}  ({len(hdr)} columns)")

    # ── graph rect ────────────────────────────────────────────────────
    graph_rect = pygame.Rect(46, HEIGHT - GRAPH_HEIGHT + 18,
                             WIDTH - 92, GRAPH_HEIGHT - 36)

    # ── interaction state ─────────────────────────────────────────────
    running     = True
    sim_running = True
    dragging    = False
    drag_start  = (0, 0)
    view_start  = (BASE_OX, BASE_OY)

    road_segs = build_road_segments(view)

    def _comm_px():
        a = world_to_screen([0, 0, 0])
        b = world_to_screen([COMM_RADIUS, 0, 0])
        return abs(b[0] - a[0])

    comm_px = _comm_px()

    # cumulative Ah per tower
    ah_cum    = {tid: 0.0 for tid in tower_ids}
    flush_ctr = 0

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
                    view["scale"] = BASE_SCALE
                    view["ox"]    = BASE_OX
                    view["oy"]    = BASE_OY
                    road_segs = build_road_segments(view)
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 2:
                dragging = True
                drag_start = ev.pos
                view_start = (view["ox"], view["oy"])
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 2:
                dragging = False
            elif ev.type == pygame.MOUSEMOTION and dragging:
                view["ox"] = view_start[0] + ev.pos[0] - drag_start[0]
                view["oy"] = view_start[1] + ev.pos[1] - drag_start[1]
                road_segs  = build_road_segments(view)
            elif ev.type == pygame.MOUSEWHEEL:
                factor = 1.1 if ev.y > 0 else 0.9
                view["scale"] = max(0.15, min(6.0, view["scale"] * factor))
                road_segs = build_road_segments(view)
                comm_px   = _comm_px()

        # ── simulation tick ───────────────────────────────────────────
        if sim_running:
            env.run(until=env.now + SIM_STEP)
            now = env.now

            # assign trucks to nearest tower within comm radius
            for t in trucks:
                best_tid, best_d = None, float('inf')
                for tid, tw in towers.items():
                    d = dist2(t.pos, tw.pos)
                    if d < best_d:
                        best_d, best_tid = d, tid
                t.connected_tower = best_tid if best_d <= COMM_RADIUS else None

            # per-tower update + CSV write
            for tid, tw in towers.items():
                if tw.pct <= 0:
                    continue

                # ── who is connected? ─────────────────────────────────
                conn_towers = []          # (other_tid, other_tower, d2, d3)
                for oid, otw in towers.items():
                    if oid == tid:
                        continue
                    d2 = dist2(tw.pos, otw.pos)
                    if d2 <= COMM_RADIUS:
                        d3 = dist3(tw.pos, otw.pos)
                        conn_towers.append((oid, otw, d2, d3))

                conn_trucks = []          # (truck, d2, d3)
                for t in trucks:
                    if t.connected_tower == tid:
                        d2 = dist2(tw.pos, t.pos)
                        d3 = dist3(tw.pos, t.pos)
                        conn_trucks.append((t, d2, d3))

                # ── power ─────────────────────────────────────────────
                mesh_p  = sum(tx_power(d2) for _, _, d2, _ in conn_towers)
                truck_p = sum(tx_power(d2) for _, d2, _  in conn_trucks)
                total_p = MOVE_POWER + IDLE_POWER + mesh_p + truck_p

                # ── battery ───────────────────────────────────────────
                v     = BATT_VMIN + (tw.pct / 100.0) * (BATT_VMAX - BATT_VMIN)
                i_a   = total_p / v if v > 0 else 0.0
                ah    = i_a * (SIM_STEP / 3600.0) * TIME_MULTIPLIER
                tw.energy -= total_p * (SIM_STEP / 3600.0) * TIME_MULTIPLIER
                tw.pct     = max(0.0, (tw.energy / BATT_CAP_WH) * 100.0)
                tw.history.append(tw.pct)
                ah_cum[tid] += ah

                # ── build CSV row ─────────────────────────────────────
                # connected-tower columns (semicolon-delimited lists)
                def _join(lst): return ";".join(str(v) for v in lst)

                ct_ids  = _join(o                           for o, _, _, _  in conn_towers)
                ct_xs   = _join(round(otw.pos[0], 3)       for _, otw, _, _ in conn_towers)
                ct_ys   = _join(round(otw.pos[1], 3)       for _, otw, _, _ in conn_towers)
                ct_zs   = _join(round(otw.pos[2], 3)       for _, otw, _, _ in conn_towers)
                ct_d2   = _join(round(d2, 3)               for _, _, d2, _  in conn_towers)
                ct_d3   = _join(round(d3, 3)               for _, _, _, d3  in conn_towers)
                ct_txpw = _join(round(tx_power(d2), 5)     for _, _, d2, _  in conn_towers)

                ck_ids  = _join(t.id                        for t, _, _  in conn_trucks)
                ck_xs   = _join(round(t.pos[0], 3)         for t, _, _  in conn_trucks)
                ck_ys   = _join(round(t.pos[1], 3)         for t, _, _  in conn_trucks)
                ck_zs   = _join(round(t.pos[2], 3)         for t, _, _  in conn_trucks)
                ck_d2   = _join(round(d2, 3)               for _, d2, _ in conn_trucks)
                ck_d3   = _join(round(d3, 3)               for _, _, d3 in conn_trucks)
                ck_txpw = _join(round(tx_power(d2), 5)     for _, d2, _ in conn_trucks)

                row = [
                    round(now, 3),
                    round(now / 60.0, 5),
                    tid,
                    round(tw.pos[0], 3),
                    round(tw.pos[1], 3),
                    round(tw.pos[2], 3),
                    round(tw.pct, 4),
                    round(v, 4),
                    round(tw.energy, 4),
                    round(BATT_CAP_WH, 4),
                    round(i_a, 6),
                    round(ah, 8),
                    round(ah_cum[tid], 6),
                    round(MOVE_POWER, 4),
                    round(IDLE_POWER, 4),
                    round(mesh_p, 5),
                    round(truck_p, 5),
                    round(total_p, 5),
                    len(conn_towers),
                    ct_ids, ct_xs, ct_ys, ct_zs, ct_d2, ct_d3, ct_txpw,
                    len(conn_trucks),
                    ck_ids, ck_xs, ck_ys, ck_zs, ck_d2, ck_d3, ck_txpw,
                ]

                # per-truck snapshot
                for t in trucks:
                    d2 = dist2(tw.pos, t.pos)
                    d3 = dist3(tw.pos, t.pos)
                    row += [
                        t.id,
                        round(t.pos[0], 3),
                        round(t.pos[1], 3),
                        round(t.pos[2], 3),
                        round(d2, 3),
                        round(d3, 3),
                        t.connected_tower if t.connected_tower else "None",
                    ]

                csv_writers[tid].writerow(row)

            flush_ctr += 1
            if flush_ctr >= CSV_FLUSH_N:
                for fh in csv_files.values():
                    fh.flush()
                flush_ctr = 0

            if all(tw.pct <= 0 for tw in towers.values()):
                sim_running = False
                print("[SIM] All towers depleted.")

        # ── RENDER ────────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        for pts, col in road_segs:
            if len(pts) >= 2:
                dc = tuple(max(0, c - 30) for c in col)
                pygame.draw.lines(screen, dc, False, pts, 1)

        w2s = world_to_screen

        if SHOW_LINKS:
            for tid, tw in towers.items():
                if tw.pct <= 0:
                    continue
                tp = w2s(tw.pos)
                for t in trucks:
                    if t.connected_tower == tid:
                        pygame.draw.line(screen, (50, 50, 80),
                                         tp, w2s(t.pos), 1)

        for tid, tw in towers.items():
            if tw.pct <= 0:
                continue
            sp = w2s(tw.pos)
            ex = int(comm_px * math.cos(ISO_ANG) * ISO_SX)
            ey = int(comm_px * math.sin(ISO_ANG) * ISO_SY)
            if ex > 4 and ey > 4:
                pygame.draw.ellipse(screen, TOWER_BASE_COLOR,
                                    pygame.Rect(sp[0]-ex, sp[1]-ey,
                                                2*ex, 2*ey), 1)
            pygame.draw.circle(screen, (80, 0, 0),        (sp[0]+2, sp[1]+2), 9)
            pygame.draw.circle(screen, TOWER_BASE_COLOR,   sp, 9)
            pygame.draw.circle(screen, WHITE,               sp, 9, 1)
            screen.blit(font.render(tw.id, True, WHITE),
                        (sp[0]+12, sp[1]-8))
            screen.blit(font.render(f"{tw.pct:.0f}%", True, tw.graph_color),
                        (sp[0]+12, sp[1]+4))

        for t in trucks:
            sp = w2s(t.pos)
            pygame.draw.circle(screen, (0, 20, 60),      (sp[0]+1, sp[1]+1), 5)
            pygame.draw.circle(screen, TRUCK_COLOR,        sp, 5)
            pygame.draw.circle(screen, (120, 160, 255),    sp, 5, 1)

        for zn in LOAD_ZONES:
            if zn in NODES:
                sp = w2s(NODES[zn])
                if 0 <= sp[0] < WIDTH and 0 <= sp[1] < MAP_H:
                    pygame.draw.circle(screen, (0, 200, 80), sp, 3)
        for zn in DUMP_ZONES:
            if zn in NODES:
                sp = w2s(NODES[zn])
                if 0 <= sp[0] < WIDTH and 0 <= sp[1] < MAP_H:
                    pygame.draw.circle(screen, (255, 180, 0), sp, 3)

        draw_elevation_bar(screen, font)

        if SHOW_GRAPH:
            pygame.draw.rect(screen, (8, 8, 14),
                             (0, HEIGHT - GRAPH_HEIGHT, WIDTH, GRAPH_HEIGHT))
            draw_battery_graph(screen, towers, font, graph_rect)
            ts = fontB.render("Tower Battery (%)", True, (180, 180, 200))
            screen.blit(ts, (graph_rect.centerx - ts.get_width()//2,
                             HEIGHT - GRAPH_HEIGHT + 2))

        if SHOW_HUD:
            draw_hud(screen, font, towers, trucks, env.now)

        pygame.draw.circle(screen, (0, 200, 80),
                           (WIDTH-120, HEIGHT-GRAPH_HEIGHT-22), 5)
        screen.blit(font.render("Load zone", True, (180, 230, 180)),
                    (WIDTH-110, HEIGHT-GRAPH_HEIGHT-28))
        pygame.draw.circle(screen, (255, 180, 0),
                           (WIDTH-120, HEIGHT-GRAPH_HEIGHT-8), 5)
        screen.blit(font.render("Dump zone", True, (255, 210, 130)),
                    (WIDTH-110, HEIGHT-GRAPH_HEIGHT-14))

        pygame.draw.line(screen, (60, 60, 70),
                         (0, HEIGHT-GRAPH_HEIGHT), (WIDTH, HEIGHT-GRAPH_HEIGHT), 1)

        if not sim_running:
            ps = fontB.render(
                "⏸  PAUSED — SPACE to resume | R to reset view | ESC to quit",
                True, (255, 230, 60))
            screen.blit(ps, (WIDTH//2 - ps.get_width()//2, MAP_H//2 - 14))

        pygame.display.flip()
        clock.tick(60)

    # ── cleanup ───────────────────────────────────────────────────────
    for fh in csv_files.values():
        fh.flush()
        fh.close()
    pygame.quit()
    print("[DONE] All CSV files flushed and closed.")


if __name__ == "__main__":
    run_simulation()