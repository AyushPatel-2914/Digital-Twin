import pygame
import numpy as np
import math
import random
import pickle
import os
import json

# --- Module Imports ---
from Map import map_loader as map_data
from config import *
from utils import Path, KalmanFilter, resist_forces, traction_force_from_power, brake_force_from_command
from car import Car
from dispatcher import Dispatcher
from graphics import grid_to_screen, screen_to_grid, draw_road_network, draw_active_path
from tooltip_overlay import get_hovered_entity, draw_tooltip
from Algorithm.planner_registry import load_local_planner, DEFAULT_GLOBAL_PLANNER, DEFAULT_LOCAL_PLANNER

# --- Network Swarm Manager ---
from Map.network_manager import (
    MobileTower, run_swarm_k_median, assign_tower_targets,
    COMM_RADIUS, BATT_CAP_WH, _dist_xy, _tower_graph_colors,
)


# ═════════════════════════════════════════════════════════════════════════════
#  Config loaders
# ═════════════════════════════════════════════════════════════════════════════

def _load_json_file(path, label):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {label}: {e}")
    return None


def load_mine_config():
    config = {"truck_count": 5, "coal_capacities": {}}
    loaded = _load_json_file("Map/mine_config.json", "mine config")
    if loaded:
        config["truck_count"]     = loaded.get("truck_count", 5)
        config["coal_capacities"] = loaded.get("coal_capacities", {})
        print(f"Mine config: {config['truck_count']} trucks.")
    else:
        if os.path.exists("truck.txt"):
            try:
                config["truck_count"] = int(open("truck.txt").read().strip())
            except ValueError:
                pass
        print(f"No mine_config.json — using {config['truck_count']} trucks.")
    return config


def load_algorithm_config():
    config = {"global_planner": DEFAULT_GLOBAL_PLANNER,
              "local_planner":  DEFAULT_LOCAL_PLANNER}
    loaded = _load_json_file("algorithm_config.json", "algorithm config")
    if loaded:
        config["global_planner"] = loaded.get("global_planner", DEFAULT_GLOBAL_PLANNER)
        config["local_planner"]  = loaded.get("local_planner",  DEFAULT_LOCAL_PLANNER)
        print(f"Algorithm config: global='{config['global_planner']}', "
              f"local='{config['local_planner']}'.")
    legacy = _load_json_file("Map/mine_config.json", "mine config")
    if legacy and (legacy.get("global_planner") or legacy.get("local_planner")):
        config["global_planner"] = legacy.get("global_planner", DEFAULT_GLOBAL_PLANNER)
        config["local_planner"]  = legacy.get("local_planner",  DEFAULT_LOCAL_PLANNER)
        print("WARNING: planner keys overridden from Map/mine_config.json.")
    return config


# ═════════════════════════════════════════════════════════════════════════════
#  Path helpers
# ═════════════════════════════════════════════════════════════════════════════

def get_path_from_nodes(route_node_names, waypoints_map):
    final_waypoints = []
    if not route_node_names:
        return []
    for i in range(len(route_node_names) - 1):
        seg_start, seg_end = route_node_names[i], route_node_names[i + 1]
        for chain_tuple, waypoints in waypoints_map.items():
            try:
                idx = chain_tuple.index(seg_start)
                if idx + 1 < len(chain_tuple) and chain_tuple[idx + 1] == seg_end:
                    s = idx * POINTS_PER_SEGMENT
                    e = (idx + 1) * POINTS_PER_SEGMENT
                    final_waypoints.extend(waypoints[s:e])
                    break
                idx = chain_tuple.index(seg_end)
                if idx + 1 < len(chain_tuple) and chain_tuple[idx + 1] == seg_start:
                    s = idx * POINTS_PER_SEGMENT
                    e = (idx + 1) * POINTS_PER_SEGMENT
                    seg = waypoints[s:e + 1]
                    final_waypoints.extend(seg[::-1][:-1])
                    break
            except ValueError:
                continue
    if final_waypoints and route_node_names[-1] in map_data.NODES:
        final_waypoints.append(map_data.NODES[route_node_names[-1]])
    return final_waypoints


# ═════════════════════════════════════════════════════════════════════════════
#  Battery graph panel  (mirrors draw_battery_graph in simulation_3d.py)
# ═════════════════════════════════════════════════════════════════════════════

GRAPH_H      = 180   # pixel height of the battery panel
GRAPH_MARGIN = 50    # left margin for % labels

def _battery_color(pct):
    """green → yellow → red  (mirrors battery_color() in simulation_3d.py)."""
    if pct > 60:
        return (int(255 * (1 - (pct - 60) / 40)), 220, 40)
    elif pct > 20:
        return (255, int(70 + 150 * (pct - 20) / 40), 20)
    return (255, 40, 40)


def draw_battery_panel(screen, network_towers, font_small, font_bold,
                       panel_rect):
    """
    Draw the battery drain graph panel at the bottom of the screen.

    panel_rect : pygame.Rect  — the full panel area (including labels).
    """
    # Panel background
    pygame.draw.rect(screen, (8, 8, 14), panel_rect)
    pygame.draw.rect(screen, (80, 80, 80), panel_rect, 1)

    # Inner graph area (leave room for left labels)
    gr = pygame.Rect(
        panel_rect.left + GRAPH_MARGIN,
        panel_rect.top  + 20,
        panel_rect.width - GRAPH_MARGIN - 10,
        panel_rect.height - 36,
    )

    # Horizontal grid lines at 0 / 25 / 50 / 75 / 100 %
    for pct in (0, 25, 50, 75, 100):
        gy = gr.bottom - (pct / 100) * gr.height
        pygame.draw.line(screen, (50, 50, 65),
                         (gr.left, int(gy)), (gr.right, int(gy)), 1)
        lbl = font_small.render(f"{pct}%", True, (110, 110, 130))
        screen.blit(lbl, (panel_rect.left + 4, int(gy) - 7))

    # One line per tower
    for tw in network_towers:
        if len(tw.history) < 2:
            continue
        n   = len(tw.history)
        pts = [
            (int(gr.left + (i / n) * gr.width),
             int(gr.bottom - (p / 100) * gr.height))
            for i, p in enumerate(tw.history)
        ]
        pygame.draw.lines(screen, tw.graph_color, False, pts, 2)

        # Live label at the right end of the line
        lx, ly = pts[-1]
        label  = font_small.render(
            f"T{tw.id}: {tw.pct:.0f}%  {tw.energy:.0f}Wh",
            True, tw.graph_color
        )
        screen.blit(label, (
            min(lx + 4, gr.right - label.get_width() - 2),
            max(ly - 10, gr.top + 2),
        ))

    # Panel title + std-dev
    pcts    = [tw.pct for tw in network_towers]
    std_now = float(np.std(pcts)) if len(pcts) > 1 else 0.0
    title   = font_bold.render(
        f"Tower Battery (%)  —  K-Median coverage  |  Std-dev: {std_now:.2f}%",
        True, (180, 180, 200),
    )
    screen.blit(title, (
        panel_rect.centerx - title.get_width() // 2,
        panel_rect.top + 2,
    ))

    # Divider line
    pygame.draw.line(screen, (60, 60, 70),
                     (panel_rect.left, panel_rect.top),
                     (panel_rect.right, panel_rect.top), 1)


# ═════════════════════════════════════════════════════════════════════════════
#  run_simulation
# ═════════════════════════════════════════════════════════════════════════════

def run_simulation():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption(
        "Pro Trucker Fleet — MPC + Swarm K-Median + Battery Graph"
    )
    clock      = pygame.time.Clock()
    font       = pygame.font.SysFont("Consolas", 18)
    font_small = pygame.font.SysFont("Consolas", 13)
    font_bold  = pygame.font.SysFont("Consolas", 14, bold=True)

    # --- Load Configurations ---
    mine_config      = load_mine_config()
    algo_config      = load_algorithm_config()
    truck_count         = mine_config["truck_count"]
    coal_capacities     = mine_config["coal_capacities"]
    global_planner_name = algo_config["global_planner"]
    local_planner_name  = algo_config["local_planner"]

    try:
        local_planner = load_local_planner(local_planner_name)
    except Exception as e:
        print(f"Planner '{local_planner_name}' failed ({e}). Using default.")
        local_planner = load_local_planner(DEFAULT_LOCAL_PLANNER)

    # --- Load cached maps ---
    wp_path    = 'Map/waypoints.pkl'
    cache_path = 'Map/map_cache.pkl'
    if not os.path.exists(wp_path) or not os.path.exists(cache_path):
        print("ERROR: map cache files missing in Map/.")
        return

    with open(wp_path,    'rb') as f: waypoints_map = pickle.load(f)
    with open(cache_path, 'rb') as f: cache_data    = pickle.load(f)
    road_graph  = cache_data.get('road_graph',  {})
    route_cache = cache_data.get('route_cache', {})

    # --- Graph repair ---
    all_terminals   = set(map_data.LOAD_ZONES + map_data.DUMP_ZONES)
    incoming_map    = {}
    outgoing_counts = {}
    for sn, edges in road_graph.items():
        outgoing_counts[sn] = len(edges)
        for tgt, _ in edges:
            incoming_map.setdefault(tgt, []).append(sn)

    def _patch(node, visited):
        if node in visited: return
        visited.add(node)
        parents = incoming_map.get(node, [])
        if not parents: return
        parent = parents[0]
        road_graph.setdefault(node, [])
        if not any(t == parent for t, _ in road_graph[node]):
            p1, p2 = map_data.NODES[node], map_data.NODES[parent]
            road_graph[node].append((parent, float(np.linalg.norm(p1 - p2))))
        if not ("hub" in parent or
                (len(incoming_map.get(parent, [])) > 1 and
                 outgoing_counts.get(parent, 0) > 1)):
            _patch(parent, visited)

    for z in all_terminals:
        _patch(z, set())

    # --- Dispatcher + trucks ---
    dispatcher = Dispatcher(road_graph,
                            coal_capacities=coal_capacities,
                            global_planner_name=global_planner_name)
    cars, kfs = [], []
    start_nodes = list(map_data.DUMP_ZONES)
    random.shuffle(start_nodes)
    for i in range(truck_count):
        sn  = start_nodes[i % len(start_nodes)]
        car = Car(i + 1, 0, 0, 0)
        car.current_node_name = sn
        if sn in map_data.NODES:
            car.x_m = map_data.NODES[sn][0]
            car.y_m = map_data.NODES[sn][1]
        cars.append(car)
        kfs.append(KalmanFilter(dt=1/60, start_x=car.x_m, start_y=car.y_m))

    print("Running initial Global Optimization...")
    dispatcher.update_global_plan(cars)

    for idx, car in enumerate(cars):
        tgt   = dispatcher.assign_task(car)
        car.target_node_name = tgt
        route = local_planner.compute_route(road_graph, car.current_node_name,
                                            tgt, cache=route_cache)
        if not route: continue
        wp = get_path_from_nodes(route, waypoints_map)
        if not wp or len(wp) < 2: continue
        car.x_m, car.y_m = wp[0][0], wp[0][1]
        car.angle         = math.atan2(wp[1][1]-wp[0][1], wp[1][0]-wp[0][0])
        car.path          = Path(wp)
        car.op_state      = "GOING_TO_ENDPOINT"
        car.desired_speed_ms = SPEED_MS_EMPTY
        kfs[idx] = KalmanFilter(dt=1/60, start_x=wp[0][0], start_y=wp[0][1])

    # --- Network towers ---
    TOWER_HUB_GOALS = ["main_hub","e_hub","sw_hub","fw_hub","n_hub","s_hub"]
    valid_hubs = [g for g in TOWER_HUB_GOALS if g in map_data.NODES]
    if not valid_hubs:
        valid_hubs = list(map_data.DUMP_ZONES) or list(map_data.NODES.keys())

    NUM_TOWERS  = 4
    tcolors     = _tower_graph_colors(NUM_TOWERS)
    network_towers = []
    for i in range(NUM_TOWERS):
        sn  = valid_hubs[i % len(valid_hubs)]
        pos = map_data.NODES[sn]
        tw  = MobileTower(
            id_code     = str(i),
            x           = float(pos[0]),
            y           = float(pos[1]),
            z           = float(pos[2]) if len(pos) > 2 else 0.0,
            graph_color = tcolors[i],
        )
        tw.current_node = sn
        tw.target_node  = sn
        network_towers.append(tw)

    print(f"[TOWERS] {NUM_TOWERS} towers spawned, speed={network_towers[0].speed} m/s")

    # --- View setup ---
    all_npos  = list(map_data.NODES.values())
    min_x, max_x = min(p[0] for p in all_npos), max(p[0] for p in all_npos)
    min_y, max_y = min(p[1] for p in all_npos), max(p[1] for p in all_npos)

    # Map area shrinks by GRAPH_H to leave room for battery panel
    map_h = HEIGHT - GRAPH_H
    scale = min(
        (WIDTH   - PADDING*2) / ((max_x - min_x) * METERS_TO_PIXELS),
        (map_h   - PADDING*2) / ((max_y - min_y) * METERS_TO_PIXELS),
    )
    pan = [
        PADDING - (min_x * METERS_TO_PIXELS * scale),
        PADDING - (min_y * METERS_TO_PIXELS * scale),
    ]

    mouse_dragging   = False
    last_mouse_pos   = None
    selected_car_idx = 0
    paused    = False
    sim_speed = 1.0

    for car in cars:
        car.run_mpc([])

    def _get_path(route):
        return get_path_from_nodes(route, waypoints_map)

    # --- Timers ---
    mpc_timer              = 0.0;  MPC_INTERVAL            = 0.1
    traffic_timer          = 0.0;  TRAFFIC_INTERVAL        = 1.0
    global_opt_timer       = 0.0;  GLOBAL_OPT_INTERVAL     = 30.0
    kmedian_timer          = 0.0;  KMEDIAN_INTERVAL        = 15.0
    battery_timer          = 0.0;  BATTERY_INTERVAL        = 1.0   # drain once / sim-sec

    # ── Main loop ─────────────────────────────────────────────────────────────
    running = True
    while running:
        frame_dt = clock.tick(60) / 1000.0
        if frame_dt == 0: continue
        sim_dt = 0.0 if paused else frame_dt * sim_speed

        if sim_dt > 0:
            mpc_timer        += sim_dt
            traffic_timer    += sim_dt
            global_opt_timer += sim_dt
            kmedian_timer    += sim_dt
            battery_timer    += sim_dt

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if   event.key == pygame.K_ESCAPE:  running = False
                elif event.key in (pygame.K_SPACE, pygame.K_p):
                    paused = not paused
                elif event.mod & pygame.KMOD_SHIFT:
                    k = event.key
                    if   k == pygame.K_1: sim_speed = 1.0
                    elif k == pygame.K_2: sim_speed = 2.0
                    elif k == pygame.K_3: sim_speed = 3.0
                    elif k == pygame.K_4: sim_speed = 4.0
                    elif k == pygame.K_5: sim_speed = 5.0
                    elif k == pygame.K_0: sim_speed = 0.5
                elif event.key == pygame.K_TAB:
                    selected_car_idx = (selected_car_idx + 1) % len(cars)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:
                    mouse_dragging, last_mouse_pos = True, event.pos
                elif event.button in (4, 5):
                    zf  = ZOOM_FACTOR if event.button == 4 else 1/ZOOM_FACTOR
                    mpm = screen_to_grid(event.pos, scale, pan)
                    scale *= zf
                    nsp   = grid_to_screen(mpm, scale, pan)
                    pan[0] += event.pos[0] - nsp[0]
                    pan[1] += event.pos[1] - nsp[1]
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION and mouse_dragging:
                pan[0] += event.pos[0] - last_mouse_pos[0]
                pan[1] += event.pos[1] - last_mouse_pos[1]
                last_mouse_pos = event.pos

        # ── Simulation updates ────────────────────────────────────────────────
        if sim_dt > 0:

            # Traffic / global optimisation
            if traffic_timer >= TRAFFIC_INTERVAL:
                traffic_timer = 0.0
                dispatcher.update_traffic_weights(cars)
            if global_opt_timer >= GLOBAL_OPT_INTERVAL:
                global_opt_timer = 0.0
                dispatcher.update_global_plan(cars)

            # K-Median coordinator
            if kmedian_timer >= KMEDIAN_INTERVAL:
                kmedian_timer = 0.0
                optimal_hubs  = run_swarm_k_median(cars, num_towers=NUM_TOWERS)
                assign_tower_targets(network_towers, optimal_hubs,
                                     local_planner, dispatcher,
                                     route_cache, _get_path)

            # MPC (10 Hz)
            if mpc_timer >= MPC_INTERVAL:
                mpc_timer = 0.0
                all_traj  = [c.planned_trajectory for c in cars]
                for i, car in enumerate(cars):
                    if car.op_state == "TURNING_AROUND": continue
                    other = [all_traj[j] for j in range(len(cars)) if j != i]
                    car.run_mpc(other, other_cars=cars)

            # Physics (60 Hz)
            for idx, car in enumerate(cars):
                kf      = kfs[idx]
                est_pos = np.array([kf.x[0], kf.x[2]])
                if car.path and car.op_state != "TURNING_AROUND":
                    car.s_path_m = car.path.project(est_pos, car.s_path_m)

                _, base_spd    = car.update_op_state(sim_dt, dispatcher)
                car.desired_speed_ms = base_spd

                if car.needs_new_path:
                    car.needs_new_path   = False
                    car.current_node_name = car.target_node_name
                    new_tgt = dispatcher.assign_task(car)
                    car.target_node_name = new_tgt
                    rnn = local_planner.compute_route(
                        dispatcher.get_graph(),
                        car.current_node_name, new_tgt, cache=route_cache)
                    if rnn:
                        wps = get_path_from_nodes(rnn, waypoints_map)
                        if wps and len(wps) >= 2:
                            car.path = Path(wps)
                            car.s_path_m = 0.0
                            car.turn_target_angle = math.atan2(
                                wps[1][1]-wps[0][1], wps[1][0]-wps[0][0])
                            car.current_mpc_control = np.zeros(2)
                            car.mpc.prev_u = np.zeros_like(car.mpc.prev_u)

                if car.op_state == "TURNING_AROUND":
                    car.execute_turn_step(sim_dt)
                else:
                    car.move(sim_dt)

                av = np.array([car.accel_ms2 * math.cos(car.angle),
                               car.accel_ms2 * math.sin(car.angle)])
                kf.predict(u=av)
                kf.update(z=car.get_noisy_measurement())

            # Tower movement
            for tw in network_towers:
                tw.update_movement(sim_dt)

            # ── Battery drain (once per sim-second) ───────────────────────────
            if battery_timer >= BATTERY_INTERVAL:
                battery_timer = 0.0

                # Build connectivity: which tower each truck is connected to
                # (nearest tower within COMM_RADIUS — mirrors simulation_3d.py)
                truck_assignment = {}   # tower_id -> list of (x,y)
                for car in cars:
                    best_tw, best_d = None, float('inf')
                    for tw in network_towers:
                        d = _dist_xy(tw.pos, (car.x_m, car.y_m))
                        if d < best_d:
                            best_d, best_tw = d, tw
                    if best_tw and best_d <= COMM_RADIUS:
                        truck_assignment.setdefault(best_tw.id, []).append(
                            (car.x_m, car.y_m))

                peer_positions = [(tw.pos[0], tw.pos[1]) for tw in network_towers]

                for tw in network_towers:
                    connected_trucks = truck_assignment.get(tw.id, [])
                    peers = [(x, y) for (x, y), other_tw
                             in zip(peer_positions, network_towers)
                             if other_tw.id != tw.id]
                    tw.update_battery(
                        dt=BATTERY_INTERVAL,
                        connected_truck_positions=connected_trucks,
                        peer_tower_positions=peers,
                    )

        # ── Rendering ─────────────────────────────────────────────────────────
        screen.fill(WHITE)
        g_to_s       = lambda pm: grid_to_screen(pm, scale, pan)
        g_to_s.scale = scale

        # Clip road rendering to map area (above battery panel)
        screen.set_clip(pygame.Rect(0, 0, WIDTH, HEIGHT - GRAPH_H))
        draw_road_network(screen, g_to_s, scale, waypoints_map)

        # Trucks
        for idx, car in enumerate(cars):
            if car.path and idx == selected_car_idx:
                draw_active_path(screen, car.path, g_to_s, scale)
            car.draw(screen, g_to_s, is_selected=(idx == selected_car_idx))

        # Towers
        for tw in network_towers:
            tp = g_to_s((tw.pos[0], tw.pos[1]))

            # Skip if tower is in the battery panel area
            if tp[1] >= HEIGHT - GRAPH_H:
                continue

            # Body colour from battery level
            bcol = (
                (int(255*(1-(tw.pct-60)/40)), 220, 40) if tw.pct > 60 else
                (255, int(70+150*(tw.pct-20)/40), 20)  if tw.pct > 20 else
                (255, 40, 40)
            )

            # Shadow + body
            pygame.draw.circle(screen, (60, 0, 0), (tp[0]+2, tp[1]+2), 9)
            pygame.draw.circle(screen, bcol,          tp, 9)
            pygame.draw.circle(screen, (255,255,255), tp, 9, 2)
            if tw.is_moving:
                pygame.draw.circle(screen, (255,255,255), tp, 3)

            # Coverage ring
            cov_px = int(70.0 * METERS_TO_PIXELS * scale)
            if cov_px > 3:
                pygame.draw.circle(screen, bcol, tp, cov_px, 1)

            # Target marker + dashed line
            if tw.target_node and tw.target_node in map_data.NODES:
                tgt_sp = g_to_s(map_data.NODES[tw.target_node][:2])
                pygame.draw.circle(screen, tw.graph_color, tgt_sp, 5, 2)
                if tw.is_moving:
                    pygame.draw.line(screen, (100,100,30), tp, tgt_sp, 1)

            # Labels
            screen.blit(font_small.render(f"T{tw.id}", True, (255,255,255)),
                        (tp[0]+12, tp[1]-8))
            screen.blit(
                font_small.render(f"{tw.pct:.0f}%  {tw.energy:.0f}Wh",
                                  True, bcol),
                (tp[0]+12, tp[1]+4))
            state = f"{tw.current_node}→{tw.target_node}"
            screen.blit(font_small.render(state, True, (180,180,180)),
                        (tp[0]+12, tp[1]+18))

        screen.set_clip(None)

        # ── Battery graph panel ────────────────────────────────────────────────
        panel_rect = pygame.Rect(0, HEIGHT - GRAPH_H, WIDTH, GRAPH_H)
        draw_battery_panel(screen, network_towers, font_small, font_bold, panel_rect)

        # ── HUD ───────────────────────────────────────────────────────────────
        if cars:
            sel  = cars[selected_car_idx]
            mvng = sum(1 for t in network_towers if t.is_moving)
            conn = sum(1 for t in network_towers if t.pct > 0)
            avg_pct = sum(t.pct for t in network_towers) / max(len(network_towers),1)
            lines = [
                f"Truck {sel.id} (TAB)  |  Towers: {conn}/{NUM_TOWERS}  Moving: {mvng}",
                f"Speed: {sel.speed_ms*3.6:.1f} km/h  |  K-Median every {KMEDIAN_INTERVAL:.0f}s  "
                f"|  Tower speed: {network_towers[0].speed:.0f} m/s",
                f"Avg battery: {avg_pct:.1f}%  |  State: {sel.op_state}",
                f"Sim: {sim_speed}x  |  {'PAUSED' if paused else 'RUNNING'}",
                "TAB=truck  SPACE=pause  SHIFT+0-5=speed  ESC=quit",
            ]
            for i, txt in enumerate(lines):
                screen.blit(font.render(txt, True, (0,0,0)), (10, 10+i*22))

        # Tooltip
        mp = pygame.mouse.get_pos()
        hv = get_hovered_entity(mp, scale, pan, cars, dispatcher)
        if hv:
            draw_tooltip(screen, mp, hv, font)

        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    run_simulation()