"""
network_manager.py — Mobile Network Coverage Optimization Module
================================================================
Changes in this version
-----------------------
1. Tower speed raised to 8.0 m/s (was 2.0) so towers reach centroid targets
   before trucks disperse to a new configuration.
2. Battery model added (mirrors simulation_3d.py exactly):
     BATT_CAP_WH  — total capacity in Wh
     energy / pct — live state per tower
     history      — list of pct samples (one per update_battery() call)
     update_battery(dt, connected_truck_count, mesh_towers)
         Call this every sim-second (or scaled dt) from the main loop.
3. graph_color — per-tower colour tuple for the battery graph panel.
4. _dist_xy / _nearest_node / _compute_kmedian_cost unchanged.
5. run_swarm_k_median / assign_tower_targets unchanged.
"""

import random
import math
import numpy as np
from Map import map_loader as map_data


# ── Battery constants (mirrors simulation_3d.py defaults) ────────────────────
BATT_VMAX    = 26.5
BATT_VMIN    = 22.0
BATT_AH      = 150.0
BATT_SCALE   = 10.0
BATT_CAP_WH  = (BATT_VMAX * BATT_AH) / BATT_SCALE   # ≈ 397.5 Wh

P0         = 5.0
PMAX       = 250.0
D0         = 80.0
PATH_LOSS  = 2.5
COMM_RADIUS = D0 * (PMAX / P0) ** (1.0 / PATH_LOSS)   # ≈ 537 m

MOVE_POWER  = 4.0   # W extra while moving
IDLE_POWER  = 3.0   # W baseline


def _tx_power(d):
    """Transmit power (W) needed to reach distance d."""
    return 0.0 if d == 0 else P0 * (d / D0) ** PATH_LOSS


# ── Spatial helpers ───────────────────────────────────────────────────────────

def _dist_xy(a, b):
    """2-D Euclidean distance, Z ignored."""
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _nearest_node(pos_xy):
    """Road-node key nearest to (x, y)."""
    px, py = float(pos_xy[0]), float(pos_xy[1])
    best, best_d = None, float('inf')
    for name, npos in map_data.NODES.items():
        d = math.hypot(float(npos[0]) - px, float(npos[1]) - py)
        if d < best_d:
            best_d, best = d, name
    return best


# ── Tower graph colours (green → yellow → red gradient over N towers) ─────────

def _tower_graph_colors(n):
    """Return n distinct RGB tuples for the battery-graph panel."""
    colors = []
    for i in range(n):
        t = i / max(n - 1, 1)
        colors.append((
            int(230),
            int(40  + t * 200),
            int(40  + t * 100),
        ))
    return colors


# ═════════════════════════════════════════════════════════════════════════════
#  MobileTower
# ═════════════════════════════════════════════════════════════════════════════

class MobileTower:
    def __init__(self, id_code, x, y, z=0.0, graph_color=(230, 140, 40)):
        self.id          = str(id_code)
        self.pos         = np.array([float(x), float(y), float(z)])

        # ── Speed: raised to 8 m/s so towers reach targets faster ──────────
        self.speed       = 8.0          # m/s  (was 2.0)

        # ── Node-graph state ────────────────────────────────────────────────
        self.current_node = ""
        self.target_node  = ""
        self.is_moving    = False

        # ── Waypoint path ───────────────────────────────────────────────────
        self.path_waypoints  = []
        self.current_wp_idx  = 0

        # ── Battery model (mirrors Tower in simulation_3d.py) ───────────────
        self.energy      = BATT_CAP_WH   # Wh remaining
        self.pct         = 100.0         # percentage 0-100
        self.history     = []            # pct samples for graph
        self.graph_color = graph_color   # line colour in battery panel

    # ── Frame movement ────────────────────────────────────────────────────────

    def update_movement(self, dt):
        """Advance tower one frame along waypoint path; update current_node."""
        if not self.is_moving or not self.path_waypoints:
            return

        if self.current_wp_idx >= len(self.path_waypoints):
            self._snap_to_target()
            return

        target_wp = np.array(self.path_waypoints[self.current_wp_idx], dtype=float)
        direction = target_wp[:2] - self.pos[:2]
        dist      = float(np.linalg.norm(direction))
        step      = self.speed * dt

        if step >= dist:
            self.pos[:2] = target_wp[:2]
            if len(target_wp) > 2:
                self.pos[2] = float(target_wp[2])

            nearest = _nearest_node(self.pos[:2])
            if nearest:
                self.current_node = nearest

            self.current_wp_idx += 1
            if self.current_wp_idx >= len(self.path_waypoints):
                self._snap_to_target()
        else:
            self.pos[:2] += (direction / dist) * step

    def _snap_to_target(self):
        if self.target_node and self.target_node in map_data.NODES:
            tp = map_data.NODES[self.target_node]
            self.pos[0] = float(tp[0])
            self.pos[1] = float(tp[1])
            if len(tp) > 2:
                self.pos[2] = float(tp[2])
        self.current_node   = self.target_node
        self.is_moving      = False
        self.path_waypoints = []
        self.current_wp_idx = 0

    # ── Battery update (call once per sim-second, or scale by dt) ────────────

    def update_battery(self, dt, connected_truck_positions, peer_tower_positions):
        """
        Drain battery based on:
          • Idle baseline power
          • Motion power (when moving)
          • Mesh link power (tx to other towers within COMM_RADIUS)
          • Truck service power  (tx to connected trucks)

        Mirrors the per-tower power block in simulation_3d.py run_simulation().

        Parameters
        ----------
        dt                       : sim-time seconds elapsed
        connected_truck_positions: list of (x, y) for trucks this tower serves
        peer_tower_positions     : list of (x, y) for all OTHER towers
        """
        if self.pct <= 0:
            return

        mesh_p = sum(
            _tx_power(_dist_xy(self.pos, op))
            for op in peer_tower_positions
            if _dist_xy(self.pos, op) <= COMM_RADIUS
        )
        truck_p = sum(
            _tx_power(_dist_xy(self.pos, tp))
            for tp in connected_truck_positions
        )
        motion_p = MOVE_POWER if self.is_moving else 0.0
        total_p  = IDLE_POWER + motion_p + mesh_p + truck_p

        # Energy drain: P * t / 3600  (Wh, t in seconds)
        self.energy -= total_p * (dt / 3600.0)
        self.pct     = max(0.0, (self.energy / BATT_CAP_WH) * 100.0)
        self.history.append(self.pct)


# ═════════════════════════════════════════════════════════════════════════════
#  K-Median algorithm
# ═════════════════════════════════════════════════════════════════════════════

def _compute_kmedian_cost(facility_nodes, demand_nodes):
    total = 0.0
    for dnode in demand_nodes:
        if dnode not in map_data.NODES:
            continue
        d_coord = map_data.NODES[dnode]
        min_d = min(
            _dist_xy(d_coord, map_data.NODES[f])
            for f in facility_nodes
            if f in map_data.NODES
        )
        total += min_d
    return total


def run_swarm_k_median(cars, num_towers, max_iterations=30):
    """
    Local-search k-median — identical logic to simulation_3d.py.
    Returns list of k node-key strings.
    """
    all_node_keys = list(map_data.NODES.keys())
    if not all_node_keys or not cars:
        return []

    k = min(num_towers, len(all_node_keys))

    truck_demand_nodes = []
    for car in cars:
        n = _nearest_node((car.x_m, car.y_m))
        if n:
            truck_demand_nodes.append(n)

    if not truck_demand_nodes:
        return random.sample(all_node_keys, k)

    while len(truck_demand_nodes) < k:
        truck_demand_nodes.append(random.choice(all_node_keys))

    print(f"[K-MEDIAN] running local-search (k={k}, demand={len(truck_demand_nodes)}) …")

    facilities   = random.sample(all_node_keys, k)
    current_cost = _compute_kmedian_cost(facilities, truck_demand_nodes)

    for _ in range(max_iterations):
        improved = False
        for i in range(k):
            for candidate in all_node_keys:
                if candidate in facilities:
                    continue
                test_facs  = facilities.copy()
                test_facs[i] = candidate
                test_cost  = _compute_kmedian_cost(test_facs, truck_demand_nodes)
                if test_cost < current_cost:
                    facilities   = test_facs
                    current_cost = test_cost
                    improved     = True
                    break
            if improved:
                break
        if not improved:
            break

    print(f"[K-MEDIAN] done → {facilities}")
    return facilities


def assign_tower_targets(network_towers, optimal_hubs,
                         local_planner, dispatcher,
                         route_cache, get_path_fn):
    """Write new targets and rebuild waypoint paths for all towers."""
    graph = dispatcher.get_graph()
    for idx, tower in enumerate(network_towers):
        if idx >= len(optimal_hubs):
            break
        new_target = optimal_hubs[idx]
        if new_target == tower.target_node:
            continue
        tower.target_node = new_target
        if new_target == tower.current_node:
            tower.is_moving = False
            continue
        route = local_planner.compute_route(
            graph, tower.current_node, new_target, cache=route_cache
        )
        if not route:
            continue
        waypoints = get_path_fn(route)
        if not waypoints:
            continue
        tower.path_waypoints  = waypoints
        tower.current_wp_idx  = 0
        tower.is_moving       = True