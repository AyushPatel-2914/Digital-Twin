import numpy as np
import matplotlib.pyplot as plt
import random

from map_data_py import EXPORT_NODES as NODES
from map_data_py import EXPORT_EDGES as EDGES


# ============================================================
# GRAPH BUILD
# ============================================================

graph = {}
for u, v in EDGES:
    graph.setdefault(u, []).append(v)
    graph.setdefault(v, []).append(u)


# ============================================================
# AGENT CLASS
# ============================================================

class Agent:
    def __init__(self, start_node, speed, is_tower=False):
        self.current = start_node
        self.target = random.choice(graph[start_node])
        self.pos = np.array(NODES[start_node], dtype=float)
        self.speed = speed
        self.is_tower = is_tower

        if is_tower:
            self.battery = 100
            self.battery_history = []

    def move(self, connected_trucks_positions=None):

        if self.is_tower and self.battery <= 0:
            return

        target_pos = np.array(NODES[self.target])
        direction = target_pos - self.pos
        dist = np.linalg.norm(direction)

        move_dist = 0

        if dist < self.speed:
            self.current = self.target
            self.target = random.choice(graph[self.current])
        else:
            step_vec = (direction / dist) * self.speed
            self.pos += step_vec
            move_dist = np.linalg.norm(step_vec)

        # ==============================
        # 🔋 BATTERY MODEL
        # ==============================
        if self.is_tower:

            # Idle drain
            idle_drain = 0.02

            # Distance-based communication drain
            comm_drain = 0
            if connected_trucks_positions:
                for truck_pos in connected_trucks_positions:
                    d = np.linalg.norm(truck_pos - self.pos)
                    comm_drain += 0.015 * (d / COMM_RANGE)

            # Movement drain
            move_drain = 0.002 * move_dist

            total_drain = idle_drain + comm_drain + move_drain

            self.battery -= total_drain
            self.battery = max(self.battery, 0)

            self.battery_history.append(self.battery)


# ============================================================
# CREATE AGENTS
# ============================================================

nodes_list = list(NODES.keys())

trucks = [Agent(random.choice(nodes_list), 20) for _ in range(10)]
towers = [Agent(random.choice(nodes_list), 8, is_tower=True) for _ in range(5)]

COMM_RANGE = 150


# ============================================================
# PLOT SETUP
# ============================================================

fig = plt.figure(figsize=(16, 8))

ax_main = fig.add_subplot(2, 3, 1, projection='3d')

battery_axes = []
for i in range(5):
    ax = fig.add_subplot(2, 3, i+2)
    battery_axes.append(ax)


# ============================================================
# SIMULATION LOOP
# ============================================================

step = 0

while True:

    ax_main.cla()

    # ---- DRAW ROADS ----
    for u, v in EDGES:
        if u in NODES and v in NODES:
            ax_main.plot(
                [NODES[u][0], NODES[v][0]],
                [NODES[u][1], NODES[v][1]],
                [NODES[u][2], NODES[v][2]],
                linewidth=1,
                alpha=0.2
            )

    # ---- MOVE TRUCKS ----
    for t in trucks:
        t.move()
        ax_main.scatter(t.pos[0], t.pos[1], t.pos[2], s=15)

    # ---- MOVE TOWERS ----
    for tower in towers:

        # find connected trucks first
        connected_positions = []

        for truck in trucks:
            dist = np.linalg.norm(truck.pos - tower.pos)
            if dist < COMM_RANGE:
                connected_positions.append(truck.pos)

        tower.move(connected_positions)

        # draw tower
        ax_main.scatter(tower.pos[0], tower.pos[1], tower.pos[2], s=120)

        # draw range
        theta = np.linspace(0, 2*np.pi, 40)
        ax_main.plot(
            tower.pos[0] + COMM_RANGE*np.cos(theta),
            tower.pos[1] + COMM_RANGE*np.sin(theta),
            np.full_like(theta, tower.pos[2]),
            linestyle='dashed'
        )

    # ========================================================
    # 🔗 CONNECTION (nearest tower only)
    # ========================================================

    for truck in trucks:

        nearest_tower = None
        min_dist = float('inf')

        for tower in towers:
            if tower.battery <= 0:
                continue

            d = np.linalg.norm(truck.pos - tower.pos)

            if d < COMM_RANGE and d < min_dist:
                min_dist = d
                nearest_tower = tower

        if nearest_tower:
            ax_main.plot(
                [truck.pos[0], nearest_tower.pos[0]],
                [truck.pos[1], nearest_tower.pos[1]],
                [truck.pos[2], nearest_tower.pos[2]],
                linewidth=1
            )

    # ========================================================
    # 📊 BATTERY GRAPHS (SEPARATE)
    # ========================================================

    for i, tower in enumerate(towers):
        battery_axes[i].cla()
        battery_axes[i].plot(tower.battery_history)
        battery_axes[i].set_title(f"Tower {i}")
        battery_axes[i].set_ylim(0, 100)

    # ========================================================
    # AXIS SETTINGS
    # ========================================================

    ax_main.set_xlim(-650, 850)
    ax_main.set_ylim(-450, 900)
    ax_main.set_zlim(-300, 300)

    ax_main.set_title(f"Step {step}")
    ax_main.view_init(elev=30, azim=120)

    plt.pause(0.01)   # FAST SIMULATION

    step += 1

    # ========================================================
    # STOP CONDITION
    # ========================================================

    if all(tower.battery <= 0 for tower in towers):
        print("All towers battery depleted. Simulation stopped.")
        break


plt.show()