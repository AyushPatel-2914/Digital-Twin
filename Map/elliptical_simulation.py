import simpy
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================
# CONFIG
# ==============================
MINE_WIDTH = 120
MINE_HEIGHT = 80
MINE_DEPTH = 50

SIM_STEP = 1

# Ellipse parameters
A = 40
B = 20
H = 15
CENTER = (60, 40, 20)

# Communication parameters
P0 = 5
PMAX = 120
D0 = 10
PATH_LOSS = 3
COMM_RADIUS = D0 * (PMAX / P0) ** (1 / PATH_LOSS)

# ==============================
# FUNCTIONS
# ==============================
def ellipse_point(theta):
    x = CENTER[0] + A * math.cos(theta)
    y = CENTER[1] + B * math.sin(theta)
    z = CENTER[2] + H * math.sin(theta)   # inclined loop
    return [x, y, z]

def distance(p1, p2):
    return math.sqrt(
        (p1[0]-p2[0])**2 +
        (p1[1]-p2[1])**2 +
        (p1[2]-p2[2])**2
    )

# ==============================
# AUTO TOWER PLACEMENT
# ==============================
PERIMETER = 2 * math.pi * math.sqrt((A*A + B*B)/2)

NUM_TOWERS = int(PERIMETER / (1.8 * COMM_RADIUS)) + 1
print("Number of towers:", NUM_TOWERS)

towers = {}
for i in range(NUM_TOWERS):
    theta = (2 * math.pi * i) / NUM_TOWERS
    towers[f"T{i}"] = ellipse_point(theta)

# ==============================
# TRUCK CLASS
# ==============================
class Truck:

    def __init__(self, env):
        self.theta = random.uniform(0, 2*math.pi)
        self.pos = ellipse_point(self.theta)
        env.process(self.move(env))

    def move(self, env):
        while True:
            self.theta += 0.05   # speed
            self.pos = ellipse_point(self.theta)
            yield env.timeout(SIM_STEP)

# ==============================
# SIMULATION SETUP
# ==============================
NUM_TRUCKS = 18

env = simpy.Environment()
trucks = [Truck(env) for _ in range(NUM_TRUCKS)]

dataset = []

# ==============================
# VISUALIZATION
# ==============================
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

timestamp = 0

while True:

    env.run(until=env.now + SIM_STEP)
    ax.clear()

    ax.set_xlim(0, MINE_WIDTH)
    ax.set_ylim(0, MINE_HEIGHT)
    ax.set_zlim(0, MINE_DEPTH)

    # ---------------------------
    # Draw ellipse
    # ---------------------------
    thetas = [i * 0.05 for i in range(200)]
    xs, ys, zs = [], [], []

    for t in thetas:
        p = ellipse_point(t)
        xs.append(p[0])
        ys.append(p[1])
        zs.append(p[2])

    ax.plot(xs, ys, zs)

    # ---------------------------
    # Plot towers
    # ---------------------------
    for name, pos in towers.items():
        ax.scatter(pos[0], pos[1], pos[2], marker='^', s=80)
        ax.text(pos[0], pos[1], pos[2], name)

    # ---------------------------
    # Plot trucks + connections
    # ---------------------------
    for i, t in enumerate(trucks):

        ax.scatter(t.pos[0], t.pos[1], t.pos[2], marker='o')

        # find towers in range
        in_range = [(name, distance(t.pos, pos))
                    for name, pos in towers.items()
                    if distance(t.pos, pos) <= COMM_RADIUS]

        if in_range:
            conn = min(in_range, key=lambda x: x[1])[0]
            tpos = towers[conn]

            ax.plot([t.pos[0], tpos[0]],
                    [t.pos[1], tpos[1]],
                    [t.pos[2], tpos[2]])
        else:
            conn = "None"  # should not happen

        dataset.append({
            "timestamp": timestamp,
            "truck_id": i,
            "x": t.pos[0],
            "y": t.pos[1],
            "z": t.pos[2],
            "connection": conn
        })

    timestamp += 1

    plt.draw()
    plt.pause(0.05)

    if timestamp > 500:
        break

# ==============================
# SAVE CSV
# ==============================
pd.DataFrame(dataset).to_csv("final_inclined_ellipse.csv", index=False)

print("Simulation completed successfully!")