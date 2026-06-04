import simpy
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================
# MAP CONFIG
# ==============================
MINE_WIDTH = 400
MINE_HEIGHT = 300
MINE_DEPTH = 120

SIM_STEP = 1
NUM_TRUCKS = 25
NUM_TOWERS = 12

# Communication
P0 = 5
PMAX = 120
D0 = 20
PATH_LOSS = 3
COMM_RADIUS = D0 * (PMAX / P0) ** (1 / PATH_LOSS)

# ==============================
# ROAD NETWORK (GRAPH)
# ==============================
nodes = {
    "A": (10,10,5),
    "B": (80,40,30),
    "C": (150,70,10),
    "D": (220,120,50),
    "E": (300,150,20),

    "F": (20,200,10),
    "G": (100,180,40),
    "H": (180,160,15),
    "I": (260,140,60),
    "J": (350,120,30),

    "K": (120,100,20),
    "L": (200,150,70),
    "M": (280,200,30)
}

# adjacency list (roads)
graph = {
    "A": ["B"],
    "B": ["A","C"],
    "C": ["B","D"],
    "D": ["C","E"],
    "E": ["D"],

    "F": ["G"],
    "G": ["F","H","K"],
    "H": ["G","I"],
    "I": ["H","J"],
    "J": ["I"],

    "K": ["G","L"],
    "L": ["K","M"],
    "M": ["L"]
}

# ==============================
# HELPERS
# ==============================
def distance(p1,p2):
    return math.sqrt(
        (p1[0]-p2[0])**2 +
        (p1[1]-p2[1])**2 +
        (p1[2]-p2[2])**2
    )

def interpolate(p1,p2,t):
    return [
        p1[0]+(p2[0]-p1[0])*t,
        p1[1]+(p2[1]-p1[1])*t,
        p1[2]+(p2[2]-p1[2])*t
    ]

# ==============================
# TRUCK CLASS (GRAPH BASED)
# ==============================
class Truck:

    def __init__(self, env):
        self.curr = random.choice(list(nodes.keys()))
        self.next = random.choice(graph[self.curr])
        self.prev = None
        self.t = 0
        self.pos = list(nodes[self.curr])

        env.process(self.move(env))

    def move(self, env):
        speed = 0.02

        while True:
            self.t += speed

            p1 = nodes[self.curr]
            p2 = nodes[self.next]

            self.pos = interpolate(p1, p2, self.t)

            if self.t >= 1:
                self.t = 0
                self.prev = self.curr
                self.curr = self.next

                neighbors = graph[self.curr]

                # dead-end → go back
                if len(neighbors) == 1:
                    self.next = neighbors[0]
                else:
                    # avoid going back unless necessary
                    choices = [n for n in neighbors if n != self.prev]
                    self.next = random.choice(choices)

            yield env.timeout(SIM_STEP)

# ==============================
# TOWER CLASS (GRAPH BASED)
# ==============================
class Tower:

    def __init__(self, env, name):
        self.name = name

        self.curr = random.choice(list(nodes.keys()))
        self.next = random.choice(graph[self.curr])
        self.prev = None
        self.t = 0
        self.pos = list(nodes[self.curr])

        env.process(self.move(env))

    def move(self, env):
        speed = 0.01

        while True:
            self.t += speed

            p1 = nodes[self.curr]
            p2 = nodes[self.next]

            self.pos = interpolate(p1, p2, self.t)

            if self.t >= 1:
                self.t = 0
                self.prev = self.curr
                self.curr = self.next

                neighbors = graph[self.curr]

                if len(neighbors) == 1:
                    self.next = neighbors[0]
                else:
                    choices = [n for n in neighbors if n != self.prev]
                    self.next = random.choice(choices)

            yield env.timeout(SIM_STEP)

# ==============================
# SIMULATION SETUP
# ==============================
env = simpy.Environment()

trucks = [Truck(env) for _ in range(NUM_TRUCKS)]
towers = [Tower(env, f"T{i}") for i in range(NUM_TOWERS)]

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

    # draw roads
    for node, neighbors in graph.items():
        for nb in neighbors:
            p1 = nodes[node]
            p2 = nodes[nb]
            ax.plot([p1[0],p2[0]],
                    [p1[1],p2[1]],
                    [p1[2],p2[2]])

    # draw nodes
    for name, pos in nodes.items():
        ax.scatter(pos[0],pos[1],pos[2],s=50)
        ax.text(pos[0],pos[1],pos[2],name)

    # towers
    for t in towers:
        ax.scatter(t.pos[0],t.pos[1],t.pos[2],marker='^',s=80)

    # trucks + connection
    for i,tr in enumerate(trucks):

        ax.scatter(tr.pos[0],tr.pos[1],tr.pos[2],marker='o')

        in_range = [(tw, distance(tr.pos, tw.pos))
                    for tw in towers
                    if distance(tr.pos, tw.pos) <= COMM_RADIUS]

        if in_range:
            tower,_ = min(in_range, key=lambda x:x[1])
            ax.plot([tr.pos[0],tower.pos[0]],
                    [tr.pos[1],tower.pos[1]],
                    [tr.pos[2],tower.pos[2]])
            conn = tower.name
        else:
            conn = "None"

        dataset.append({
            "time":timestamp,
            "truck":i,
            "x":tr.pos[0],
            "y":tr.pos[1],
            "z":tr.pos[2],
            "connection":conn
        })

    timestamp += 1

    plt.draw()
    plt.pause(0.05)

    if timestamp > 500:
        break

# save
pd.DataFrame(dataset).to_csv("graph_based_simulation.csv", index=False)

print("Graph-based realistic simulation complete!")