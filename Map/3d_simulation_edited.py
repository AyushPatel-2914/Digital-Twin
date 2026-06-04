import pygame
import simpy
import math
import random
import pickle
import heapq
import pandas as pd
import json
import map_data

# ==============================
# LOAD CONFIG
# ==============================
with open("config.json") as f:
    cfg = json.load(f)

NUM_TRUCKS = cfg["NUM_TRUCKS"]
NUM_TOWERS = cfg["NUM_TOWERS"]
SPEED_TRUCK = cfg["SPEED_TRUCK"]
SPEED_TOWER = cfg["SPEED_TOWER"]
SIM_STEP = cfg["SIM_STEP"]

MOVE_POWER = cfg["MOVE_POWER"]
IDLE_POWER = cfg["IDLE_POWER"]

P0 = cfg["P0"]
PMAX = cfg["PMAX"]
D0 = cfg["D0"]
PATH_LOSS = cfg["PATH_LOSS"]

BATTERY_VOLTAGE = cfg["BATTERY_VOLTAGE_MAX"]
BATTERY_CAPACITY_WH = (BATTERY_VOLTAGE * cfg["BATTERY_AH"]) / 10

COMM_RADIUS = D0 * (PMAX / P0) ** (1 / PATH_LOSS)

# ==============================
# DISPLAY
# ==============================
WIDTH, HEIGHT = 1400, 900

WHITE = (255,255,255)
GRAY = (120,120,120)

METERS_TO_PIXELS = 6.0
POINTS_PER_SEGMENT = 20

# ==============================
# LOAD MAP
# ==============================
with open("map_cache.pkl","rb") as f:
    road_graph = pickle.load(f)["road_graph"]

with open("waypoints.pkl","rb") as f:
    waypoints_map = pickle.load(f)

# ==============================
# HELPERS
# ==============================
def distance(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)

def tx_power(d):
    if d==0: return 0
    return min(P0*(d/D0)**PATH_LOSS, PMAX)

def generate_color(seed):
    random.seed(seed)
    return (
        random.randint(80,255),
        random.randint(80,255),
        random.randint(80,255)
    )

# ==============================
# A*
# ==============================
def heuristic(a,b):
    return distance(map_data.NODES[a], map_data.NODES[b])

def a_star(graph,start,goal):
    pq=[(0,start)]
    g={n:1e9 for n in graph}
    g[start]=0
    parent={}

    while pq:
        _,cur=heapq.heappop(pq)

        if cur==goal:
            path=[]
            while cur in parent:
                path.append(cur)
                cur=parent[cur]
            path.append(start)
            return path[::-1]

        for nei,w in graph[cur]:
            cost=g[cur]+w
            if cost<g[nei]:
                g[nei]=cost
                parent[nei]=cur
                heapq.heappush(pq,(cost+heuristic(nei,goal),nei))
    return []

# ==============================
# WAYPOINTS
# ==============================
def get_waypoints(route):
    pts=[]
    for i in range(len(route)-1):
        a,b=route[i],route[i+1]
        for chain,wps in waypoints_map.items():
            if a in chain and b in chain:
                i1,i2=chain.index(a),chain.index(b)
                if i1<i2:
                    pts+=wps[i1*POINTS_PER_SEGMENT:i2*POINTS_PER_SEGMENT]
                else:
                    pts+=wps[i2*POINTS_PER_SEGMENT:i1*POINTS_PER_SEGMENT][::-1]
                break
    return pts

# ==============================
# MOVEMENT
# ==============================
def move(env,obj):
    cur=random.choice(obj.goals)
    obj.pos=list(map_data.NODES[cur])

    while True:
        target=random.choice(obj.goals)
        while target==cur:
            target=random.choice(obj.goals)

        route=a_star(road_graph,cur,target)
        wps=get_waypoints(route)
        cur=target

        while wps:
            wp=wps[0]
            dx,dy,dz=wp[0]-obj.pos[0],wp[1]-obj.pos[1],wp[2]-obj.pos[2]
            dist=math.sqrt(dx*dx+dy*dy+dz*dz)

            slope=dz/(dist+0.1)
            speed=obj.speed*(1-max(0,slope*3))
            step=speed*SIM_STEP

            if dist<=step:
                obj.pos=list(wp)
                wps.pop(0)
            else:
                obj.pos[0]+=dx/dist*step
                obj.pos[1]+=dy/dist*step
                obj.pos[2]+=dz/dist*step

            yield env.timeout(SIM_STEP)

# ==============================
# CLASSES
# ==============================
class Truck:
    def __init__(self,env,i):
        self.id=f"truck{i}"
        self.pos=[0,0,0]
        self.speed=SPEED_TRUCK
        self.goals=map_data.LOAD_ZONES+map_data.DUMP_ZONES
        self.connected=None
        env.process(move(env,self))

class Tower:
    def __init__(self,env,i,node):
        self.id=f"T{i}"
        self.pos=list(map_data.NODES[node])
        self.speed=SPEED_TOWER
        self.goals=["main_hub","e_hub","fw_hub","n_hub","ne_hub","s_hub","sw_hub","service_hub"]

        self.energy=BATTERY_CAPACITY_WH
        self.pct=100
        env.process(move(env,self))

# ==============================
# CSV
# ==============================
def build_columns():
    cols=["time","tower_id","x","y","power","battery"]
    for i in range(NUM_TRUCKS):
        cols += [f"truck{i}_x",f"truck{i}_y"]
        for j in range(NUM_TOWERS):
            cols.append(f"truck{i}_dist_T{j}")
        cols.append(f"truck{i}_connected")
    return cols

datasets={f"T{i}":{"cols":build_columns(),"rows":[]} for i in range(NUM_TOWERS)}

# ==============================
# MAIN
# ==============================
def run():
    pygame.init()
    screen=pygame.display.set_mode((WIDTH,HEIGHT), pygame.RESIZABLE)
    clock=pygame.time.Clock()

    env=simpy.Environment()

    hubs=["main_hub","e_hub","fw_hub","n_hub","ne_hub","s_hub","sw_hub","service_hub"]

    towers={f"T{i}":Tower(env,i,random.choice(hubs)) for i in range(NUM_TOWERS)}
    trucks=[Truck(env,i) for i in range(NUM_TRUCKS)]

    TOWER_COLORS={tid:generate_color(i) for i,tid in enumerate(towers.keys())}
    TRUCK_COLORS=[generate_color(i+100) for i in range(NUM_TRUCKS)]

    scale=0.5
    pan=[WIDTH//2,HEIGHT//2]
    dragging=False
    last_mouse=(0,0)
    ZOOM=1.1

    running=True
    while running:

        for e in pygame.event.get():
            if e.type==pygame.QUIT:
                running=False

            elif e.type==pygame.MOUSEBUTTONDOWN:
                if e.button==1:
                    dragging=True
                    last_mouse=pygame.mouse.get_pos()

            elif e.type==pygame.MOUSEBUTTONUP:
                dragging=False

            elif e.type==pygame.MOUSEMOTION and dragging:
                mx,my=pygame.mouse.get_pos()
                pan[0]+=mx-last_mouse[0]
                pan[1]+=my-last_mouse[1]
                last_mouse=(mx,my)

            elif e.type==pygame.MOUSEWHEEL:
                mouse=pygame.mouse.get_pos()
                old=scale
                scale*=ZOOM if e.y>0 else 1/ZOOM
                scale=max(0.05,min(scale,5))
                pan[0]=mouse[0]-(mouse[0]-pan[0])*(scale/old)
                pan[1]=mouse[1]-(mouse[1]-pan[1])*(scale/old)

        env.run(until=env.now+SIM_STEP)

        # CONNECT
        for tr in trucks:
            best=None
            best_d=1e9
            for tid,t in towers.items():
                d=distance(tr.pos,t.pos)
                if d<best_d and d<=COMM_RADIUS:
                    best_d=d
                    best=tid
            tr.connected=best

        # ENERGY + CSV
        all_dead=True
        for tid,t in towers.items():
            if t.pct>0:
                all_dead=False

            truck_power=sum(tx_power(distance(t.pos,tr.pos))
                            for tr in trucks if tr.connected==tid)

            total=MOVE_POWER+IDLE_POWER+truck_power
            t.energy-=total*(SIM_STEP/3600)
            t.pct=max(0,(t.energy/BATTERY_CAPACITY_WH)*100)

            row=[env.now/60,tid,t.pos[0],t.pos[1],total,t.pct]

            for tr in trucks:
                row.append(tr.pos[0])
                row.append(tr.pos[1])
                for ot in towers.values():
                    row.append(distance(tr.pos,ot.pos))
                row.append(tr.connected)

            datasets[tid]["rows"].append(row)

        if all_dead:
            print("All towers dead → stopping simulation")
            running=False

        # DRAW
        screen.fill(WHITE)

        def g(p):
            return (int(p[0]*METERS_TO_PIXELS*scale+pan[0]),
                    int(p[1]*METERS_TO_PIXELS*scale+pan[1]))

        for wps in waypoints_map.values():
            pygame.draw.lines(screen,GRAY,False,[g(p) for p in wps],2)

        for tid,t in towers.items():
            sp=g(t.pos)
            color=TOWER_COLORS[tid]
            r=int(COMM_RADIUS*METERS_TO_PIXELS*scale)
            pygame.draw.circle(screen,color,sp,r,1)
            pygame.draw.circle(screen,color,sp,6)

        for i,tr in enumerate(trucks):
            sp=g(tr.pos)
            if tr.connected:
                pygame.draw.line(screen,(150,150,150),sp,g(towers[tr.connected].pos),1)
            pygame.draw.circle(screen,TRUCK_COLORS[i],sp,4)

        pygame.display.flip()
        clock.tick(60)

    for tid,d in datasets.items():
        pd.DataFrame(d["rows"],columns=d["cols"]).to_csv(f"{tid}.csv",index=False)

    pygame.quit()

if __name__=="__main__":
    run()