import pygame
import simpy
import math
import random
import pickle
import map_data
import heapq
import pandas as pd

# ==============================
# CONFIGURATION
# ==============================
WIDTH, HEIGHT = 1200, 900
GRAPH_HEIGHT = 200

WHITE, BLACK, GRAY = (255,255,255),(0,0,0),(100,100,100)
PURPLE_NODE, GREEN, RED = (150,0,150),(0,200,0),(200,0,0)
TRUCK_COLOR = (0,255,255)

METERS_TO_PIXELS = 6.0
POINTS_PER_SEGMENT = 20

NUM_TRUCKS = 20
NUM_TOWERS = 8

# ==============================
# RF + BATTERY
# ==============================
BATTERY_VOLTAGE_MAX = 26.5
BATTERY_VOLTAGE_MIN = 22.0
BATTERY_AH = 150

BATTERY_SCALE_FACTOR = 10
BATTERY_CAPACITY_WH = (BATTERY_VOLTAGE_MAX * BATTERY_AH) / BATTERY_SCALE_FACTOR

MOVE_POWER = 4
IDLE_POWER = 3

P0 = 5
PMAX = 250
D0 = 80
PATH_LOSS = 2.5
COMM_RADIUS = D0 * (PMAX / P0) ** (1 / PATH_LOSS)

SPEED_TOWER = 5.0
SPEED_TRUCK = 12.0

SIM_STEP = 1
TIME_MULTIPLIER = 1

# ==============================
# DATA STORAGE
# ==============================
tower_datasets = {f"T{i}": [] for i in range(NUM_TOWERS)}

def generate_color(i):
    random.seed(i)
    return (random.randint(80,255),random.randint(80,255),random.randint(80,255))

# ==============================
# LOAD MAP
# ==============================
with open('map_cache.pkl','rb') as f:
    road_graph = pickle.load(f)['road_graph']
with open('waypoints.pkl','rb') as f:
    waypoints_map = pickle.load(f)

# ==============================
# HELPERS
# ==============================
def distance(p1,p2):
    return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def tx_power(d):
    if d==0: return 0
    return P0*(d/D0)**PATH_LOSS

# ==============================
# PATHFINDING
# ==============================
def a_star(graph,start,goal):
    open_set=[(0,start)]
    came={}
    g={n:float('inf') for n in graph}
    g[start]=0

    while open_set:
        _,cur=heapq.heappop(open_set)
        if cur==goal:
            path=[]
            while cur in came:
                path.append(cur)
                cur=came[cur]
            path.append(start)
            return list(reversed(path))

        for nei,w in graph[cur]:
            temp=g[cur]+w
            if temp<g[nei]:
                came[nei]=cur
                g[nei]=temp
                heapq.heappush(open_set,(temp,nei))
    return []

def get_waypoints(route):
    final=[]
    if not route: return final
    for i in range(len(route)-1):
        s,e=route[i],route[i+1]
        for chain,wps in waypoints_map.items():
            try:
                idx=chain.index(s)
                if chain[idx+1]==e:
                    final.extend(wps[idx*POINTS_PER_SEGMENT:(idx+1)*POINTS_PER_SEGMENT])
                    break
            except: pass
    return final

# ==============================
# MOVEMENT
# ==============================
def navigate(env,entity):
    cur=random.choice(entity.allowed_goals)
    entity.pos[:]=map_data.NODES[cur]

    while True:
        if hasattr(entity,"pct") and entity.pct<=0:
            yield env.timeout(SIM_STEP)
            continue

        target=random.choice(entity.allowed_goals)
        while target==cur:
            target=random.choice(entity.allowed_goals)

        route=a_star(road_graph,cur,target)
        wps=get_waypoints(route)
        cur=target

        while wps:
            if hasattr(entity,"pct") and entity.pct<=0:
                break

            wp=wps[0]
            dx,dy=wp[0]-entity.pos[0],wp[1]-entity.pos[1]
            dist=math.sqrt(dx*dx+dy*dy)
            step=entity.speed*SIM_STEP

            if dist<=step:
                entity.pos[:]=wp
                wps.pop(0)
            else:
                entity.pos[0]+=(dx/dist)*step
                entity.pos[1]+=(dy/dist)*step

            yield env.timeout(SIM_STEP)

# ==============================
# CLASSES
# ==============================
class Truck:
    def __init__(self,env,i):
        self.id=f"T{i}"
        self.pos=[0,0]
        self.speed=SPEED_TRUCK
        self.allowed_goals=map_data.LOAD_ZONES+map_data.DUMP_ZONES
        self.connected_tower=None
        env.process(navigate(env,self))

class Tower:
    def __init__(self,env,tid):
        self.id=tid
        self.pos=[0,0]
        self.speed=SPEED_TOWER
        self.allowed_goals=['main_hub','e_hub','sw_hub','fw_hub','n_hub','s_hub']
        self.energy=BATTERY_CAPACITY_WH
        self.pct=100
        self.history=[]
        env.process(navigate(env,self))

# ==============================
# DRAW GRAPH
# ==============================
def draw_graph(screen,towers,font):
    rect=pygame.Rect(50,HEIGHT-GRAPH_HEIGHT+20,WIDTH-100,GRAPH_HEIGHT-40)
    pygame.draw.rect(screen,(30,30,30),rect)
    pygame.draw.rect(screen,WHITE,rect,2)

    for t in towers.values():
        if len(t.history)>1:
            pts=[]
            for i,p in enumerate(t.history):
                x=rect.x+(i/len(t.history))*rect.width
                y=rect.bottom-(p/100)*rect.height
                pts.append((int(x),int(y)))

            pygame.draw.lines(screen,TOWER_COLORS[t.id],False,pts,2)

            current=t.history[-1]
            screen.blit(font.render(f"{t.id}: {current:.1f}%",True,TOWER_COLORS[t.id]),
                        (pts[-1][0]+5,pts[-1][1]-10))

# ==============================
# MAIN
# ==============================
def run_simulation():
    pygame.init()
    screen=pygame.display.set_mode((WIDTH,HEIGHT))
    clock=pygame.time.Clock()
    font=pygame.font.SysFont("Consolas",14)

    nodes=list(map_data.NODES.values())
    min_x,max_x=min(p[0] for p in nodes),max(p[0] for p in nodes)
    min_y,max_y=min(p[1] for p in nodes),max(p[1] for p in nodes)

    scale=min((WIDTH-40)/((max_x-min_x)*METERS_TO_PIXELS),
              ((HEIGHT-GRAPH_HEIGHT)-40)/((max_y-min_y)*METERS_TO_PIXELS))

    pan=[20-(min_x*METERS_TO_PIXELS*scale),
         20-(min_y*METERS_TO_PIXELS*scale)]

    env=simpy.Environment()

    tower_ids=[f"T{i}" for i in range(NUM_TOWERS)]
    global TOWER_COLORS
    TOWER_COLORS={tid:generate_color(i) for i,tid in enumerate(tower_ids)}

    towers={tid:Tower(env,tid) for tid in tower_ids}
    trucks=[Truck(env,i) for i in range(NUM_TRUCKS)]

    sim_running=True
    running=True

    while running:
        for e in pygame.event.get():
            if e.type==pygame.QUIT:
                running=False

        if sim_running:
            env.run(until=env.now+SIM_STEP)

            # connectivity
            for t in trucks:
                nearest=min(towers.items(),key=lambda x:distance(t.pos,x[1].pos))
                if distance(t.pos,nearest[1].pos)<=COMM_RADIUS and nearest[1].pct>0:
                    t.connected_tower=nearest[0]
                else:
                    t.connected_tower=None

            # energy + dataset
            for tid,t in towers.items():

                if t.pct<=0:
                    t.pct=0
                    continue

                mesh=0
                for oid,ot in towers.items():
                    if oid!=tid and ot.pct>0:
                        d=distance(t.pos,ot.pos)
                        if d<=COMM_RADIUS:
                            mesh+=tx_power(d)

                truck_power=sum(tx_power(distance(t.pos,tr.pos))
                                for tr in trucks if tr.connected_tower==tid)

                total_power=MOVE_POWER+IDLE_POWER+mesh+truck_power

                t.energy-=total_power*(SIM_STEP/3600)*TIME_MULTIPLIER
                t.pct=max(0,(t.energy/BATTERY_CAPACITY_WH)*100)

                t.history.append(t.pct)
                if len(t.history)>300:
                    t.history.pop(0)

                # STORE CSV DATA
                tower_datasets[tid].append({
                    "time": env.now,
                    "x": t.pos[0],
                    "y": t.pos[1],
                    "battery_pct": t.pct,
                    "num_trucks": sum(1 for tr in trucks if tr.connected_tower==tid),
                    "mesh_links": sum(
                        1 for oid,ot in towers.items()
                        if oid!=tid and distance(t.pos,ot.pos)<=COMM_RADIUS
                    ),
                    "total_power": total_power
                })

            if all(t.pct<=0 for t in towers.values()):
                print("All towers dead → simulation stopped")
                sim_running=False

        # DRAW
        screen.fill(WHITE)
        g=lambda p:(int(p[0]*METERS_TO_PIXELS*scale+pan[0]),
                    int(p[1]*METERS_TO_PIXELS*scale+pan[1]))

        for wps in waypoints_map.values():
            if len(wps)>1:
                pygame.draw.lines(screen,GRAY,False,[g(p) for p in wps],2)

        for tr in trucks:
            if tr.connected_tower:
                pygame.draw.line(screen,(150,150,150),
                                 g(tr.pos),
                                 g(towers[tr.connected_tower].pos),1)

        for tr in trucks:
            pygame.draw.circle(screen,TRUCK_COLOR,g(tr.pos),5)

        for t in towers.values():
            pos=g(t.pos)
            if t.pct>0:
                pygame.draw.circle(screen,TOWER_COLORS[t.id],pos,
                                   int(COMM_RADIUS*METERS_TO_PIXELS*scale),1)
            pygame.draw.circle(screen,TOWER_COLORS[t.id],pos,8)

        pygame.draw.rect(screen,BLACK,(0,HEIGHT-GRAPH_HEIGHT,WIDTH,GRAPH_HEIGHT))
        draw_graph(screen,towers,font)

        pygame.display.flip()
        clock.tick(60)

    # SAVE CSV FILES
    for tid,data in tower_datasets.items():
        pd.DataFrame(data).to_csv(f"tower_{tid}.csv",index=False)

    print("CSV files saved successfully!")
    pygame.quit()

if __name__=="__main__":
    run_simulation()