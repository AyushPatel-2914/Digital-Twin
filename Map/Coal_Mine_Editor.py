"""
3D Coal Mine Editor - Configure coal capacities for load zones with Elevation Support
Matches Map_Editor.py style and auto-syncs with 3D map_data.py
"""
import pygame
import numpy as np
import json
import os
import math

# --- EDITOR SETTINGS ---
WIDTH, HEIGHT = 1200, 900
WHITE, BLACK, GRAY = (255, 255, 255), (0, 0, 0), (100, 100, 100)
GREEN, RED, PURPLE, ORANGE = (0, 200, 0), (200, 0, 0), (150, 0, 150), (255, 165, 0)
COAL_COLOR_BASE = (139, 69, 19) 
HIGHLIGHT_COLOR = (255, 255, 0)

ROAD_WIDTH_M = 8.0
ZOOM_FACTOR = 1.1
PADDING = 50
METERS_TO_PIXELS = 6.0
PIXELS_TO_METERS = 1.0 / METERS_TO_PIXELS
CLICK_THRESHOLD_PX = 25 

MAP_DATA_FILE = 'map_data.py'
CONFIG_FILE = 'mine_config.json'
DEFAULT_COAL_CAPACITY = 100
DEFAULT_TRUCK_COUNT = 5

# 3D Depth Settings (How much height affects the vertical position)
PITCH = 1.2 

# --- DATA STORAGE ---
NODES = {}
EDGES = []
LOAD_ZONES = []
DUMP_ZONES = []
VISUAL_ROAD_CHAINS = []

# Configuration data
config = {
    "truck_count": DEFAULT_TRUCK_COUNT,
    "coal_capacities": {}
}

# --- HELPER FUNCTIONS ---
def load_map_data():
    """Load 3D map data from map_data.py"""
    global NODES, EDGES, LOAD_ZONES, DUMP_ZONES, VISUAL_ROAD_CHAINS
    print(f"Loading 3D map data from {MAP_DATA_FILE}...")
    try:
        with open(MAP_DATA_FILE, 'r') as f:
            content = f.read()
        
        sandbox = {'np': np}
        exec(content, sandbox)

        NODES = sandbox.get('NODES', {})
        EDGES = sandbox.get('EDGES', [])
        LOAD_ZONES = sandbox.get('LOAD_ZONES', [])
        DUMP_ZONES = sandbox.get('DUMP_ZONES', [])
        VISUAL_ROAD_CHAINS = sandbox.get('VISUAL_ROAD_CHAINS', [])

        print(f"Loaded {len(NODES)} nodes, {len(LOAD_ZONES)} load zones.")
    except Exception as e:
        print(f"Error loading map data: {e}")
        NODES, EDGES, LOAD_ZONES, DUMP_ZONES, VISUAL_ROAD_CHAINS = {}, [], [], [], []

def load_config():
    """Load configuration from mine_config.json"""
    global config
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            print(f"Loaded config: {config['truck_count']} trucks.")
        except Exception as e:
            print(f"Error loading config: {e}")

def save_config():
    """Save configuration to mine_config.json"""
    global config
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4, sort_keys=True)
        print(f"Config saved to {CONFIG_FILE}")
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

def sync_config_with_map():
    """Synchronize config with current LOAD_ZONES from map_data.py."""
    global config
    current_mines = set(LOAD_ZONES)
    config_mines = set(config.get("coal_capacities", {}).keys())
    
    # Remove deleted mines
    for mine in (config_mines - current_mines):
        del config["coal_capacities"][mine]
    
    # Add new mines
    for mine in (current_mines - config_mines):
        config["coal_capacities"][mine] = DEFAULT_COAL_CAPACITY
    
    if (config_mines - current_mines) or (current_mines - config_mines):
        save_config()

def get_coal_color(capacity, max_capacity=500):
    """Get color intensity based on coal capacity (darker = more coal)"""
    ratio = min(capacity / max_capacity, 1.0)
    r = int(222 - ratio * 83)
    g = int(184 - ratio * 115)
    b = int(135 - ratio * 116)
    return (max(0, r), max(0, g), max(0, b))

# --- 3D PROJECTION & DRAWING ---
PRE_CALCULATED_SPLINES = []

def catmull_rom_3d(t, p0, p1, p2, p3):
    """Interpolate X, Y, and Z"""
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * (t**2) + (-p0 + 3 * p1 - 3 * p2 + p3) * (t**3))

def generate_curvy_path_3d(node_list, points_per_segment=20):
    all_waypoints = []
    if not node_list or len(node_list) < 2: return []
    padded = [node_list[0]] + node_list + [node_list[-1]]
    for i in range(len(padded) - 3):
        p0, p1, p2, p3 = padded[i:i+4]
        if i == 0: all_waypoints.append(p1)
        for j in range(1, points_per_segment + 1):
            all_waypoints.append(catmull_rom_3d(j / float(points_per_segment), p0, p1, p2, p3))
    return all_waypoints

def rebuild_splines():
    global PRE_CALCULATED_SPLINES
    PRE_CALCULATED_SPLINES = []
    for chain in VISUAL_ROAD_CHAINS:
        node_coords = [NODES[name] for name in chain if name in NODES]
        if len(node_coords) < 2: continue
        PRE_CALCULATED_SPLINES.append(generate_curvy_path_3d(node_coords))

def grid_to_screen_3d(pos_m, scale, pan):
    """Project [x, y, z] to screen coordinates"""
    x, y, z = pos_m
    px = x * METERS_TO_PIXELS * scale + pan[0]
    # Height (z) subtracts from Y position to move it 'up'
    py = (y * METERS_TO_PIXELS * scale) - (z * METERS_TO_PIXELS * scale * PITCH) + pan[1]
    return (int(px), int(py))

def get_mine_at_pos_3d(pos_px, scale, pan):
    """Check mouse click against projected mine positions"""
    for mine_name in LOAD_ZONES:
        if mine_name not in NODES: continue
        screen_pos = grid_to_screen_3d(NODES[mine_name], scale, pan)
        if math.hypot(pos_px[0] - screen_pos[0], pos_px[1] - screen_pos[1]) < CLICK_THRESHOLD_PX:
            return mine_name
    return None

def draw_road_network_3d(screen, scale, pan):
    road_width_px = max(1, int(ROAD_WIDTH_M * METERS_TO_PIXELS * scale))
    for waypoints in PRE_CALCULATED_SPLINES:
        if len(waypoints) < 2: continue
        pts = [grid_to_screen_3d(p, scale, pan) for p in waypoints]
        pygame.draw.lines(screen, GRAY, False, pts, road_width_px)
    
    for name, pos in NODES.items():
        if name in LOAD_ZONES: continue
        color = RED if name in DUMP_ZONES else PURPLE
        pygame.draw.circle(screen, color, grid_to_screen_3d(pos, scale, pan), max(2, int(scale * 3)))

def draw_coal_mines_3d(screen, scale, pan, font, hovered_mine=None):
    for mine_name in LOAD_ZONES:
        if mine_name not in NODES: continue
        pos_m = NODES[mine_name]
        pos_px = grid_to_screen_3d(pos_m, scale, pan)
        capacity = config["coal_capacities"].get(mine_name, DEFAULT_COAL_CAPACITY)
        
        radius = int(max(8, min(25, 8 + capacity / 20)) * scale)
        color = get_coal_color(capacity)
        
        # Draw shadow line to ground level (Z=0)
        if abs(pos_m[2]) > 0.1:
            ground_px = grid_to_screen_3d([pos_m[0], pos_m[1], 0], scale, pan)
            pygame.draw.line(screen, (220, 220, 220), pos_px, ground_px, 1)

        # Draw mine
        pygame.draw.circle(screen, color, pos_px, radius)
        border_col = HIGHLIGHT_COLOR if mine_name == hovered_mine else BLACK
        pygame.draw.circle(screen, border_col, pos_px, radius, 3 if mine_name == hovered_mine else 2)
        
        if scale > 0.4:
            cap_surf = font.render(str(capacity), True, WHITE)
            screen.blit(cap_surf, cap_surf.get_rect(center=pos_px))

def show_input_dialog(screen, font, title, current_value):
    dialog_rect = pygame.Rect((WIDTH - 300) // 2, (HEIGHT - 150) // 2, 300, 150)
    input_text = str(current_value)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN: return int(input_text) if input_text else 0
                if event.key == pygame.K_ESCAPE: return None
                if event.key == pygame.K_BACKSPACE: input_text = input_text[:-1]
                elif event.unicode.isdigit(): input_text += event.unicode
        
        pygame.draw.rect(screen, WHITE, dialog_rect)
        pygame.draw.rect(screen, BLACK, dialog_rect, 3)
        
        t_surf = font.render(title, True, BLACK)
        screen.blit(t_surf, (dialog_rect.x + 15, dialog_rect.y + 20))
        
        box_rect = pygame.Rect(dialog_rect.x + 15, dialog_rect.y + 60, 270, 40)
        pygame.draw.rect(screen, (240, 240, 240), box_rect)
        pygame.draw.rect(screen, BLACK, box_rect, 1)
        
        val_surf = font.render(input_text + "|", True, BLACK)
        screen.blit(val_surf, (box_rect.x + 10, box_rect.y + 10))
        
        hint = font.render("Enter: Save | Esc: Cancel", True, GRAY)
        screen.blit(hint, (dialog_rect.x + 15, dialog_rect.y + 115))
        
        pygame.display.flip()

# --- MAIN EDITOR LOOP ---
def run_editor():
    global config
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("3D Coal Mine Editor")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 16)

    load_map_data()
    load_config()
    sync_config_with_map()
    rebuild_splines()

    # --- Initial View Setup ---
    all_pos = list(NODES.values()) if NODES else [np.array([0,0,0])]
    min_x, max_x = min(p[0] for p in all_pos), max(p[0] for p in all_pos)
    min_y, max_y = min(p[1] for p in all_pos), max(p[1] for p in all_pos)
    
    map_w = max(1.0, max_x - min_x)
    map_h = max(1.0, max_y - min_y)
    
    scale = min((WIDTH - 150) / (map_w * 6), (HEIGHT - 150) / (map_h * 6))
    pan = [WIDTH // 2 - ((min_x + max_x) / 2 * 6 * scale), HEIGHT // 2 - ((min_y + max_y) / 2 * 6 * scale)]

    status_text = "Click a mine to edit capacity"
    mouse_dragging = False
    hovered_mine = None

    running = True
    while running:
        clock.tick(60)
        m_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    status_text = "Configuration Saved!" if save_config() else "Save Error!"
                elif event.key == pygame.K_t:
                    res = show_input_dialog(screen, font, "Global Truck Count:", config["truck_count"])
                    if res is not None: config["truck_count"] = res
                elif event.key == pygame.K_r:
                    load_map_data()
                    sync_config_with_map()
                    rebuild_splines()
                    status_text = "Map Data Reloaded."

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    clicked = get_mine_at_pos_3d(m_pos, scale, pan)
                    if clicked:
                        new_val = show_input_dialog(screen, font, f"Capacity for {clicked} (kg):", config["coal_capacities"][clicked])
                        if new_val is not None:
                            config["coal_capacities"][clicked] = new_val
                elif event.button == 3:
                    mouse_dragging, last_m = True, event.pos
                elif event.button in (4, 5):
                    z_fac = ZOOM_FACTOR if event.button == 4 else 1/ZOOM_FACTOR
                    scale *= z_fac

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3: mouse_dragging = False
            
            elif event.type == pygame.MOUSEMOTION:
                if mouse_dragging:
                    pan[0] += m_pos[0] - last_m[0]
                    pan[1] += m_pos[1] - last_m[1]
                    last_m = m_pos
                else:
                    hovered_mine = get_mine_at_pos_3d(m_pos, scale, pan)

        # --- Drawing ---
        screen.fill(WHITE)
        draw_road_network_3d(screen, scale, pan)
        draw_coal_mines_3d(screen, scale, pan, font, hovered_mine)

        # HUD
        hud = [
            f"3D Coal Editor | [S]ave | [T]rucks: {config['truck_count']} | [R]eload",
            f"Panning: Right-Click | Zoom: Wheel | Mines: {len(LOAD_ZONES)}",
            status_text
        ]
        for i, text in enumerate(hud):
            screen.blit(font.render(text, True, BLACK), (10, 10 + i * 22))

        if hovered_mine:
            cap = config["coal_capacities"].get(hovered_mine, 0)
            node_z = NODES[hovered_mine][2]
            info = font.render(f"{hovered_mine} | Elev: {node_z}m | Cap: {cap}kg", True, BLACK)
            screen.blit(info, (m_pos[0] + 15, m_pos[1] + 15))

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    run_editor()