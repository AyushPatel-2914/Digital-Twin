import pygame
import numpy as np
import map_data
import pickle
import os

# --- SETTINGS ---
WIDTH, HEIGHT = 1200, 900
WHITE, BLACK, GRAY = (255, 255, 255), (0, 0, 0), (100, 100, 100)
PURPLE_NODE, WAYPOINT_COLOR = (150, 0, 150), (0, 150, 255)
ROAD_WIDTH_M = 8.0
ZOOM_FACTOR = 1.1
PADDING = 50
METERS_TO_PIXELS = 6.0
PIXELS_TO_METERS = 1.0 / METERS_TO_PIXELS
POINTS_PER_SEGMENT = 20

# 3D Visual Settings
HEIGHT_STRETCH = 1.5 # How much Z affects the vertical position on screen

# --- WAYPOINT GENERATION (UPDATED FOR 3D) ---
def catmull_rom_point_3d(t, p0, p1, p2, p3):
    """
    Calculates a single point (x, y, z) on a Catmull-Rom spline.
    NumPy handles the 3D vector math automatically.
    """
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * (t**2) + (-p0 + 3 * p1 - 3 * p2 + p3) * (t**3))

def generate_curvy_path_from_nodes_3d(node_list: list[np.ndarray], points_per_segment=POINTS_PER_SEGMENT) -> list[np.ndarray]:
    """Generates a smooth list of 3D waypoint coordinates [x, y, z]."""
    all_waypoints_m = []
    if not node_list or len(node_list) < 2: return []
    
    # Pad the node list with 3D nodes
    node_list_padded = [node_list[0]] + node_list + [node_list[-1]]

    for i in range(len(node_list_padded) - 3):
        p0, p1, p2, p3 = node_list_padded[i:i+4]
        
        if i == 0:
            all_waypoints_m.append(p1)

        for j in range(1, points_per_segment + 1):
            t = j / float(points_per_segment)
            point = catmull_rom_point_3d(t, p0, p1, p2, p3)
            all_waypoints_m.append(point)
    return all_waypoints_m

def generate_all_waypoints_3d():
    print("Generating 3D waypoints for all visual road chains...")
    waypoints_data = {}
    for chain in map_data.VISUAL_ROAD_CHAINS:
        chain_key = tuple(chain)
        # map_data.NODES must now contain [x, y, z]
        node_coords = [map_data.NODES[node_name] for node_name in chain_key if node_name in map_data.NODES]
        
        if len(node_coords) < 2:
            continue
        
        waypoints = generate_curvy_path_from_nodes_3d(node_coords)
        waypoints_data[chain_key] = waypoints
        
    print(f"Generated 3D waypoints for {len(waypoints_data)} chains.")
    return waypoints_data

def save_waypoints_data(waypoints_data):
    filepath = 'waypoints.pkl'
    with open(filepath, 'wb') as f:
        pickle.dump(waypoints_data, f)
    print(f"3D Waypoints saved to {filepath}")

# --- 3D TO 2D PROJECTION ---
def grid_to_screen_3d(pos_m, scale, pan):
    """
    Projects a 3D coordinate [x, y, z] to 2D screen pixels.
    Z (height) moves the point UP on the screen (negative Y).
    """
    x, y, z = pos_m
    pos_px_x = x * METERS_TO_PIXELS
    # The 'fake' 3D effect: screen Y = (world Y) - (world Z * stretch)
    pos_px_y = (y * METERS_TO_PIXELS) - (z * METERS_TO_PIXELS * HEIGHT_STRETCH)
    
    return (int(pos_px_x * scale + pan[0]), int(pos_px_y * scale + pan[1]))

def screen_to_grid_2d(pos_px, scale, pan):
    """Note: Converting screen back to 3D is ambiguous without a fixed Z."""
    grid_pos_px = ((pos_px[0] - pan[0]) / scale, (pos_px[1] - pan[1]) / scale)
    return (grid_pos_px[0] * PIXELS_TO_METERS, grid_pos_px[1] * PIXELS_TO_METERS)

# --- DRAWING FUNCTIONS ---
def draw_road_network_3d(screen, scale, pan, splines):
    road_width_px = max(1, int(ROAD_WIDTH_M * METERS_TO_PIXELS * scale))
    for waypoints in splines:
        if len(waypoints) < 2: continue
        # Project each 3D point to 2D
        road_px = [grid_to_screen_3d(p, scale, pan) for p in waypoints]
        pygame.draw.lines(screen, GRAY, False, road_px, road_width_px)
    
    for node_name, pos_m in map_data.NODES.items():
        if node_name in map_data.LOAD_ZONES: color = (0, 200, 0)
        elif node_name in map_data.DUMP_ZONES: color = (200, 0, 0)
        else: color = PURPLE_NODE
        
        # Draw node circle at projected 3D position
        pygame.draw.circle(screen, color, grid_to_screen_3d(pos_m, scale, pan), max(2, int(scale * 4)))

def draw_waypoints_3d(screen, scale, pan, waypoints_map):
    for waypoints_list in waypoints_map.values():
        for point_m in waypoints_list:
            pygame.draw.circle(screen, WAYPOINT_COLOR, grid_to_screen_3d(point_m, scale, pan), max(1, int(scale * 1.5)))

# --- MAIN EDITOR LOOP ---
def run_waypoint_editor():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("3D Waypoint Editor")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 16)

    background_splines_map = generate_all_waypoints_3d()
    generated_waypoints_map = {}
    status_text = "3D MODE: Press [A] to Generate, [S] to Save"

    # --- View State ---
    all_nodes_m = list(map_data.NODES.values()) if map_data.NODES else [np.array([0,0,0])]
    min_x_m = min(p[0] for p in all_nodes_m)
    max_x_m = max(p[0] for p in all_nodes_m)
    min_y_m = min(p[1] for p in all_nodes_m)
    max_y_m = max(p[1] for p in all_nodes_m)
    
    map_w_m = max(1.0, max_x_m - min_x_m)
    map_h_m = max(1.0, max_y_m - min_y_m)
    
    scale = min((WIDTH - PADDING * 2) / (map_w_m * METERS_TO_PIXELS), 
                (HEIGHT - PADDING * 2) / (map_h_m * METERS_TO_PIXELS))
    
    pan = [PADDING - (min_x_m * METERS_TO_PIXELS * scale), 
           PADDING - (min_y_m * METERS_TO_PIXELS * scale)]
    
    mouse_dragging, last_mouse_pos = False, None

    running = True
    while running:
        clock.tick(60)
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    generated_waypoints_map = generate_all_waypoints_3d()
                    status_text = f"Generated 3D waypoints. Press [S] to save."
                elif event.key == pygame.K_s:
                    if generated_waypoints_map:
                        save_waypoints_data(generated_waypoints_map)
                        status_text = "Saved 3D waypoints to waypoints.pkl"
                elif event.key == pygame.K_l:
                    if os.path.exists('waypoints.pkl'):
                        with open('waypoints.pkl', 'rb') as f:
                            generated_waypoints_map = pickle.load(f)
                        status_text = "Loaded 3D waypoints."

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3: 
                    mouse_dragging, last_mouse_pos = True, event.pos
                elif event.button in (4, 5): 
                    zoom_factor = ZOOM_FACTOR if event.button == 4 else 1 / ZOOM_FACTOR
                    scale *= zoom_factor

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3: mouse_dragging = False
            
            elif event.type == pygame.MOUSEMOTION:
                if mouse_dragging:
                    dx, dy = mouse_pos[0] - last_mouse_pos[0], mouse_pos[1] - last_mouse_pos[1]
                    pan[0] += dx
                    pan[1] += dy
                    last_mouse_pos = mouse_pos

        # --- Drawing ---
        screen.fill(WHITE)
        
        # Draw 3D road network
        draw_road_network_3d(screen, scale, pan, background_splines_map.values())
        
        # Draw 3D Waypoints
        if generated_waypoints_map:
            draw_waypoints_3d(screen, scale, pan, generated_waypoints_map)

        # --- HUD ---
        hud_texts = [
            "3D Waypoint Editor",
            "[A] Generate 3D Paths | [S] Save | [L] Load",
            "Right-Click+Drag to Pan | Scroll to Zoom",
            status_text
        ]
        for i, text in enumerate(hud_texts):
            text_surface = font.render(text, True, BLACK)
            screen.blit(text_surface, (10, 10 + i * 20))

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    run_waypoint_editor()