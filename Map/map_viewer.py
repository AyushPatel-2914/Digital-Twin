import pygame
import numpy as np
import math
import map_data

# --- VISUAL & GAME SETTINGS ---
WIDTH, HEIGHT = 1200, 900
WHITE, GRAY, PURPLE_NODE, ORANGE = (255, 255, 255), (100, 100, 100), (150, 0, 150), (255, 165, 0)
BLACK = (0, 0, 0)

ROAD_WIDTH_M = 8.0
ZOOM_FACTOR = 1.1
PADDING = 50
METERS_TO_PIXELS = 6.0
PIXELS_TO_METERS = 1.0 / METERS_TO_PIXELS

# 3D Depth Settings
# Higher PITCH makes the elevation changes look more dramatic
PITCH = 1.2 

# --- GLOBAL CACHE for drawing ---
PRE_CALCULATED_SPLINES = []

# --- 3D HELPER FUNCTIONS ---
def catmull_rom_point_3d(t, p0, p1, p2, p3):
    """Calculates 3D coordinates [x, y, z] using Catmull-Rom."""
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * (t**2) + (-p0 + 3 * p1 - 3 * p2 + p3) * (t**3))

def generate_curvy_path_from_nodes_3d(node_list: list[np.ndarray], points_per_segment=20) -> list[np.ndarray]:
    """Generates a smooth 3D list of points."""
    all_waypoints_m = []
    if not node_list or len(node_list) < 2: return []
    
    node_list_padded = [node_list[0]] + node_list + [node_list[-1]]

    for i in range(len(node_list_padded) - 3):
        p0, p1, p2, p3 = node_list_padded[i:i+4]
        if i == 0: all_waypoints_m.append(p1)
        for j in range(1, points_per_segment + 1):
            t = j / float(points_per_segment)
            point = catmull_rom_point_3d(t, p0, p1, p2, p3)
            all_waypoints_m.append(point)
    return all_waypoints_m

# --- 3D Projection Functions ---
def grid_to_screen_3d(pos_m, scale, pan):
    """
    Projects 3D [x, y, z] to 2D screen pixels.
    Elevation (z) pulls the point UP (negative screen y).
    """
    x, y, z = pos_m
    
    # Calculate base 2D position
    px = x * METERS_TO_PIXELS
    py = y * METERS_TO_PIXELS
    
    # Apply vertical offset based on height (z)
    # Pitch determines how much height 'distorts' the vertical axis
    px_final = px * scale + pan[0]
    py_final = (py * scale) - (z * METERS_TO_PIXELS * scale * PITCH) + pan[1]
    
    return (int(px_final), int(py_final))

def screen_to_grid_2d(pos_px, scale, pan):
    """Reverse projection (Approximates X, Y assuming Z=0)."""
    gx = (pos_px[0] - pan[0]) / scale * PIXELS_TO_METERS
    gy = (pos_px[1] - pan[1]) / scale * PIXELS_TO_METERS
    return (gx, gy)

def draw_road_network_3d(screen, scale, pan):
    road_width_px = max(1, int(ROAD_WIDTH_M * METERS_TO_PIXELS * scale))
    
    # 1. Draw road splines (projected from 3D)
    for waypoints in PRE_CALCULATED_SPLINES:
        if len(waypoints) < 2: continue
        road_px = [grid_to_screen_3d(p, scale, pan) for p in waypoints]
        pygame.draw.lines(screen, GRAY, False, road_px, road_width_px)
    
    # 2. Draw Nodes (projected from 3D)
    for node_name, pos_m in map_data.NODES.items():
        if node_name in map_data.LOAD_ZONES: color = (0, 200, 0)
        elif node_name in map_data.DUMP_ZONES: color = (200, 0, 0)
        elif node_name in map_data.FUEL_ZONES: color = ORANGE
        else: color = PURPLE_NODE
        
        screen_pos = grid_to_screen_3d(pos_m, scale, pan)
        pygame.draw.circle(screen, color, screen_pos, max(2, int(scale * 4)))
        
        # Draw a small shadow or 'ground line' if node is elevated
        if abs(pos_m[2]) > 0.1:
            ground_pos = grid_to_screen_3d([pos_m[0], pos_m[1], 0], scale, pan)
            pygame.draw.line(screen, (200, 200, 200), screen_pos, ground_pos, 1)

def run_viewer():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("3D Map Viewer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 14)

    # --- Pre-calculate road visuals ---
    print("Pre-calculating 3D road visuals...")
    for chain in map_data.VISUAL_ROAD_CHAINS:
        node_coords = [map_data.NODES[node_name] for node_name in chain if node_name in map_data.NODES]
        if len(node_coords) < 2: continue
        PRE_CALCULATED_SPLINES.append(generate_curvy_path_from_nodes_3d(node_coords))

    # --- Setup View ---
    all_nodes_m = list(map_data.NODES.values())
    min_x_m = min(p[0] for p in all_nodes_m)
    max_x_m = max(p[0] for p in all_nodes_m)
    min_y_m = min(p[1] for p in all_nodes_m)
    max_y_m = max(p[1] for p in all_nodes_m)
    
    map_w_m = max(1.0, max_x_m - min_x_m)
    map_h_m = max(1.0, max_y_m - min_y_m)
    
    scale = min((WIDTH - PADDING * 2) / (map_w_m * METERS_TO_PIXELS), (HEIGHT - PADDING * 2) / (map_h_m * METERS_TO_PIXELS))
    pan = [PADDING - (min_x_m * METERS_TO_PIXELS * scale), PADDING - (min_y_m * METERS_TO_PIXELS * scale)]
    
    mouse_dragging, last_mouse_pos = False, None
    show_node_names = False

    running = True
    while running:
        clock.tick(60)
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n:
                    show_node_names = not show_node_names
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: 
                    mouse_dragging, last_mouse_pos = True, event.pos
                elif event.button in (4, 5):
                    zoom_factor = ZOOM_FACTOR if event.button == 4 else 1 / ZOOM_FACTOR
                    scale *= zoom_factor
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1: 
                mouse_dragging = False
            elif event.type == pygame.MOUSEMOTION and mouse_dragging:
                dx, dy = event.pos[0] - last_mouse_pos[0], event.pos[1] - last_mouse_pos[1]
                pan[0] += dx
                pan[1] += dy
                last_mouse_pos = event.pos

        # --- Drawing ---
        screen.fill(WHITE)
        
        # Draw the road network using the new 3D logic
        draw_road_network_3d(screen, scale, pan)

        if show_node_names:
            for name, pos_m in map_data.NODES.items():
                s_pos = grid_to_screen_3d(pos_m, scale, pan)
                text_surface = font.render(f"{name} (Z:{pos_m[2]})", True, BLACK)
                screen.blit(text_surface, (s_pos[0] + 8, s_pos[1]))

        # --- HUD ---
        hud_text = font.render(f"3D Viewer | Scale: {scale:.2f} | Tilt: {PITCH} | Toggle Node Names: N", True, BLACK)
        screen.blit(hud_text, (10, 10))

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    run_viewer()