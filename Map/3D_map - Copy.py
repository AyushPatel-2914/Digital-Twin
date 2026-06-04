import numpy as np
import matplotlib.pyplot as plt

import numpy as np

# --- 3D MAP DATA ---
# Each node is now np.array([X, Y, Z])

NODES = {
    "connector_1_1": np.array([150.0, 140.0, 100.0]),
    "connector_1_2": np.array([100.0, 100.0, 100.0]),
    "connector_1_3": np.array([70.0, 150.0, -100.0]),
    "connector_2_2": np.array([250.0, 250.0, 110.0]),
    "connector_2_3": np.array([260.0, 280.0, -110.0]),
    "dump_hub_1a": np.array([180.0, 280.0, -110.0]),
    "dump_hub_1b": np.array([200.0, 280.0, -100.0]),
    "dump_spur_1a_1": np.array([190.0, 310.0, 0.0]),
    "dump_spur_1a_2": np.array([180.0, 320.0, 0.0]),
    "dump_spur_1b_1": np.array([210.0, 310.0, 0.0]),
    "dump_spur_1b_2": np.array([220.0, 320.0, 0.0]),
    "dump_zone_1": np.array([180.0, 340.0, 0.0]),
    "dump_zone_10": np.array([700.0, 320.0, 0.0]),
    "dump_zone_11": np.array([750.0, 320.0, 0.0]),
    "dump_zone_12": np.array([650.0, 370.0, 0.0]),
    "dump_zone_13": np.array([700.0, 370.0, 0.0]),
    "dump_zone_14": np.array([750.0, 370.0, 0.0]),
    "dump_zone_15": np.array([50.0, 740.0, 0.0]),
    "dump_zone_16": np.array([150.0, 790.0, 0.0]),
    "dump_zone_17": np.array([250.0, 740.0, 0.0]),
    "dump_zone_18": np.array([-50.0, 740.0, 0.0]),
    "dump_zone_19": np.array([50.0, 840.0, 0.0]),
    "dump_zone_2": np.array([220.0, 340.0, 0.0]),
    "dump_zone_20": np.array([250.0, 840.0, 0.0]),
    "dump_zone_21": np.array([350.0, 740.0, 0.0]),
    "dump_zone_22": np.array([-170.0, 450.0, 0.0]),
    "dump_zone_23": np.array([-100.0, 520.0, 0.0]),
    "dump_zone_24": np.array([-30.0, 450.0, 0.0]),
    "dump_zone_3": np.array([650.0, 180.0, 0.0]),
    "dump_zone_4": np.array([700.0, 180.0, 0.0]),
    "dump_zone_5": np.array([750.0, 180.0, 0.0]),
    "dump_zone_6": np.array([650.0, 130.0, 0.0]),
    "dump_zone_7": np.array([700.0, 130.0, 0.0]),
    "dump_zone_8": np.array([750.0, 130.0, 0.0]),
    "dump_zone_9": np.array([650.0, 320.0, 0.0]),
    "e_connector_n": np.array([750.0, 225.0, 0.0]),
    "e_connector_s": np.array([750.0, 275.0, 0.0]),
    "e_grid_1_a": np.array([600.0, 200.0, 0.0]),
    "e_grid_1_b": np.array([650.0, 200.0, 0.0]),
    "e_grid_1_c": np.array([700.0, 200.0, 0.0]),
    "e_grid_1_d": np.array([750.0, 200.0, 0.0]),
    "e_grid_2_a": np.array([600.0, 150.0, 0.0]),
    "e_grid_2_b": np.array([650.0, 150.0, 0.0]),
    "e_grid_2_c": np.array([700.0, 150.0, 0.0]),
    "e_grid_2_d": np.array([750.0, 150.0, 0.0]),
    "e_grid_3_a": np.array([600.0, 300.0, 0.0]),
    "e_grid_3_b": np.array([650.0, 300.0, 0.0]),
    "e_grid_3_c": np.array([700.0, 300.0, 0.0]),
    "e_grid_3_d": np.array([750.0, 300.0, 0.0]),
    "e_grid_4_a": np.array([600.0, 350.0, 0.0]),
    "e_grid_4_b": np.array([650.0, 350.0, 0.0]),
    "e_grid_4_c": np.array([700.0, 350.0, 0.0]),
    "e_grid_4_d": np.array([750.0, 350.0, 0.0]),
    "e_haul_1": np.array([300.0, 250.0, 0.0]),
    "e_haul_2": np.array([400.0, 250.0, 0.0]),
    "e_haul_3": np.array([500.0, 250.0, 0.0]),
    "e_hub": np.array([600.0, 250.0, 0.0]),
    "fuel_1": np.array([700.0, 630.0, 0.0]),
    "fuel_2": np.array([700.0, 670.0, 0.0]),
    "fw_haul_1": np.array([-50.0, 250.0, 0.0]),
    "fw_haul_2": np.array([-150.0, 250.0, 0.0]),
    "fw_haul_3": np.array([-250.0, 250.0, 0.0]),
    "fw_hub": np.array([-350.0, 250.0, 0.0]),
    "fw_load_spur_1": np.array([-470.0, 150.0, 0.0]),
    "fw_load_spur_10": np.array([-600.0, 270.0, 0.0]),
    "fw_load_spur_11": np.array([-550.0, 470.0, 0.0]),
    "fw_load_spur_12": np.array([-200.0, 420.0, 0.0]),
    "fw_load_spur_2": np.array([-500.0, 270.0, 0.0]),
    "fw_load_spur_3": np.array([-450.0, 370.0, 0.0]),
    "fw_load_spur_4": np.array([-300.0, 320.0, 0.0]),
    "fw_load_spur_5": np.array([-520.0, 100.0, 0.0]),
    "fw_load_spur_6": np.array([-550.0, 270.0, 0.0]),
    "fw_load_spur_7": np.array([-500.0, 420.0, 0.0]),
    "fw_load_spur_8": np.array([-250.0, 370.0, 0.0]),
    "fw_load_spur_9": np.array([-570.0, 50.0, 0.0]),
    "fw_pit_1_a": np.array([-350.0, 150.0, 0.0]),
    "fw_pit_1_b": np.array([-450.0, 150.0, 0.0]),
    "fw_pit_1_c": np.array([-500.0, 250.0, 0.0]),
    "fw_pit_1_d": np.array([-450.0, 350.0, 0.0]),
    "fw_pit_1_e": np.array([-350.0, 350.0, 0.0]),
    "fw_pit_1_f": np.array([-300.0, 300.0, 0.0]),
    "fw_pit_1_g": np.array([-300.0, 200.0, 0.0]),
    "fw_pit_2_a": np.array([-350.0, 100.0, 0.0]),
    "fw_pit_2_b": np.array([-500.0, 100.0, 0.0]),
    "fw_pit_2_c": np.array([-550.0, 250.0, 0.0]),
    "fw_pit_2_d": np.array([-500.0, 400.0, 0.0]),
    "fw_pit_2_e": np.array([-350.0, 400.0, 0.0]),
    "fw_pit_2_f": np.array([-250.0, 350.0, 0.0]),
    "fw_pit_2_g": np.array([-250.0, 150.0, 0.0]),
    "fw_pit_3_a": np.array([-350.0, 50.0, 0.0]),
    "fw_pit_3_b": np.array([-550.0, 50.0, 0.0]),
    "fw_pit_3_c": np.array([-600.0, 250.0, 0.0]),
    "fw_pit_3_d": np.array([-550.0, 450.0, 0.0]),
    "fw_pit_3_e": np.array([-350.0, 450.0, 0.0]),
    "fw_pit_3_f": np.array([-200.0, 400.0, 0.0]),
    "fw_pit_3_g": np.array([-200.0, 100.0, 0.0]),
    "ix_east_1": np.array([180.0, 250.0, 0.0]),
    "ix_east_2": np.array([200.0, 250.0, 0.0]),
    "ix_north_1": np.array([150.0, 220.0, 0.0]),
    "ix_north_2": np.array([150.0, 200.0, 0.0]),
    "ix_west_1": np.array([120.0, 250.0, 0.0]),
    "ix_west_2": np.array([100.0, 250.0, 0.0]),
    "load_hub_1a": np.array([120.0, 200.0, 0.0]),
    "load_hub_1b": np.array([100.0, 200.0, 0.0]),
    "load_hub_2a": np.array([180.0, 200.0, 0.0]),
    "load_hub_2b": np.array([200.0, 200.0, 0.0]),
    "load_spur_1a_1": np.array([90.0, 190.0, 0.0]),
    "load_spur_1a_2": np.array([80.0, 180.0, 0.0]),
    "load_spur_1b_1": np.array([110.0, 190.0, 0.0]),
    "load_spur_1b_2": np.array([120.0, 180.0, 0.0]),
    "load_spur_2a_1": np.array([190.0, 170.0, 0.0]),
    "load_spur_2a_2": np.array([180.0, 150.0, 0.0]),
    "load_spur_2b_1": np.array([210.0, 170.0, 0.0]),
    "load_spur_2b_2": np.array([220.0, 150.0, 0.0]),
    "load_zone_1": np.array([80.0, 160.0, 0.0]),
    "load_zone_10": np.array([-500.0, 440.0, 0.0]),
    "load_zone_11": np.array([-250.0, 390.0, 0.0]),
    "load_zone_12": np.array([-590.0, 50.0, 0.0]),
    "load_zone_13": np.array([-600.0, 290.0, 0.0]),
    "load_zone_14": np.array([-550.0, 490.0, 0.0]),
    "load_zone_15": np.array([-200.0, 440.0, 0.0]),
    "load_zone_16": np.array([50.0, -290.0, 0.0]),
    "load_zone_17": np.array([150.0, -340.0, 0.0]),
    "load_zone_18": np.array([250.0, -290.0, 0.0]),
    "load_zone_19": np.array([0.0, -290.0, 0.0]),
    "load_zone_2": np.array([120.0, 160.0, 0.0]),
    "load_zone_20": np.array([150.0, -390.0, 0.0]),
    "load_zone_21": np.array([300.0, -290.0, 0.0]),
    "load_zone_22": np.array([-170.0, 400.0, 0.0]),
    "load_zone_23": np.array([-100.0, 330.0, 0.0]),
    "load_zone_24": np.array([500.0, -190.0, 0.0]),
    "load_zone_25": np.array([600.0, -240.0, 0.0]),
    "load_zone_26": np.array([700.0, -190.0, 0.0]),
    "load_zone_27": np.array([450.0, -190.0, 0.0]),
    "load_zone_28": np.array([600.0, -290.0, 0.0]),
    "load_zone_29": np.array([750.0, -190.0, 0.0]),
    "load_zone_3": np.array([200.0, 120.0, 0.0]),
    "load_zone_4": np.array([-490.0, 150.0, 0.0]),
    "load_zone_5": np.array([-500.0, 290.0, 0.0]),
    "load_zone_6": np.array([-450.0, 390.0, 0.0]),
    "load_zone_7": np.array([-300.0, 340.0, 0.0]),
    "load_zone_8": np.array([-540.0, 100.0, 0.0]),
    "load_zone_9": np.array([-550.0, 290.0, 0.0]),
    "main_hub": np.array([150.0, 250.0, 0.0]),
    "n_haul_1": np.array([150.0, 100.0, 0.0]),
    "n_haul_2": np.array([150.0, 0.0, 0.0]),
    "n_haul_3": np.array([150.0, -100.0, 0.0]),
    "n_hub": np.array([150.0, -200.0, 0.0]),
    "n_load_spur_1": np.array([50.0, -270.0, 0.0]),
    "n_load_spur_2": np.array([150.0, -320.0, 0.0]),
    "n_load_spur_3": np.array([250.0, -270.0, 0.0]),
    "n_load_spur_4": np.array([0.0, -270.0, 0.0]),
    "n_load_spur_5": np.array([150.0, -370.0, 0.0]),
    "n_load_spur_6": np.array([300.0, -270.0, 0.0]),
    "n_q_1_a": np.array([100.0, -200.0, 0.0]),
    "n_q_1_b": np.array([50.0, -250.0, 0.0]),
    "n_q_1_c": np.array([100.0, -300.0, 0.0]),
    "n_q_1_d": np.array([200.0, -300.0, 0.0]),
    "n_q_1_e": np.array([250.0, -250.0, 0.0]),
    "n_q_1_f": np.array([200.0, -200.0, 0.0]),
    "n_q_2_a": np.array([50.0, -200.0, 0.0]),
    "n_q_2_b": np.array([0.0, -250.0, 0.0]),
    "n_q_2_c": np.array([50.0, -350.0, 0.0]),
    "n_q_2_d": np.array([250.0, -350.0, 0.0]),
    "n_q_2_e": np.array([300.0, -250.0, 0.0]),
    "n_q_2_f": np.array([250.0, -200.0, 0.0]),
    "ne_haul_1": np.array([600.0, 100.0, 0.0]),
    "ne_haul_2": np.array([600.0, 0.0, 0.0]),
    "ne_hub": np.array([600.0, -100.0, 0.0]),
    "ne_load_spur_1": np.array([500.0, -170.0, 0.0]),
    "ne_load_spur_2": np.array([600.0, -220.0, 0.0]),
    "ne_load_spur_3": np.array([700.0, -170.0, 0.0]),
    "ne_load_spur_4": np.array([450.0, -170.0, 0.0]),
    "ne_load_spur_5": np.array([600.0, -290.0, 0.0]),
    "ne_load_spur_6": np.array([750.0, -170.0, 0.0]),
    "ne_q_1_a": np.array([550.0, -100.0, 0.0]),
    "ne_q_1_b": np.array([500.0, -150.0, 0.0]),
    "ne_q_1_c": np.array([550.0, -200.0, 0.0]),
    "ne_q_1_d": np.array([650.0, -200.0, 0.0]),
    "ne_q_1_e": np.array([700.0, -150.0, 0.0]),
    "ne_q_1_f": np.array([650.0, -100.0, 0.0]),
    "ne_q_2_a": np.array([500.0, -100.0, 0.0]),
    "ne_q_2_b": np.array([450.0, -150.0, 0.0]),
    "ne_q_2_c": np.array([500.0, -250.0, 0.0]),
    "ne_q_2_d": np.array([700.0, -250.0, 0.0]),
    "ne_q_2_e": np.array([750.0, -150.0, 0.0]),
    "ne_q_2_f": np.array([700.0, -100.0, 0.0]),
    "parking_1": np.array([500.0, 630.0, 0.0]),
    "parking_2": np.array([500.0, 670.0, 0.0]),
    "purple_auto_3": np.array([150.8, 489.6, 0.0]),
    "purple_auto_fix_1": np.array([80.0, 166.7, 0.0]),
    "purple_auto_fix_2": np.array([-200.0, 250.0, 0.0]),
    "purple_auto_fix_3": np.array([-300.0, 250.0, 0.0]),
    "s_connector_1": np.array([0.0, 550.0, 0.0]),
    "s_connector_2": np.array([-50.0, 450.0, 0.0]),
    "s_dump_spur_1": np.array([50.0, 720.0, 0.0]),
    "s_dump_spur_2": np.array([150.0, 770.0, 0.0]),
    "s_dump_spur_3": np.array([250.0, 720.0, 0.0]),
    "s_dump_spur_4": np.array([-50.0, 720.0, 0.0]),
    "s_dump_spur_5": np.array([50.0, 820.0, 0.0]),
    "s_dump_spur_6": np.array([250.0, 820.0, 0.0]),
    "s_dump_spur_7": np.array([350.0, 720.0, 0.0]),
    "s_haul_1": np.array([150.0, 350.0, 0.0]),
    "s_haul_2": np.array([150.0, 450.0, 0.0]),
    "s_haul_3": np.array([150.0, 550.0, 0.0]),
    "s_hub": np.array([150.0, 650.0, 0.0]),
    "s_sp_1_a": np.array([100.0, 650.0, 0.0]),
    "s_sp_1_b": np.array([50.0, 700.0, 0.0]),
    "s_sp_1_c": np.array([100.0, 750.0, 0.0]),
    "s_sp_1_d": np.array([200.0, 750.0, 0.0]),
    "s_sp_1_e": np.array([250.0, 700.0, 0.0]),
    "s_sp_1_f": np.array([200.0, 650.0, 0.0]),
    "s_sp_2_a": np.array([0.0, 650.0, 0.0]),
    "s_sp_2_b": np.array([-50.0, 700.0, 0.0]),
    "s_sp_2_c": np.array([0.0, 750.0, 0.0]),
    "s_sp_2_d": np.array([100.0, 800.0, 0.0]),
    "s_sp_2_e": np.array([200.0, 800.0, 0.0]),
    "s_sp_2_f": np.array([300.0, 750.0, 0.0]),
    "s_sp_2_g": np.array([350.0, 700.0, 0.0]),
    "s_sp_2_h": np.array([300.0, 650.0, 0.0]),
    "service_exit_1": np.array([150.0, 280.0, 0.0]),
    "service_exit_2": np.array([160.0, 270.0, 0.0]),
    "service_haul_1": np.array([600.0, 400.0, 0.0]),
    "service_haul_2": np.array([600.0, 500.0, 0.0]),
    "service_hub": np.array([600.0, 600.0, 0.0]),
    "service_loop_1": np.array([550.0, 600.0, 0.0]),
    "service_loop_2": np.array([500.0, 650.0, 0.0]),
    "service_loop_3": np.array([550.0, 700.0, 0.0]),
    "service_loop_4": np.array([650.0, 700.0, 0.0]),
    "service_loop_5": np.array([700.0, 650.0, 0.0]),
    "service_loop_6": np.array([650.0, 600.0, 0.0]),
    "start_zone": np.array([150.0, 300.0, 0.0]),
    "sw_connector_1": np.array([-100.0, 400.0, 0.0]),
    "sw_connector_2": np.array([-50.0, 350.0, 0.0]),
    "sw_dump_spur_1": np.array([-150.0, 450.0, 0.0]),
    "sw_dump_spur_2": np.array([-100.0, 500.0, 0.0]),
    "sw_dump_spur_3": np.array([-50.0, 450.0, 0.0]),
    "sw_haul_1": np.array([100.0, 300.0, 0.0]),
    "sw_haul_2": np.array([50.0, 350.0, 0.0]),
    "sw_haul_3": np.array([0.0, 400.0, 0.0]),
    "sw_hub": np.array([-100.0, 450.0, 0.0]),
    "sw_load_spur_1": np.array([-150.0, 400.0, 0.0]),
    "sw_load_spur_2": np.array([-100.0, 350.0, 0.0]),
}

# -----------------------------
# HEIGHT FUNCTION
# -----------------------------
def get_z(x, y):
    return (
        0.02 * x
        + 0.015 * y
        + 20 * np.sin(x / 200)
        + 15 * np.cos(y / 150)
    )

# Apply height to all nodes
for key in NODES:
    x, y, _ = NODES[key]
    z = get_z(x, y)
    NODES[key] = np.array([x, y, z])

# EDGES, ZONES, and CHAINS remain unchanged structure-wise
EDGES = [
    ('connector_1_1', 'connector_1_2'),
    ('connector_1_1', 'n_haul_1'),
    ('connector_1_2', 'connector_1_3'),
    ('connector_1_3', 'purple_auto_fix_1'),
    ('connector_2_2', 'connector_2_3'),
    ('connector_2_2', 'e_haul_1'),
    ('connector_2_2', 'ix_east_2'),
    ('connector_2_3', 'dump_zone_2'),
    ('dump_hub_1a', 'dump_hub_1b'),
    ('dump_hub_1a', 's_haul_1'),
    ('dump_hub_1b', 'dump_spur_1a_1'),
    ('dump_hub_1b', 'dump_spur_1b_1'),
    ('dump_spur_1a_1', 'dump_spur_1a_2'),
    ('dump_spur_1a_2', 'dump_zone_1'),
    ('dump_spur_1b_1', 'dump_spur_1b_2'),
    ('dump_spur_1b_2', 'dump_zone_2'),
    ('e_connector_n', 'e_connector_s'),
    ('e_grid_1_a', 'e_grid_1_b'),
    ('e_grid_1_a', 'e_grid_2_a'),
    ('e_grid_1_a', 'ne_haul_1'),
    ('e_grid_1_b', 'dump_zone_3'),
    ('e_grid_1_b', 'e_grid_1_c'),
    ('e_grid_1_b', 'e_grid_2_b'),
    ('e_grid_1_c', 'dump_zone_4'),
    ('e_grid_1_c', 'e_grid_1_d'),
    ('e_grid_1_c', 'e_grid_2_c'),
    ('e_grid_1_d', 'dump_zone_5'),
    ('e_grid_1_d', 'e_connector_n'),
    ('e_grid_1_d', 'e_grid_2_d'),
    ('e_grid_2_a', 'e_grid_2_b'),
    ('e_grid_2_b', 'dump_zone_6'),
    ('e_grid_2_b', 'e_grid_2_c'),
    ('e_grid_2_c', 'dump_zone_7'),
    ('e_grid_2_c', 'e_grid_2_d'),
    ('e_grid_2_d', 'dump_zone_8'),
    ('e_grid_2_d', 'e_connector_n'),
    ('e_grid_3_a', 'e_grid_3_b'),
    ('e_grid_3_a', 'e_grid_4_a'),
    ('e_grid_3_b', 'dump_zone_9'),
    ('e_grid_3_b', 'e_grid_3_c'),
    ('e_grid_3_b', 'e_grid_4_b'),
    ('e_grid_3_c', 'dump_zone_10'),
    ('e_grid_3_c', 'e_grid_3_d'),
    ('e_grid_3_c', 'e_grid_4_c'),
    ('e_grid_3_d', 'dump_zone_11'),
    ('e_grid_3_d', 'e_connector_s'),
    ('e_grid_3_d', 'e_grid_4_d'),
    ('e_grid_4_a', 'e_grid_4_b'),
    ('e_grid_4_a', 'service_haul_1'),
    ('e_grid_4_b', 'dump_zone_12'),
    ('e_grid_4_b', 'e_grid_4_c'),
    ('e_grid_4_c', 'dump_zone_13'),
    ('e_grid_4_c', 'e_grid_4_d'),
    ('e_grid_4_d', 'dump_zone_14'),
    ('e_grid_4_d', 'e_connector_s'),
    ('e_haul_1', 'e_haul_2'),
    ('e_haul_2', 'e_haul_3'),
    ('e_haul_3', 'e_hub'),
    ('e_hub', 'e_grid_1_a'),
    ('e_hub', 'e_grid_3_a'),
    ('fw_haul_1', 'fw_haul_2'),
    ('fw_haul_2', 'purple_auto_fix_2'),
    ('fw_haul_3', 'purple_auto_fix_2'),
    ('fw_haul_3', 'purple_auto_fix_3'),
    ('fw_hub', 'fw_pit_1_a'),
    ('fw_hub', 'purple_auto_fix_3'),
    ('fw_load_spur_1', 'load_zone_4'),
    ('fw_load_spur_10', 'load_zone_13'),
    ('fw_load_spur_11', 'load_zone_14'),
    ('fw_load_spur_12', 'load_zone_15'),
    ('fw_load_spur_2', 'load_zone_5'),
    ('fw_load_spur_3', 'load_zone_6'),
    ('fw_load_spur_4', 'load_zone_7'),
    ('fw_load_spur_5', 'load_zone_8'),
    ('fw_load_spur_6', 'load_zone_9'),
    ('fw_load_spur_7', 'load_zone_10'),
    ('fw_load_spur_8', 'load_zone_11'),
    ('fw_load_spur_9', 'load_zone_12'),
    ('fw_pit_1_a', 'fw_pit_1_b'),
    ('fw_pit_1_a', 'fw_pit_2_a'),
    ('fw_pit_1_b', 'fw_load_spur_1'),
    ('fw_pit_1_b', 'fw_pit_1_c'),
    ('fw_pit_1_c', 'fw_load_spur_2'),
    ('fw_pit_1_c', 'fw_pit_1_d'),
    ('fw_pit_1_d', 'fw_load_spur_3'),
    ('fw_pit_1_d', 'fw_pit_1_e'),
    ('fw_pit_1_e', 'fw_pit_1_f'),
    ('fw_pit_1_e', 'fw_pit_2_e'),
    ('fw_pit_1_f', 'fw_load_spur_4'),
    ('fw_pit_1_f', 'purple_auto_fix_3'),
    ('fw_pit_1_g', 'fw_pit_1_a'),
    ('fw_pit_1_g', 'fw_pit_2_g'),
    ('fw_pit_1_g', 'purple_auto_fix_3'),
    ('fw_pit_2_a', 'fw_pit_2_b'),
    ('fw_pit_2_a', 'fw_pit_3_a'),
    ('fw_pit_2_b', 'fw_load_spur_5'),
    ('fw_pit_2_b', 'fw_pit_2_c'),
    ('fw_pit_2_c', 'fw_load_spur_6'),
    ('fw_pit_2_c', 'fw_pit_2_d'),
    ('fw_pit_2_d', 'fw_load_spur_7'),
    ('fw_pit_2_d', 'fw_pit_2_e'),
    ('fw_pit_2_e', 'fw_pit_2_f'),
    ('fw_pit_2_e', 'fw_pit_3_e'),
    ('fw_pit_2_f', 'fw_load_spur_8'),
    ('fw_pit_2_f', 'fw_pit_2_g'),
    ('fw_pit_2_g', 'fw_pit_2_a'),
    ('fw_pit_2_g', 'fw_pit_3_g'),
    ('fw_pit_3_a', 'fw_pit_3_b'),
    ('fw_pit_3_b', 'fw_load_spur_9'),
    ('fw_pit_3_b', 'fw_pit_3_c'),
    ('fw_pit_3_c', 'fw_load_spur_10'),
    ('fw_pit_3_c', 'fw_pit_3_d'),
    ('fw_pit_3_d', 'fw_load_spur_11'),
    ('fw_pit_3_d', 'fw_pit_3_e'),
    ('fw_pit_3_e', 'fw_pit_3_f'),
    ('fw_pit_3_f', 'fw_load_spur_12'),
    ('fw_pit_3_f', 'purple_auto_fix_2'),
    ('fw_pit_3_g', 'fw_pit_3_a'),
    ('fw_pit_3_g', 'purple_auto_fix_2'),
    ('ix_east_1', 'ix_east_2'),
    ('ix_east_2', 'dump_hub_1b'),
    ('ix_east_2', 'e_haul_1'),
    ('ix_east_2', 'load_hub_2b'),
    ('ix_north_1', 'ix_north_2'),
    ('ix_north_2', 'load_hub_1a'),
    ('ix_north_2', 'load_hub_2a'),
    ('ix_north_2', 'n_haul_1'),
    ('ix_west_1', 'ix_west_2'),
    ('ix_west_2', 'connector_1_3'),
    ('ix_west_2', 'fw_haul_1'),
    ('ix_west_2', 'load_hub_1b'),
    ('load_hub_1a', 'load_hub_1b'),
    ('load_hub_1b', 'load_spur_1a_1'),
    ('load_hub_1b', 'load_spur_1b_1'),
    ('load_hub_1b', 'purple_auto_fix_1'),
    ('load_hub_2a', 'load_hub_2b'),
    ('load_hub_2b', 'load_spur_2a_1'),
    ('load_hub_2b', 'load_spur_2b_1'),
    ('load_spur_1a_1', 'load_spur_1a_2'),
    ('load_spur_1a_2', 'purple_auto_fix_1'),
    ('load_spur_1b_1', 'load_spur_1b_2'),
    ('load_spur_1b_2', 'load_zone_2'),
    ('load_spur_2a_1', 'load_spur_2a_2'),
    ('load_spur_2a_2', 'load_zone_3'),
    ('load_spur_2b_1', 'load_spur_2b_2'),
    ('load_spur_2b_2', 'load_zone_3'),
    ('load_zone_1', 'purple_auto_fix_1'),
    ('load_zone_3', 'connector_1_1'),
    ('main_hub', 'dump_hub_1a'),
    ('main_hub', 'ix_east_1'),
    ('main_hub', 'ix_north_1'),
    ('main_hub', 'ix_west_1'),
    ('main_hub', 'sw_haul_1'),
    ('n_haul_1', 'n_haul_2'),
    ('n_haul_2', 'n_haul_3'),
    ('n_haul_3', 'n_hub'),
    ('n_hub', 'n_q_1_a'),
    ('n_load_spur_1', 'load_zone_16'),
    ('n_load_spur_2', 'load_zone_17'),
    ('n_load_spur_3', 'load_zone_18'),
    ('n_load_spur_4', 'load_zone_19'),
    ('n_load_spur_5', 'load_zone_20'),
    ('n_load_spur_6', 'load_zone_21'),
    ('n_q_1_a', 'n_q_1_b'),
    ('n_q_1_a', 'n_q_2_a'),
    ('n_q_1_b', 'n_load_spur_1'),
    ('n_q_1_b', 'n_q_1_c'),
    ('n_q_1_c', 'n_load_spur_2'),
    ('n_q_1_c', 'n_q_1_d'),
    ('n_q_1_d', 'n_q_1_e'),
    ('n_q_1_e', 'n_load_spur_3'),
    ('n_q_1_e', 'n_q_1_f'),
    ('n_q_1_f', 'n_hub'),
    ('n_q_2_a', 'n_q_2_b'),
    ('n_q_2_b', 'n_load_spur_4'),
    ('n_q_2_b', 'n_q_2_c'),
    ('n_q_2_c', 'n_load_spur_5'),
    ('n_q_2_c', 'n_q_2_d'),
    ('n_q_2_d', 'n_q_2_e'),
    ('n_q_2_e', 'n_load_spur_6'),
    ('n_q_2_e', 'n_q_2_f'),
    ('n_q_2_f', 'n_q_1_f'),
    ('ne_haul_1', 'ne_haul_2'),
    ('ne_haul_2', 'ne_hub'),
    ('ne_hub', 'ne_q_1_a'),
    ('ne_load_spur_1', 'load_zone_24'),
    ('ne_load_spur_2', 'load_zone_25'),
    ('ne_load_spur_3', 'load_zone_26'),
    ('ne_load_spur_4', 'load_zone_27'),
    ('ne_load_spur_5', 'load_zone_28'),
    ('ne_load_spur_6', 'load_zone_29'),
    ('ne_q_1_a', 'ne_q_1_b'),
    ('ne_q_1_a', 'ne_q_2_a'),
    ('ne_q_1_b', 'ne_load_spur_1'),
    ('ne_q_1_b', 'ne_q_1_c'),
    ('ne_q_1_c', 'ne_load_spur_2'),
    ('ne_q_1_c', 'ne_q_1_d'),
    ('ne_q_1_d', 'ne_q_1_e'),
    ('ne_q_1_e', 'ne_load_spur_3'),
    ('ne_q_1_e', 'ne_q_1_f'),
    ('ne_q_1_f', 'ne_hub'),
    ('ne_q_2_a', 'ne_q_2_b'),
    ('ne_q_2_b', 'ne_load_spur_4'),
    ('ne_q_2_b', 'ne_q_2_c'),
    ('ne_q_2_c', 'ne_load_spur_5'),
    ('ne_q_2_c', 'ne_q_2_d'),
    ('ne_q_2_d', 'ne_q_2_e'),
    ('ne_q_2_e', 'ne_load_spur_6'),
    ('ne_q_2_e', 'ne_q_2_f'),
    ('ne_q_2_f', 'ne_q_1_f'),
    ('s_connector_1', 's_connector_2'),
    ('s_connector_2', 'sw_hub'),
    ('s_dump_spur_1', 'dump_zone_15'),
    ('s_dump_spur_2', 'dump_zone_16'),
    ('s_dump_spur_3', 'dump_zone_17'),
    ('s_dump_spur_4', 'dump_zone_18'),
    ('s_dump_spur_5', 'dump_zone_19'),
    ('s_dump_spur_6', 'dump_zone_20'),
    ('s_dump_spur_7', 'dump_zone_21'),
    ('s_haul_1', 's_haul_2'),
    ('s_haul_2', 's_haul_3'),
    ('s_haul_3', 's_connector_1'),
    ('s_haul_3', 's_hub'),
    ('s_hub', 's_sp_1_a'),
    ('s_sp_1_a', 's_sp_1_b'),
    ('s_sp_1_a', 's_sp_2_a'),
    ('s_sp_1_b', 's_dump_spur_1'),
    ('s_sp_1_b', 's_sp_1_c'),
    ('s_sp_1_c', 's_dump_spur_2'),
    ('s_sp_1_c', 's_sp_1_d'),
    ('s_sp_1_d', 's_sp_1_e'),
    ('s_sp_1_e', 's_dump_spur_3'),
    ('s_sp_1_e', 's_sp_1_f'),
    ('s_sp_1_f', 's_hub'),
    ('s_sp_2_a', 's_sp_2_b'),
    ('s_sp_2_b', 's_dump_spur_4'),
    ('s_sp_2_b', 's_sp_2_c'),
    ('s_sp_2_c', 's_sp_2_d'),
    ('s_sp_2_d', 's_dump_spur_5'),
    ('s_sp_2_d', 's_sp_2_e'),
    ('s_sp_2_e', 's_dump_spur_6'),
    ('s_sp_2_e', 's_sp_2_f'),
    ('s_sp_2_f', 's_sp_2_g'),
    ('s_sp_2_g', 's_dump_spur_7'),
    ('s_sp_2_g', 's_sp_2_h'),
    ('s_sp_2_h', 's_sp_1_f'),
    ('service_exit_1', 'service_exit_2'),
    ('service_exit_2', 'main_hub'),
    ('service_haul_1', 'service_haul_2'),
    ('service_haul_2', 'service_hub'),
    ('service_hub', 'service_loop_1'),
    ('service_hub', 'service_loop_6'),
    ('service_loop_1', 'service_loop_2'),
    ('service_loop_2', 'parking_1'),
    ('service_loop_2', 'parking_2'),
    ('service_loop_2', 'service_loop_3'),
    ('service_loop_3', 'service_loop_4'),
    ('service_loop_5', 'fuel_1'),
    ('service_loop_5', 'fuel_2'),
    ('service_loop_5', 'service_loop_4'),
    ('service_loop_6', 'service_loop_5'),
    ('start_zone', 'service_exit_1'),
    ('sw_connector_1', 'sw_haul_3'),
    ('sw_connector_2', 'sw_haul_3'),
    ('sw_dump_spur_1', 'dump_zone_22'),
    ('sw_dump_spur_2', 'dump_zone_23'),
    ('sw_dump_spur_3', 'dump_zone_24'),
    ('sw_haul_1', 'sw_haul_2'),
    ('sw_haul_2', 'sw_haul_3'),
    ('sw_haul_3', 'sw_hub'),
    ('sw_hub', 'sw_dump_spur_1'),
    ('sw_hub', 'sw_dump_spur_2'),
    ('sw_hub', 'sw_dump_spur_3'),
    ('sw_hub', 'sw_load_spur_1'),
    ('sw_hub', 'sw_load_spur_2'),
    ('sw_load_spur_1', 'load_zone_22'),
    ('sw_load_spur_2', 'load_zone_23'),
]

# -----------------------------
# SMOOTH HEIGHT ALONG ROADS
# -----------------------------
def smooth_z(nodes, edges, iterations=3):
    for _ in range(iterations):
        new_nodes = {}
        for node in nodes:
            neighbors = [v for u, v in edges if u == node] + \
                        [u for u, v in edges if v == node]
            
            if not neighbors:
                new_nodes[node] = nodes[node]
                continue
            
            avg_z = np.mean([nodes[n][2] for n in neighbors])
            x, y, z = nodes[node]
            new_z = 0.7 * z + 0.3 * avg_z
            
            new_nodes[node] = np.array([x, y, new_z])
        
        nodes.update(new_nodes)

# Apply smoothing
smooth_z(NODES, EDGES)

LOAD_ZONES = [
    "load_zone_1", "load_zone_10", "load_zone_11", "load_zone_12", "load_zone_13",
    "load_zone_14", "load_zone_15", "load_zone_16", "load_zone_17", "load_zone_18",
    "load_zone_19", "load_zone_2", "load_zone_20", "load_zone_21", "load_zone_22",
    "load_zone_23", "load_zone_24", "load_zone_25", "load_zone_26", "load_zone_27",
    "load_zone_28", "load_zone_29", "load_zone_3", "load_zone_4", "load_zone_5",
    "load_zone_6", "load_zone_7", "load_zone_8", "load_zone_9",
]

DUMP_ZONES = [
    "dump_zone_1", "dump_zone_10", "dump_zone_11", "dump_zone_12", "dump_zone_13",
    "dump_zone_14", "dump_zone_15", "dump_zone_16", "dump_zone_17", "dump_zone_18",
    "dump_zone_19", "dump_zone_2", "dump_zone_20", "dump_zone_21", "dump_zone_22",
    "dump_zone_23", "dump_zone_24", "dump_zone_3", "dump_zone_4", "dump_zone_5",
    "dump_zone_6", "dump_zone_7", "dump_zone_8", "dump_zone_9", "parking_1", "parking_2",
]

FUEL_ZONES = [
    "fuel_1", "fuel_2",
]

VISUAL_ROAD_CHAINS = [
    ['start_zone', 'service_exit_1', 'service_exit_2', 'main_hub'],
    ['main_hub', 'ix_west_1', 'ix_west_2'],
    ['main_hub', 'ix_east_1', 'ix_east_2'],
    ['main_hub', 'ix_north_1', 'ix_north_2'],
    ['main_hub', 'dump_hub_1a', 'dump_hub_1b'],
    ['ix_west_2', 'load_hub_1b'],
    ['ix_north_2', 'load_hub_1a', 'load_hub_1b'],
    ['ix_north_2', 'load_hub_2a', 'load_hub_2b'],
    ['ix_east_2', 'load_hub_2b'],
    ['ix_east_2', 'dump_hub_1b'],
    ['connector_2_2', 'connector_2_3', 'dump_zone_2'],
    ['load_hub_1b', 'load_spur_1a_1', 'load_spur_1a_2', 'purple_auto_fix_1', 'load_zone_1'],
    ['load_hub_1b', 'load_spur_1b_1', 'load_spur_1b_2', 'load_zone_2'],
    ['load_hub_2b', 'load_spur_2a_1', 'load_spur_2a_2', 'load_zone_3'],
    ['load_hub_2b', 'load_spur_2b_1', 'load_spur_2b_2', 'load_zone_3'],
    ['dump_hub_1b', 'dump_spur_1a_1', 'dump_spur_1a_2', 'dump_zone_1'],
    ['dump_hub_1b', 'dump_spur_1b_1', 'dump_spur_1b_2', 'dump_zone_2'],
    ['load_zone_3', 'connector_1_1', 'connector_1_2', 'connector_1_3', 'purple_auto_fix_1', 'load_hub_1b'],
    ['ix_west_2', 'fw_haul_1', 'fw_haul_2', 'purple_auto_fix_2', 'fw_haul_3', 'purple_auto_fix_3', 'fw_hub'],
    ['fw_hub', 'fw_pit_1_a', 'fw_pit_1_b', 'fw_pit_1_c', 'fw_pit_1_d', 'fw_pit_1_e', 'fw_pit_1_f', 'purple_auto_fix_3', 'fw_pit_1_g', 'fw_pit_1_a'],
    ['fw_pit_1_b', 'fw_load_spur_1', 'load_zone_4'],
    ['fw_pit_1_c', 'fw_load_spur_2', 'load_zone_5'],
    ['fw_pit_1_d', 'fw_load_spur_3', 'load_zone_6'],
    ['fw_pit_1_f', 'fw_load_spur_4', 'load_zone_7'],
    ['fw_pit_1_a', 'fw_pit_2_a', 'fw_pit_2_b', 'fw_pit_2_c', 'fw_pit_2_d', 'fw_pit_2_e', 'fw_pit_1_e'],
    ['fw_pit_1_g', 'fw_pit_2_g', 'fw_pit_2_f', 'fw_pit_2_e'],
    ['fw_pit_2_b', 'fw_load_spur_5', 'load_zone_8'],
    ['fw_pit_2_c', 'fw_load_spur_6', 'load_zone_9'],
    ['fw_pit_2_d', 'fw_load_spur_7', 'load_zone_10'],
    ['fw_pit_2_f', 'fw_load_spur_8', 'load_zone_11'],
    ['fw_pit_2_a', 'fw_pit_3_a', 'fw_pit_3_b', 'fw_pit_3_c', 'fw_pit_3_d', 'fw_pit_3_e', 'fw_pit_2_e'],
    ['fw_pit_2_g', 'fw_pit_3_g', 'purple_auto_fix_2', 'fw_pit_3_f', 'fw_pit_3_e'],
    ['fw_pit_3_b', 'fw_load_spur_9', 'load_zone_12'],
    ['fw_pit_3_c', 'fw_load_spur_10', 'load_zone_13'],
    ['fw_pit_3_d', 'fw_load_spur_11', 'load_zone_14'],
    ['fw_pit_3_f', 'fw_load_spur_12', 'load_zone_15'],
    ['ix_north_2', 'connector_1_1', 'n_haul_1', 'n_haul_2', 'n_haul_3', 'n_hub'],
    ['n_hub', 'n_q_1_a', 'n_q_1_b', 'n_q_1_c', 'n_q_1_d', 'n_q_1_e', 'n_q_1_f', 'n_hub'],
    ['n_q_1_b', 'n_load_spur_1', 'load_zone_16'],
    ['n_q_1_c', 'n_load_spur_2', 'load_zone_17'],
    ['n_q_1_e', 'n_load_spur_3', 'load_zone_18'],
    ['n_q_1_a', 'n_q_2_a', 'n_q_2_b', 'n_q_2_c', 'n_q_2_d', 'n_q_2_e', 'n_q_2_f', 'n_q_1_f'],
    ['n_q_2_b', 'n_load_spur_4', 'load_zone_19'],
    ['n_q_2_c', 'n_load_spur_5', 'load_zone_20'],
    ['n_q_2_e', 'n_load_spur_6', 'load_zone_21'],
    ['connector_2_2', 'e_haul_1', 'e_haul_2', 'e_haul_3', 'e_hub'],
    ['e_hub', 'e_grid_1_a', 'e_grid_1_b', 'e_grid_1_c', 'e_grid_1_d'],
    ['e_grid_1_a', 'e_grid_2_a'],
    ['e_grid_1_b', 'e_grid_2_b'],
    ['e_grid_1_c', 'e_grid_2_c'],
    ['e_grid_1_d', 'e_grid_2_d'],
    ['e_grid_2_a', 'e_grid_2_b', 'e_grid_2_c', 'e_grid_2_d'],
    ['e_grid_1_b', 'dump_zone_3'],
    ['e_grid_1_c', 'dump_zone_4'],
    ['e_grid_1_d', 'dump_zone_5'],
    ['e_grid_2_b', 'dump_zone_6'],
    ['e_grid_2_c', 'dump_zone_7'],
    ['e_grid_2_d', 'dump_zone_8'],
    ['e_hub', 'e_grid_3_a', 'e_grid_3_b', 'e_grid_3_c', 'e_grid_3_d'],
    ['e_grid_3_a', 'e_grid_4_a'],
    ['e_grid_3_b', 'e_grid_4_b'],
    ['e_grid_3_c', 'e_grid_4_c'],
    ['e_grid_3_d', 'e_grid_4_d'],
    ['e_grid_4_a', 'e_grid_4_b', 'e_grid_4_c', 'e_grid_4_d'],
    ['e_grid_3_b', 'dump_zone_9'],
    ['e_grid_3_c', 'dump_zone_10'],
    ['e_grid_3_d', 'dump_zone_11'],
    ['e_grid_4_b', 'dump_zone_12'],
    ['e_grid_4_c', 'dump_zone_13'],
    ['e_grid_4_d', 'dump_zone_14'],
    ['e_grid_1_d', 'e_connector_n', 'e_connector_s', 'e_grid_3_d'],
    ['e_grid_2_d', 'e_connector_n'],
    ['e_grid_4_d', 'e_connector_s'],
    ['dump_hub_1a', 's_haul_1', 's_haul_2', 's_haul_3', 's_hub'],
    ['s_hub', 's_sp_1_a', 's_sp_1_b', 's_sp_1_c', 's_sp_1_d', 's_sp_1_e', 's_sp_1_f', 's_hub'],
    ['s_sp_1_b', 's_dump_spur_1', 'dump_zone_15'],
    ['s_sp_1_c', 's_dump_spur_2', 'dump_zone_16'],
    ['s_sp_1_e', 's_dump_spur_3', 'dump_zone_17'],
    ['s_sp_1_a', 's_sp_2_a', 's_sp_2_b', 's_sp_2_c', 's_sp_2_d', 's_sp_2_e', 's_sp_2_f', 's_sp_2_g', 's_sp_2_h', 's_sp_1_f'],
    ['s_sp_2_b', 's_dump_spur_4', 'dump_zone_18'],
    ['s_sp_2_d', 's_dump_spur_5', 'dump_zone_19'],
    ['s_sp_2_e', 's_dump_spur_6', 'dump_zone_20'],
    ['s_sp_2_g', 's_dump_spur_7', 'dump_zone_21'],
    ['s_haul_3', 's_connector_1', 's_connector_2'],
    ['main_hub', 'sw_haul_1', 'sw_haul_2', 'sw_haul_3', 'sw_hub'],
    ['sw_hub', 'sw_dump_spur_1', 'dump_zone_22'],
    ['sw_hub', 'sw_dump_spur_2', 'dump_zone_23'],
    ['sw_hub', 'sw_dump_spur_3', 'dump_zone_24'],
    ['sw_hub', 'sw_load_spur_1', 'load_zone_22'],
    ['sw_hub', 'sw_load_spur_2', 'load_zone_23'],
    ['sw_connector_1', 'sw_haul_3'],
    ['sw_connector_2', 'sw_haul_3'],
    ('s_connector_2', 'sw_hub'),
    ('e_grid_1_a', 'ne_haul_1'),
    ('ne_haul_1', 'ne_haul_2'),
    ('ne_haul_2', 'ne_hub'),
    ('ne_hub', 'ne_q_1_a'),
    ('ne_q_1_a', 'ne_q_1_b'),
    ('ne_q_1_b', 'ne_q_1_c'),
    ('ne_q_1_c', 'ne_q_1_d'),
    ('ne_q_1_d', 'ne_q_1_e'),
    ('ne_q_1_e', 'ne_q_1_f'),
    ('ne_q_1_f', 'ne_hub'),
    ('ne_q_1_b', 'ne_load_spur_1', 'load_zone_24'),
    ('ne_q_1_c', 'ne_load_spur_2', 'load_zone_25'),
    ('ne_q_1_e', 'ne_load_spur_3', 'load_zone_26'),
    ('ne_q_1_a', 'ne_q_2_a'),
    ('ne_q_2_a', 'ne_q_2_b'),
    ('ne_q_2_b', 'ne_q_2_c'),
    ('ne_q_2_c', 'ne_q_2_d'),
    ('ne_q_2_d', 'ne_q_2_e'),
    ('ne_q_2_e', 'ne_q_2_f'),
    ('ne_q_2_f', 'ne_q_1_f'),
    ('ne_q_2_b', 'ne_load_spur_4', 'load_zone_27'),
    ('ne_q_2_c', 'ne_load_spur_5', 'load_zone_28'),
    ('ne_q_2_e', 'ne_load_spur_6', 'load_zone_29'),
    ('e_grid_4_a', 'service_haul_1'),
    ('service_haul_1', 'service_haul_2'),
    ('service_haul_2', 'service_hub'),
    ('service_hub', 'service_loop_1'),
    ('service_loop_1', 'service_loop_2'),
    ('service_loop_2', 'service_loop_3'),
    ('service_hub', 'service_loop_6'),
    ('service_loop_6', 'service_loop_5'),
    ('service_loop_5', 'service_loop_4'),
    ('service_loop_3', 'service_loop_4'),
    ('service_loop_2', 'parking_1'),
    ('service_loop_2', 'parking_2'),
    ('service_loop_5', 'fuel_1'),
    ('service_loop_5', 'fuel_2'),
    ['connector_2_2', 'ix_east_2'],
]


# ============================================================
# 🧠 STEP 4: AUTO-DETECT PIT CENTERS
# ============================================================

def compute_pit_centers(nodes):
    pit_groups = {
        "fw": [],
        "n": [],
        "ne": [],
        "s": []
    }

    for name, (x, y, _) in nodes.items():
        if "fw_pit" in name:
            pit_groups["fw"].append((x, y))
        elif "n_q" in name:
            pit_groups["n"].append((x, y))
        elif "ne_q" in name:
            pit_groups["ne"].append((x, y))
        elif "s_sp" in name:
            pit_groups["s"].append((x, y))

    centers = []
    for group in pit_groups.values():
        if group:
            cx = np.mean([p[0] for p in group])
            cy = np.mean([p[1] for p in group])
            centers.append((cx, cy))

    return centers


PIT_CENTERS = compute_pit_centers(NODES)


# ============================================================
# 🧠 STEP 5: HEIGHT FUNCTION (MULTI-PIT)
# ============================================================

def get_z(x, y):
    # Base terrain
    base = (
        0.02 * x +
        0.015 * y +
        20 * np.sin(x / 200) +
        15 * np.cos(y / 150)
    )

    # Apply multiple pits
    for cx, cy in PIT_CENTERS:
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)

        if dist < 220:
            base -= (220 - dist) * 1.3   # pit depth

    return base


# ============================================================
# 🧠 STEP 6: APPLY HEIGHT TO ALL NODES
# ============================================================

for key in NODES:
    x, y, _ = NODES[key]
    NODES[key] = np.array([x, y, get_z(x, y)])


# ============================================================
# 🧠 STEP 7: FIX EDGES (IMPORTANT)
# ============================================================

clean_edges = []

for e in EDGES:
    if len(e) == 2:
        clean_edges.append(e)
    else:
        # Convert ('a','b','c') → (a,b),(b,c)
        for i in range(len(e) - 1):
            clean_edges.append((e[i], e[i+1]))

EDGES = clean_edges


# ============================================================
# 🧠 STEP 8: SMOOTH HEIGHT (REALISTIC ROADS)
# ============================================================

def smooth_z(nodes, edges, iterations=7):
    for _ in range(iterations):
        new_nodes = {}

        for node in nodes:
            neighbors = [v for u, v in edges if u == node] + \
                        [u for u, v in edges if v == node]

            if not neighbors:
                new_nodes[node] = nodes[node]
                continue

            avg_z = np.mean([nodes[n][2] for n in neighbors if n in nodes])

            x, y, z = nodes[node]
            new_z = 0.7 * z + 0.3 * avg_z

            new_nodes[node] = np.array([x, y, new_z])

        nodes.update(new_nodes)


smooth_z(NODES, EDGES)


# ============================================================
# 📊 STEP 9: 3D VISUALIZATION
# ============================================================

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')


# ---- DRAW EDGES (ROADS) ----
for u, v in EDGES:
    if u in NODES and v in NODES:
        ax.plot(
            [NODES[u][0], NODES[v][0]],
            [NODES[u][1], NODES[v][1]],
            [NODES[u][2], NODES[v][2]],
            linewidth=1
        )


# ---- DRAW NODES ----
for name, (x, y, z) in NODES.items():

    if name in LOAD_ZONES:
        ax.scatter(x, y, z, s=25)
    elif name in DUMP_ZONES:
        ax.scatter(x, y, z, s=25)
    elif name in FUEL_ZONES:
        ax.scatter(x, y, z, s=40)
    else:
        ax.scatter(x, y, z, s=10)


# ---- OPTIONAL SMOOTH CHAINS ----
for chain in VISUAL_ROAD_CHAINS:
    xs, ys, zs = [], [], []

    for node in chain:
        if node in NODES:
            xs.append(NODES[node][0])
            ys.append(NODES[node][1])
            zs.append(NODES[node][2])

    if len(xs) > 1:
        ax.plot(xs, ys, zs, linewidth=2)


# ============================================================
# 🎯 FINAL SETTINGS
# ============================================================

ax.set_title("3D Coal Mine Map")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Elevation")

ax.set_xlim(-650, 850)
ax.set_ylim(-450, 900)
ax.set_zlim(-300, 300)

ax.view_init(elev=30, azim=120)

plt.tight_layout()
plt.show()