"""
Helper functions to clean up yolo results. 
"""
from collections import deque
import math
import copy

DIMENSIONS_MAP = {
    "resistor": {"x": 100, "y": 40, "invert": False},
    "capacitor": {"x": 40, "y": 40, "invert": True},
    "inductor": {"x": 100, "y": 40, "invert": False},
    "voltage-dc": {"x": 40, "y": 40, "invert": True},
    "voltage-ac": {"x": 100, "y": 40, "invert": False},
    "diode": {"x": 100, "y": 40, "invert": False},
    "switch": {"x": 100, "y": 40, "invert": False},
    "wire": {"x": 0, "y": 0, "invert": False},
    "varistor": {"x": 100, "y": 40, "invert": False},
    "fuse": {"x": 100, "y": 40, "invert": False},
    "motor": {"x": 100, "y": 40, "invert": False},
    "diode-zenor": {"x": 100, "y": 40, "invert": False},
    "capacitor-polarized": {"x": 40, "y": 40, "invert": True},
    "current_source": {"x": 100, "y": 40, "invert": False}
}

SPECIAL_TYPES = { 
    "crossover": {"x": 40, "y": 40, "snapPoints": [{"dx": -20, "dy": 0}, {"dx": 0, "dy": 20}, {"dx": 0, "dy": -20}, {"dx": 20, "dy": 0}]},
    "terminal-neg": {"x": 40, "y": 40, "snapPoints": [{"dx": 0, "dy": -20}]},
    "terminal-pos": {"x": 40, "y": 40, "snapPoints": [{"dx": 0, "dy": 20}]},
    "thyristor": {"x": 100, "y": 40, "snapPoints": [{"dx": -50, "dy": 0}, {"dx": 50, "dy": 0}, {"dx": 30, "dy": 20}]},
    "not": {"x": 80, "y": 40, "snapPoints": [{"dx": -40, "dy": 0}, {"dx": 40, "dy": 0}]},
    "or": {"x": 80, "y": 40, "snapPoints": [{"dx": -40, "dy": -10}, {"dx": -40, "dy": 10}, {"dx": 40, "dy": 0}]},
    "nor": {"x": 80, "y": 40, "snapPoints": [{"dx": -40, "dy": -10}, {"dx": -40, "dy": 10}, {"dx": 40, "dy": 0}]},
    "xor": {"x": 80, "y": 40, "snapPoints": [{"dx": -40, "dy": -10}, {"dx": -40, "dy": 10}, {"dx": 40, "dy": 0}]},
    "nand": {"x": 80, "y": 40, "snapPoints": [{"dx": -40, "dy": -10}, {"dx": -40, "dy": 10}, {"dx": 40, "dy": 0}]},
    "and": {"x": 80, "y": 40, "snapPoints": [{"dx": -40, "dy": -10}, {"dx": -40, "dy": 10}, {"dx": 40, "dy": 0}]},
    "opAmp": {"x": 100, "y": 100, "snapPoints": [{"dx": -50, "dy": -30}, {"dx": -50, "dy": 30}, {"dx": 0, "dy": -50}, {"dx": 0, "dy": 50}, {"dx": 50, "dy": 0}]},
    "resistor-photo": {"x": 100, "y": 100, "snapPoints": [{"dx": -50, "dy": 0}, {"dx": 50, "dy": 0}]},
    "transistor": {"x": 80, "y": 100, "snapPoints": [{"dx": -40, "dy": 0}, {"dx": 20, "dy": -50}, {"dx": 20, "dy": 50}]},
    "microphone": {"x": 80, "y": 100, "snapPoints": [{"dx": 0, "dy": -50}, {"dx": 0, "dy": 50}]},
    "transistor-photo": {"x": 80, "y": 100, "snapPoints": [{"dx": 20, "dy": -50}, {"dx": 20, "dy": 50}]},
    "transistor-PNP": {"x": 80, "y": 100, "snapPoints": [{"dx": -40, "dy": 0}, {"dx": 20, "dy": -50}, {"dx": 20, "dy": 50}]},
    "speaker": {"x": 80, "y": 100, "snapPoints": [{"dx": -10, "dy": -50}, {"dx": -10, "dy": 50}]},
    "diode-light_emitting": {"x": 100, "y": 80, "snapPoints": [{"dx": -50, "dy": 0}, {"dx": 50, "dy": 0}]},
    "transformer": {"x": 80, "y": 100, "snapPoints": [{"dx": -40, "dy": -40}, {"dx": -40, "dy": 40}, {"dx": 40, "dy": -40}, {"dx": 40, "dy": 40}]},
    "triac": {"x": 100, "y": 100, "snapPoints": [{"dx": 0, "dy": -50}, {"dx": 0, "dy": 50}, {"dx": 50, "dy": 40}]},
    "diac": {"x": 100, "y": 100, "snapPoints": [{"dx": 0, "dy": -50}, {"dx": 0, "dy": 50}]},
    "ground": {"x": 40, "y": 40, "snapPoints": [{"dx": 0, "dy": -20}]},
}

NAME_MAPPING = {
    'and': "and",
    'Capacitor': "capacitor", 
    'Crossover': "crossover", 
    'Diode': "diode",
    'Junction': "node",
    'NOT': "not",
    'Not': "not",
    'OR': "or",
    'Resistor': "and",
    'Text': "text",
    'Transistor': "transistor",
    'Voltage-AC': "voltage-ac",
    'Voltage-ac': "voltage-ac",
    'Zenor Diode': "diode-zenor",
    'and': "and",
    'antenna': None,
    'capacitor': "capacitor",
    'capacitor-polarized': "capacitor-polarized",
    'capacitor-unpolarized': "capacitor",
    'crossover': "crossover",
    'current': "current_source",
    'diac': "diac",
    'diode': "diode",
    'diode-light_emitting': "diode-light_emitting",
    'fuse': "fuse",
    'gnd': "ground",
    'inductor': "inductor",
    'integrated_circuit': None,
    'integrated_cricuit-ne555': None,
    'junction': "node",
    'junctions': "node",
    'lamp': "null",
    'microphone': "microphone",
    'motor': "motor",
    'nand': "nand",
    'nor': "nor",
    'not': "not",
    'operational_amplifier': "opAmp",
    'optocoupler': None,
    'or': "or",
    'probe-current': None,
    'relay': None,
    'resistor': "resistor",
    'resistor-adjustable': "varistor",
    'resistor-photo': "resistor-photo",
    'schmitt_trigger': None,
    'socket': None, 
    'speaker': "speaker",
    'switch': "switch",
    'terminal': "terminal-pos",
    'text': "text", 
    'thyristor': "thyristor",
    'transformer': "transformer",
    'transistor': 'transistor',
    'transistor-npn': "transistor",
    'transistor-photo': "transistor-photo",
    'triac': "triac",
    'varistor': "varistor", 
    'voltage ac': "voltage-ac", 
    'voltage dc': "voltage-dc", 
    'voltage-ac': "voltage-ac", 
    'voltage-dc': "voltage-dc",
    'voltage-dc_ac': "voltage-ac", 
    'voltage-dc_regulator': "voltage-dc", 
    'voltage_ac': "voltage-ac", 
    'voltage_dc': "voltage-ac", 
    'vss': "terminal-pos", 
    'xor': "xor", 
    'zener': "diode-zenor"
}

"""
Flow to get component rotation. 
"""

def bbox_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def rotate_point(px, py, angle_deg):
    """Rotate point (px, py) by angle around origin (0,0)."""
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return px * cos_t - py * sin_t, px * sin_t + py * cos_t

def get_snap_points_for_type(component_type):
    """Return canonical snap points for a component type (relative to center)."""
    if component_type in SPECIAL_TYPES:
        base = SPECIAL_TYPES[component_type]
        return [(sp["dx"], sp["dy"]) for sp in base["snapPoints"]]
    elif component_type in DIMENSIONS_MAP:
        base = DIMENSIONS_MAP[component_type]
        half_x, half_y = base["x"] / 2, base["y"] / 2
        if base["invert"]:
            return [(0, -half_y), (0, half_y)]
        else:
            return [(-half_x, 0), (half_x, 0)]
    else:
        return []

def infer_rotation(component_type, bbox_xyxy, detected_connection_points):
    """
    detected_connection_points: list of ("x", "y") in image space.
    bbox_xyxy: (x1, y1, x2, y2)
    """
    if not detected_connection_points or len(detected_connection_points) == 0:
        return 0

    cx, cy = bbox_center(bbox_xyxy)
    # Move detected points to be relative to center
    rel_points = [(px - cx, py - cy) for px, py in detected_connection_points]
    canonical_points = get_snap_points_for_type(component_type)

    if not rel_points or not canonical_points: return 0

    best_angle = 0
    min_error = float("inf")

    for angle in [0, 90, 180, 270]:
        rotated_canon = [rotate_point(dx, dy, angle) for dx, dy in canonical_points]
        # Match points by nearest neighbor
        total_error = 0
        for p in rel_points:
            
            nearest = min(rotated_canon, key=lambda c: math.hypot(c[0] - p[0], c[1] - p[1]))
            total_error += math.hypot(nearest[0] - p[0], nearest[1] - p[1])
        if total_error < min_error:
            min_error = total_error
            best_angle = angle

    return best_angle

def lines_to_points(lines: list[int]) -> list[tuple[int, int]]:
    """
    Get points from lines. 
    """
    if not lines:
        return []

    res = []
    for line in lines:
        x1, y1, x2, y2 = line
        res.extend([(x1, y1), (x2, y2)])
    return res

def snap_to_grid(coord, grid_size=10):
    """
    Snap component coord to grid. 
    """
    return round(coord/grid_size) * grid_size

def transform(coord, expand=1, shift=0):
    return snap_to_grid(coord) * expand + shift

def get_snap_points(xPos: int, yPos: int, type: str, rotation: int) -> list[dict[str, int]]:
    """
    Use rotation and snap point mappings to get snap points.
    """
    if type in DIMENSIONS_MAP:
        width, height = DIMENSIONS_MAP[type]["x"], DIMENSIONS_MAP[type]["y"]
        invert = DIMENSIONS_MAP[type]["invert"]
        if invert:
            angle = rotation + 90 * math.pi / 180
        else:
            angle = rotation * math.pi / 180

        dx = ((width * math.cos(angle) + height * math.sin(angle)) / 2) * math.cos(angle)
        dy = ((height * math.cos(angle) + width * math.sin(angle)) / 2) * math.sin(angle)

        return [{"x": snap_to_grid(xPos + dx), "y": snap_to_grid(yPos + dy)}, 
                {"x": snap_to_grid(xPos - dx), "y": snap_to_grid(yPos - dy)}]
    
    elif type in SPECIAL_TYPES:
        result = []
        for point in SPECIAL_TYPES[type]["snapPoints"]:
            angle = rotation * math.pi / 180
            rotated_x = point["dx"] * math.cos(angle) - point["dy"] * math.sin(angle)
            rotated_y = point["dx"] * math.sin(angle) + point["dy"] * math.cos(angle)
            result.append({"x": snap_to_grid(xPos + rotated_x), "y": snap_to_grid(yPos + rotated_y)})

        return result

    elif type == "node":
        return [{"x": snap_to_grid(xPos), "y": snap_to_grid(yPos)}]
    else:
        return []

# idk if I need this, let it rest for now. 
def normalize_lines(connecting_lines: dict, expand=2, shift = 100):
    result = {}
    for component_id, lines in connecting_lines.items():
        lines_normalized = []
        for line in lines:
            line_snapped = []
            for point in line:
                line_snapped.append(snap_to_grid(point) * expand + shift)
            lines_normalized.append({"x1": line_snapped[0], "y1": line_snapped[1], 
             "x2": line_snapped[2], "y2": line_snapped[3]})
        
            


def format_classes(yolo_components: dict, connecting_lines, expand=1, shift=0):
    """Format classes given by sorting"""
    result = {}
    for comp_id, value in yolo_components.items():
        # switch info
        name = NAME_MAPPING[value["name"]]
        x_pos = snap_to_grid(value["coords"][0]) * expand + shift
        y_pos = snap_to_grid(value["coords"][1]) * expand + shift

        # handle exceptions
        if not name:
            continue
        if comp_id in connecting_lines:
            lines = connecting_lines[comp_id]
        else:
            lines = []
        
        # points for lines function, get rotation and snap points. 
        if name != "node":
            points = lines_to_points(lines)
            rotation = infer_rotation(name, tuple(value["coords"]), points)
            snap_points = get_snap_points(x_pos, y_pos, name, rotation)
        else:
            rotation = 0
        
        result[comp_id] = {"id": comp_id, 
                           "name": value["id"], 
                           "value": 0, 
                           "rotation": rotation,
                           "type": name, 
                           "xPos": x_pos, 
                           "yPos": y_pos, 
                           "snapPoints": snap_points}
        
    return result

def midpoint(x1, y1, x2, y2):
    """Helper to get midpoint of two points."""
    return (x1 + x2) / 2, (y1 + y2) / 2

def line_orientation(x1, y1, x2, y2, tolerance=1e-6):
    """
    Determine if the line defined by points (x1, y1) and (x2, y2) is more horizontal or vertical.
    """
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    
    if abs(dx - dy) < tolerance:
        return "diagonal"  # optionally handle near 45° lines
    elif dx > dy:
        return "horizontal"
    else:
        return "vertical"
    
def get_closest_snap_point(snap_points, comp_point):
    closest = None
    dist = float("inf")
    for snap_point in snap_points:
        curr_dist = math.hypot(comp_point[0] - snap_point["x"], comp_point[1] - snap_point["y"])
        if curr_dist < dist:
            dist = curr_dist
            closest = snap_point

    return closest

def bfs_layout3(component_ids, component_wires, DIMENSIONS_MAP=DIMENSIONS_MAP, SPECIAL_TYPES=SPECIAL_TYPES,
                grid_size=10, max_iterations=1000, expand=1.5, shift=50):
    """
    BFS traversal to align components & wires.
    Components are transformed (*expand + shift) BEFORE starting.
    Wires are also transformed.
    Diagonal wires are snapped to grid but rendered diagonally.
    """

    # Deep copy so we don't mutate the original
    new_component_ids = copy.deepcopy(component_ids)
    original_positions = {cid: (comp["xPos"], comp["yPos"]) for cid, comp in component_ids.items()}

    # Transform all components
    for comp in new_component_ids.values():
        comp["xPos"] = transform(comp["xPos"], expand, shift)
        comp["yPos"] = transform(comp["yPos"], expand, shift)
        comp["snapPoints"] = get_snap_points(comp["xPos"], comp["yPos"], comp["type"], comp.get("rotation", 0))

    # Transform all wires
    transformed_wires = {}
    for cid, wires in component_wires.items():
        transformed_wires[cid] = []
        for wire in wires:
            x1, y1, x2, y2 = wire
            transformed_wires[cid].append((
                transform(x1, expand, shift),
                transform(y1, expand, shift),
                transform(x2, expand, shift),
                transform(y2, expand, shift)
            ))

    # BFS setup
    visited = set()
    queue = deque()

    # Pick a start component
    start_cid = min(new_component_ids.keys(),
                    key=lambda cid: new_component_ids[cid]["xPos"] + new_component_ids[cid]["yPos"])
    queue.append(start_cid)
    visited.add(start_cid)

    # Wire storage
    new_wires = {cid: [] for cid in new_component_ids}

    iteration_count = 0
    while queue and iteration_count < max_iterations:
        iteration_count += 1
        current_cid = queue.popleft()
        current_comp = new_component_ids[current_cid]

        # Snap component to grid
        current_comp["xPos"] = snap_to_grid(current_comp["xPos"], grid_size)
        current_comp["yPos"] = snap_to_grid(current_comp["yPos"], grid_size)
        current_comp["snapPoints"] = get_snap_points(current_comp["xPos"], current_comp["yPos"],
                                                     current_comp["type"], current_comp.get("rotation", 0))

        for wire in transformed_wires.get(current_cid, []):
            x1, y1, x2, y2 = wire

            # Determine which endpoint is closest to the current component
            dist1 = math.hypot(x1 - current_comp["xPos"], y1 - current_comp["yPos"])
            dist2 = math.hypot(x2 - current_comp["xPos"], y2 - current_comp["yPos"])
            if dist1 <= dist2:
                comp_endpoint = (x1, y1)
                other_endpoint = (x2, y2)
            else:
                comp_endpoint = (x2, y2)
                other_endpoint = (x1, y1)

            # Snap to nearest component snap point
            closest_snap = get_closest_snap_point(current_comp["snapPoints"], comp_endpoint)
            comp_endpoint = (snap_to_grid(closest_snap["x"], grid_size),
                             snap_to_grid(closest_snap["y"], grid_size))

            # Determine line orientation
            orientation = line_orientation(x1, y1, x2, y2)

            # Adjust other endpoint according to orientation
            if orientation == "vertical":
                other_endpoint = (comp_endpoint[0], snap_to_grid(other_endpoint[1], grid_size))
            elif orientation == "horizontal":
                other_endpoint = (snap_to_grid(other_endpoint[0], grid_size), comp_endpoint[1])
            else:  # diagonal
                continue
               # other_endpoint = (snap_to_grid(other_endpoint[0], grid_size),
                               #   snap_to_grid(other_endpoint[1], grid_size))

            # Store updated wire
            mid_x, mid_y = midpoint(comp_endpoint[0], comp_endpoint[1],
                                    other_endpoint[0], other_endpoint[1])
            new_wires[current_cid].append({
                "x1": comp_endpoint[0],
                "y1": comp_endpoint[1],
                "x2": other_endpoint[0],
                "y2": other_endpoint[1],
                "x3": mid_x,
                "y3": mid_y
            })

            # Find the next closest component at the other end
            closest_other_cid = None
            min_dist = float("inf")
            for cid, comp in new_component_ids.items():
                if cid == current_cid:
                    continue
                dist = math.hypot(other_endpoint[0] - comp["xPos"], other_endpoint[1] - comp["yPos"])
                if dist < min_dist and dist <= grid_size * 2:
                    min_dist = dist
                    closest_other_cid = cid

            # Move the next component to align with wire and enqueue
            if closest_other_cid and closest_other_cid not in visited:
                other_comp = new_component_ids[closest_other_cid]
                other_snap = get_closest_snap_point(other_comp["snapPoints"], other_endpoint)
                offset_x = other_endpoint[0] - other_snap["x"]
                offset_y = other_endpoint[1] - other_snap["y"]
                other_comp["xPos"] += offset_x
                other_comp["yPos"] += offset_y
                other_comp["snapPoints"] = get_snap_points(other_comp["xPos"], other_comp["yPos"],
                                                           other_comp["type"], other_comp.get("rotation", 0))
                visited.add(closest_other_cid)
                queue.append(closest_other_cid)

    # Snap any disconnected components
    for cid in new_component_ids:
        if cid not in visited:
            comp = new_component_ids[cid]
            comp["xPos"] = snap_to_grid(comp["xPos"], grid_size)
            comp["yPos"] = snap_to_grid(comp["yPos"], grid_size)
            comp["snapPoints"] = get_snap_points(comp["xPos"], comp["yPos"],
                                                 comp["type"], comp.get("rotation", 0))

    # Flatten wire list
    flat_wires = []
    for wires in new_wires.values():
        flat_wires.extend(wires)

    return new_component_ids, flat_wires, original_positions

# --- Rewritten BFS Layout Logic ---
def get_component_center(component):
    """Calculates the center of a component."""
    return component['xPos'], component['yPos']

def get_wire_midpoint(wire):
    """Calculates the midpoint of a wire."""
    return (wire[0] + wire[2]) / 2, (wire[1] + wire[3]) / 2

def distance_between_points(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def bfs_layout_v4(component_ids, component_wires, grid_size=10, expand=1.5, shift=50):
    """
    Performs a breadth-first layout of components and wires after an initial transformation.
    """
    # 1. Initialization and Transformation
    new_components = copy.deepcopy(component_ids)
    
    # Transform component positions and update their snap points
    for comp in new_components.values():
        comp['xPos'] = transform(comp['xPos'], expand, shift)
        comp['yPos'] = transform(comp['yPos'], expand, shift)
        # Recalculate snap points based on the new position
        comp['snapPoints'] = get_snap_points(
            comp['xPos'], comp['yPos'], comp['type'], comp.get('rotation', 0)
        )

    # Transform all wire coordinates
    all_wires = []
    for wire_list in component_wires.values():
        for wire in wire_list:
            transformed_wire = [transform(float(c), expand, shift) for c in wire]
            all_wires.append(tuple(transformed_wire))
    unvisited_wires = set(all_wires)

    new_wires = []
    unvisited_components = set(new_components.keys())
    queue = deque()
    
    if not new_components:
        return {}, []

    # Start with the component with the lowest y, then x position
    start_comp_id = min(new_components.keys(), key=lambda cid: (new_components[cid]['yPos'], new_components[cid]['xPos']))
    
    queue.append(start_comp_id)
    if start_comp_id in unvisited_components:
        unvisited_components.remove(start_comp_id)

    # 2. Main BFS Loop (operates on transformed coordinates)
    while queue:
        current_comp_id = queue.popleft()
        current_comp = new_components[current_comp_id]

        # Position is already transformed, but we snap it again to ensure alignment during layout
        current_comp['xPos'] = snap_to_grid(current_comp['xPos'], grid_size)
        current_comp['yPos'] = snap_to_grid(current_comp['yPos'], grid_size)

        # Find the closest unvisited wire
        comp_center = get_component_center(current_comp)
        closest_wire = None
        min_dist_to_wire = float('inf')

        for wire_tuple in unvisited_wires:
            wire_midpoint = get_wire_midpoint(wire_tuple)
            dist = distance_between_points(comp_center, wire_midpoint)
            if dist < min_dist_to_wire:
                min_dist_to_wire = dist
                closest_wire = list(wire_tuple)
        
        if not closest_wire:
            continue

        unvisited_wires.remove(tuple(closest_wire))
        
        # 3. Snap Wire to Component
        p1 = (closest_wire[0], closest_wire[1])
        p2 = (closest_wire[2], closest_wire[3])

        if distance_between_points(p1, comp_center) < distance_between_points(p2, comp_center):
            start_point, end_point = p1, p2
        else:
            start_point, end_point = p2, p1
            
        target_snap_point = get_closest_snap_point(current_comp.get('snapPoints', []), start_point)
        if not target_snap_point: continue

        snapped_start_point = (target_snap_point['x'], target_snap_point['y'])

        # 4. Adjust Wire Orientation
        orientation = line_orientation(start_point[0], start_point[1], end_point[0], end_point[1])
        dx, dy = end_point[0] - start_point[0], end_point[1] - start_point[1]

        if orientation == 'horizontal':
            snapped_end_point = (snapped_start_point[0] + dx, snapped_start_point[1])
        elif orientation == 'vertical':
            snapped_end_point = (snapped_start_point[0], snapped_start_point[1] + dy)
        else: # Diagonal
            snapped_end_point = (snapped_start_point[0] + dx, snapped_start_point[1] + dy)
            
        snapped_end_point = (snap_to_grid(snapped_end_point[0], grid_size), snap_to_grid(snapped_end_point[1], grid_size))

        new_wires.append({"x1": snapped_start_point[0], "y1": snapped_start_point[1], 
                          "x2": snapped_end_point[0], "y2": snapped_end_point[1],
                            "x3": snap_to_grid((snapped_start_point[0] + snapped_start_point[1]) / 2), 
                            "x4": snap_to_grid((snapped_end_point[0] + snapped_end_point[1]) / 2)})

        # 5. Find and Move the Next Component
        next_comp_id = None
        min_dist_to_comp = float('inf')

        for comp_id in unvisited_components:
            next_comp_candidate = new_components[comp_id]
            next_comp_center = get_component_center(next_comp_candidate)
            dist = distance_between_points(snapped_end_point, next_comp_center)
            if dist < min_dist_to_comp:
                min_dist_to_comp = dist
                next_comp_id = comp_id
        
        if not next_comp_id:
            continue
            
        next_comp = new_components[next_comp_id]
        next_snap_point = get_closest_snap_point(next_comp.get('snapPoints', []), snapped_end_point)
        if not next_snap_point: continue

        shift_x = snapped_end_point[0] - next_snap_point['x']
        shift_y = snapped_end_point[1] - next_snap_point['y']

        next_comp['xPos'] += shift_x
        next_comp['yPos'] += shift_y
        for sp in next_comp.get('snapPoints', []):
            sp['x'] += shift_x
            sp['y'] += shift_y
            
        unvisited_components.remove(next_comp_id)
        queue.append(next_comp_id)

    return new_components, new_wires, None

if __name__ == "__main__":
    print(infer_rotation("resistor", (0, 0, 40, 100), [(20, 0), (20, 100)]))