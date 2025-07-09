"""
Functions to clean up hough transform.
"""

import math

def unnest_list(lst: list[list[list[int, int, int, int]]]) -> list[list[int, int, int, int]]:
    """
    Remove uncessary brackets.
    """

    return [item[0] for item in lst]


def get_angle(line: list[int, int, int, int]) -> float:
    """
    Get normalized angle of line segment.
    """
    x0, y0, x1, y1 = line
    rad = math.atan2((y1 - y0), (x1 - x0))
    deg = math.degrees(rad)

    if deg < 0:
        deg += 180
    return deg

def get_length(line: list[int, int, int, int]) -> float:
    """
    Get length of line segment.
    """

    x0, y0, x1, y1 = line
    return math.hypot((x1 - x0), (y1 - y0))

def update_with_angles(lines: list[list[list[int, int, int, int]]]) -> None:
    """
    Update each line with its normalized angle.
    """

    for line in lines:
        angle = get_angle(line[0])
        line[0].append(angle)

def point_to_segment_distance(point: list[int, int], 
                              line: list[int, int, int, int]) -> float:
    """
    Return distance from a point to that of a line segment.
    """

    px, py = point
    x0, y0, x1, y1 = line
    dx = x1 - x0
    dy = y1 - y0

    if dx == dy == 0:
        return math.hypot(px - x1, py - y1)
    
    # projection factor
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t)) # segment, not infinite line
    projx = x1 + t * dx
    projy = y1 + t * dy

    return math.hypot(px - projx, py - projy)

def segment_to_segment_distance(line1, line2) -> float:
    """
    Calculate min distance from one segment to another.
    """
    
    d1 = point_to_segment_distance(line1[:2], line2)
    d2 = point_to_segment_distance(line1[2:], line2)
    d3 = point_to_segment_distance(line2[:2], line1)
    d4 = point_to_segment_distance(line2[2:], line1)

    return min(d1, d2, d3, d4)


def check_lines_similar(line1, line2, angle_threshold, distance_threshold) -> bool:
    """
    Decide if two lines are close and have similar slope. 
    """
    # check angle
    deg1, deg2 = get_angle(line1), get_angle(line2)
    if abs(deg2 - deg1) > angle_threshold:
        return False
    
    # check closeness
    distance = segment_to_segment_distance(line1, line2)
    if distance > distance_threshold:
        return False
    
    return True

def merge_two_lines(line1, line2) -> list[int, int, int, int]:
    """
    Merge two lines into the longest possible combo.
    Conceptually, lines must be similar for this to be sensible. 
    """

    x01, y01, x11, y11 = line1
    x02, y02, x12, y12 = line2

    dist1 = get_length(line1)
    dist2 = get_length(line2)
    l3, dist3 = [x01, y01, x12, y12], get_length([x01, y01, x12, y12])
    l4, dist4 = [x02, y02, x11, y11], get_length([x02, y02, x11, y11])

    dct = {dist1: line1, dist2: line2, dist3: l3, dist4: l4}
    maxx = max(dist1, dist2, dist3, dist4)

    return dct[maxx]

def merge_lines(lines: list[list[int, int, int, int]], 
                angle_threshold: int, distance_threshold: int) -> None:
   
    lines = lines.copy()  # work on a copy

    changed = True
    while changed:
        changed = False
        merged = False

        n = len(lines)
        for i in range(n):
            for j in range(i + 1, n):
                if check_lines_similar(lines[i], lines[j], angle_threshold, distance_threshold):
                    merged_line = merge_two_lines(lines[i], lines[j])

                    # Create a new working list with merged result
                    new_lines = [line for k, line in enumerate(lines) if k not in (i, j)]
                    new_lines.append(merged_line)

                    lines = new_lines
                    changed = True
                    merged = True
                    break  # restart loop
            if merged:
                break

    return lines


### TODO: Determine if this is necessary
def cluster_points(lines: list[list[int, int, int, int]], distance_threshold) -> dict[list[int, int], list[list[int, int]]]:
    """
    Cluster points that are close to one another and keep a running average
    cluster center. 
    """

    for idx1, line1 in enumerate(lines):
        x01, y01, x11, y11 = line1

        for line2 in lines[idx1:]:
            x02, y02, x12, y12 = line2

            dist = min(math.hypot(x01 - x02, y01 - y02), 
                       math.hypot(x01 - x12, y01 - y12), 
                       math.hypot(x11 - x02, y11 - y02), 
                       math.hypot(x11 - x12, y11 - y12))
    
    raise NotImplementedError



def identify_nodes(lines, distance_threshold) -> list[list[float, float]]:
    """
    Identify nodes based on clustering line endpoints as well as definition
    of a node. 
        - Return list of node coordinates
    """

