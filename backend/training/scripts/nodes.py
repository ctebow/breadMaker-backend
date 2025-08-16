"""
Functions to clean up hough transform.
"""
import cv2
import math
import matplotlib.pyplot as plt
from classes import run_yolo_test, results_to_boxes
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
    i = 0
    changed = True
    while changed or i < 1000:
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
        i += 1
    return lines

def merge_lines_one_pass(lines, angle_threshold, distance_threshold):
    """
    Merge lines based on how close and how similar they are. 
    """

    merged_flags = [False] * len(lines)
    merged_lines = []

    n = len(lines)
    for i in range(n):
        if merged_flags[i]:
            continue

        base = lines[i]
        for j in range(i + 1, n):
            if merged_flags[j]:
                continue

            if check_lines_similar(base, lines[j], angle_threshold, distance_threshold):
                base = merge_two_lines(base, lines[j])
                merged_flags[j] = True  # Don't use this line again

        merged_lines.append(base)

    return merged_lines

def point_in_box(point: list[int, int], box: list[int, int, int, int]) -> bool:
    """
    Helper for line in region. 
    """
    x, y = point
    tx, ty, bx, by = box
    return tx <= x <= bx and ty <= y <= by

def ccw(ax, ay, bx, by, cx, cy):
    """
    Helper for do_lines_intersect.
    """
    return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

def do_lines_intersect(line, region_line):
    """
    Helper for line in region
    """
    tx, ty, bx, by = region_line
    lx1, ly1, lx2, ly2 = line

    return ccw(lx1, ly1, tx, ty, bx, by) != ccw(lx2, ly2, tx, ty, bx, by) and ccw(lx1, ly1, lx2, ly2, tx, ty) != ccw(lx1, ly1, lx2, ly2, bx, by)

def line_in_region(line: list[int, int, int, int], region: list[int, int, int, int]) -> bool:
    """
    Determines if a line fully or partially lies within a bounding box region. 
    """

    tx, ty, bx, by = region # top left, bottom right
    xl1, yl1, xl2, yl2 = line

    if point_in_box([xl1, yl1], region) and point_in_box([xl2, yl2], region):
        return True
    edges = [
       [tx, ty, bx, ty], # top
       [bx, ty, bx, by], # left
       [bx, by, tx, by], # bottom
       [tx, by, tx, ty], # right
    ]

    for edge in edges:
        if do_lines_intersect(line, edge):
            return True
    return False


def lsd_detection(image_path):
    """
    Use cv2 line segment detection.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lsd = cv2.createLineSegmentDetector(0) # 0 for default parameters
    lines = lsd.detect(gray)[0]
    return lines

"""
Testing Code
"""


results = run_yolo_test('training/data/bestv2_june26.pt', 'training/training_images/test5.jpg', show=False, print=False)
boxes = results_to_boxes(results)
print(boxes)

img = cv2.imread('training/training_images/test5.jpg')
lines = unnest_list(lsd_detection('training/training_images/test5.jpg'))
for _ in range(2):
    lines = merge_lines_one_pass(lines, 5, 5)
print(len(lines))

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Detected Lines")
plt.axis("off")
plt.show()
