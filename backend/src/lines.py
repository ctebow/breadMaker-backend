"""
Post processing for cv2 line segment detection.

Take a list of every detected line instance, and parse it until only lines that
connect two+ detected YOLO classes remain.

Important Parameters:
    -- merge_angle: the threshold for difference in angle between two lines to
    determine if they are similar. I've found 30 deg works well
    -- merge_distance: the threshold for distance between two lines to determine
    if they are similar. I've found 10 pixels works well
    delete_distance: the threshold for length of a line as well as distance from
    a line to a bounding box for it to survive being deleted. I've found 5 
    pixels works well. 

Uses:
    -- cv2 for image processing and line segment detection
    -- time for benchmarking (~0.9s per image)
    -- shapely for easy geometry calculations
"""

import time
import cv2
import math
import matplotlib.pyplot as plt
from shapely.geometry import LineString, box as shapely_box
from typing import List, Union
from ultralytics import YOLO
import numpy as np

"""
Functions to merge Lines. 
Using merge_lines_one_pass 2-3 times is usually enough.
"""

def resize_image(file, max_size=1000, path=False):
    """
    Resize an image for optimal size for lsd
    """
    if not path:
        nparr = np.frombuffer(file, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(file)
    height, width, _ = img.shape
    scale = max_size / max(height, width)
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale)
    return img

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
    t = ((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t)) # segment, not infinite line
    projx = x0 + t * dx
    projy = y0 + t * dy

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
    diff = abs(deg2 - deg1)
    diff = min(diff, 180 - diff)
    if diff > angle_threshold:
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

"""
Functions to trim and delete lines.
"""

def flatten_lines(lines):
    flat = []
    for line in lines:
        if isinstance(line[0], list):  # Nested list
            flat.extend(line)
        else:
            flat.append(line)
    return flat


def clip_line_to_box_shapely(line: List[Union[int, float]], box: List[int]) -> Union[None, List[List[int]]]:
    """
    Clip a line using Shapely so it stays outside a given box.
    If the line is entirely inside the box, return None.
    If the line is partially inside, return clipped segment(s).
    """
    x0, y0, x1, y1 = map(float, line)
    line_geom = LineString([(x0, y0), (x1, y1)])
    box_geom = shapely_box(*box)  # x_min, y_min, x_max, y_max

    # If fully inside the box, discard it
    if box_geom.contains(line_geom):
        return None

    # Subtract the box from the line
    clipped = line_geom.difference(box_geom)

    # Return a list of lines (each as [x0, y0, x1, y1])
    if clipped.is_empty:
        return None

    clipped_lines = []

    if clipped.geom_type == 'LineString':
        coords = list(clipped.coords)
        clipped_lines.append([int(coords[0][0]), int(coords[0][1]), int(coords[1][0]), int(coords[1][1])])
    elif clipped.geom_type == 'MultiLineString':
        for segment in clipped.geoms:
            coords = list(segment.coords)
            if len(coords) == 2:
                clipped_lines.append([int(coords[0][0]), int(coords[0][1]), int(coords[1][0]), int(coords[1][1])])
    else:
        return None  # Some unexpected geometry

    return clipped_lines if clipped_lines else None

def remove_or_trim_lines_shapely(lines: List[List[int]], boxes: List[List[int]]) -> List[List[int]]:
    """
    Removes or trims lines that intersect any given box.
    Handles nested output, ensures int type consistency.
    """
    final_lines = []

    for line in lines:
        new_segments = [line]
        for box in boxes:
            updated_segments = []
            for segment in new_segments:
                result = clip_line_to_box_shapely(segment, box)
                if result is None:
                    continue  # Entire segment removed
                updated_segments.extend(result)
            new_segments = updated_segments

        final_lines.extend(new_segments)

    return final_lines

def point_close_to_box(point: List[int], box: List[int], distance_threshold: int) -> bool:
    """
    Determine if a point is close to length of box edge.
    """

    tx, ty, bx, by = box
    edges = [
        [tx, ty, bx, ty], # top
        [bx, ty, bx, by], # right
        [bx, by, tx, by], # bottom
        [tx, by, tx, ty], # left
    ]

    return any(point_to_segment_distance(point, edge) < distance_threshold for edge in edges)


def delete_bad_lines(lines: List[List[int]], boxes: List[List[int]], distance_threshold: int, length_threshold: int) -> List[List[int]]:
    """
    Delete lines that are not connecting two components, and lines that are too
    short.
    """
    result = []
    for line in lines:
        if get_length(line) < length_threshold:
            continue
        count = 0
        x1, y1, x2, y2 = line
        for box in boxes:
            if point_close_to_box([x1, y1], box, distance_threshold):
                count += 1
                break  # no need to check other boxes for this endpoint
        for box in boxes:
            if point_close_to_box([x2, y2], box, distance_threshold):
                count += 1
                break
        if count >= 2:  
            result.append(line)

    return result

"""
Functions to run lsd detection and visualize results. 
"""

def results_to_boxes(results) -> list:
    """
    Given a YOLO result object, converts it into a list of boxes.
    """
    
    assert len(results) == 1, "Process one image at a time, please"
    r = results[0]
    return [box for box in r.boxes]

def results_to_coords(results) -> list[list]:
    """
    Given a YOLO result object, convert it into a list of list of coords.
    """

    boxes = results_to_boxes(results)
    result = [box.xyxy.tolist()[0] for box in boxes]
    return result

def run_yolo_test(weights_path, image_path, show=False, verbose=False):
    """
    Display an image with yolo object recognition bounding boxes. Also returns results.
    """

    model = YOLO(weights_path)
    results = model.predict(source=image_path, conf=0.2)
    if show == True:
        for r in results:
            annotated = r.plot()
            cv2.imshow("YOLO Results", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    if verbose == True:
        r = results[0]
        for box in r.boxes:
            print("Label: ", r.names[int(box.cls[0])])
            print("Confidence: ", float(box.conf[0]))
            print("Bounding Box (x1 y1 x2 y2): ", box.xyxy[0].tolist())

    return results

def lsd_detection(img):
    """
    Use cv2 line segment detection.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lsd = cv2.createLineSegmentDetector(0) # 0 for default parameters
    lines = lsd.detect(gray)[0]
    return lines

def plot_everything(results, lines, img):
    """
    Plot boxes and lines over the image. Use matplotlib.
    """
    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    
    for i, line in enumerate(lines):
        x1, y1, x2, y2 = line
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[i % 3], 7)

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax = plt.gca()

    # Loop through detections
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        x1, y1, x2, y2 = box.tolist()
        label = results.names[int(cls)]

        # Draw rectangle
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        # Draw label
        ax.text(x1, y1 - 5, f"{label} {conf:.2f}", color='white', fontsize=10,
                bbox=dict(facecolor='red', alpha=0.5, edgecolor='none'))

    plt.axis("off")
    plt.show()

"""
Testing Code
"""

def run_lines_algorithm(weights_path, file_path, show=True, merge_angle=30, merge_distance=10, delete_distance=5,
                        length_threshold=5, times_merge=2, trim=True, merge=True, delete=True):
    """
    Run a test for what I have so far, print pertinent information. 
    """
    t1 = time.time()
    img = resize_image(file_path, path=True)
    
    results = run_yolo_test(weights_path, img, show=False, verbose=False)
    boxes = results_to_coords(results)

    
    lines_new = unnest_list(lsd_detection(img))
    print(f'Raw number of lines: {len(lines_new)}')

    if trim:
        lines_new = remove_or_trim_lines_shapely(lines_new, boxes)
        print(f'Num lines outside boxes: {len(lines_new)}')

    if merge:
        for _ in range(times_merge):
            lines_new = merge_lines_one_pass(lines_new, merge_angle, merge_distance)
        print(f'Num lines after merge: {len(lines_new)}')

    if delete:
        lines_new = delete_bad_lines(lines_new, boxes, delete_distance, length_threshold)
        print(f'Num lines after delete: {len(lines_new)}')
    
    t2 = time.time()
    print(f'Execution time: {t2 - t1:.6f}s')
    if show:
        plot_everything(results[0], lines_new, img)

    return results, lines_new


MERGE_ANGLE = 30
MERGE_DISTANCE = 10
DELETE_DISTANCE = 5
LENGTH_THRESHOLD = 5

def run_yolo(weights_path, image):
    t1 = time.time()
    img = resize_image(image)
    results = run_yolo_test(weights_path, img, show=False, verbose=False)
    boxes = results_to_coords(results)
    lines_new = unnest_list(lsd_detection(img))
    lines_new = remove_or_trim_lines_shapely(lines_new, boxes)
    for _ in range(2):
        lines_new = merge_lines_one_pass(lines_new, MERGE_ANGLE, MERGE_DISTANCE)
    lines_new = delete_bad_lines(lines_new, boxes, DELETE_DISTANCE, LENGTH_THRESHOLD)
    t2 = time.time()
    return results, lines_new, t2 - t1


