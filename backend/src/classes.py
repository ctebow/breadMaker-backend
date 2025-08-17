"""
Circuit classes for breadMaker.
"""
from paddleocr import PaddleOCR
from .lines import point_close_to_box, results_to_boxes, run_lines_algorithm, run_yolo
from .format import format_classes, bfs_layout_v4

"""
Ok I think I want to move away from the idea of using classes in python -->
Simpler to generate a netlist just using a connections dict.
I think the key to adding box to box recognition as well as node merging is to
just keep it simple and do those both in the step where I make the connections
dict in the first place. 
"""

### MERGE OVERLAPPING BOUNDING BOXES --> THIS IS MOST URGENT YOU ARE MISSING CONNECTIONS
### I think I am also trimming a tad too aggressively, better to have uncessary lines that can be ruled out later 
## versus some stuff missing. I should investigate this. 

### TODO: Add a merge nodes function if there's repetitive stuff. --> If im just conecting things, its safe to always merge two nodes that are directly connected. 
### TODO: Figure out consistency for the class definitions, how to get to netlist.
### TODO: Add check for boxes that are overlapping and thus have no connecting line, need to implement in function that places lines. 
### TODO: Add error propogation for entire logic flow, where when somethign goes wrong I can either pivot or display appropriate message. 
# -- Figure out how much I wanna do for this, cause idk if i'm gonna keep this as backend. 
### TODO: Add case where for connecting boxes to lines, if both ends of line touch same box, get rid of line. 
### TODO: Keep track of case where there is a junction to junction connection, probably should just treat them as components. 
### TODO: Add in all the text recog features and integrate with JSON. 

"""
Pre-processing: Get text to each text object
"""

def boxes_overlap(box1, box2, threshold=0.3):
    """
    Return True if the two boxes overlap significantly.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # Intersection
    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Areas
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return False

    iou = inter_area / union_area
    return iou > threshold


def filter_class(results, class_name: str):
    """
    Filter out a particular class from results.

    results IS A YOLO RESULTS OBJECT, USE YOLO METHODS TO ISOLATE CLASSES.
    Returns list of boxes.
    """
    assert len(results) == 1, "0 or more than 1 image inputted, assumes singular image for detection run."
    r = results[0] 
    target_cls_id = [k for k, v in r.names.items() if v == class_name][0]
    filtered = [box for box in r.boxes if int(box.cls[0]) == target_cls_id]
    
    return filtered

def text_box_to_str(yolo_results, paddle_results):
    """
    Link text box classes to the text inside them.
    """
    yolo_text = filter_class(yolo_results, 'text')
    paddle_processed = []
    for res in paddle_results:
        for txt, box in zip(res['rec_texts'], res['rec_boxes']):
            box = [int(num) for num in box]
            paddle_processed.append((txt, box))
    result = []
    for text, small_box in paddle_processed:
        for box in yolo_text:
            big_box = box.xyxy[0].tolist()
            if boxes_overlap(big_box, small_box):
                result.append((box, text))
    return result

def run_paddle_test(image_path) -> list[tuple[str, list]]:
    """
    Test tesseract to see if it gives me good bounding boxes. 
    """
    ocr = PaddleOCR(use_textline_orientation=True, lang='en')
    results = ocr.predict(image_path)
    output = []
    for res in results:
        for txt, box in zip(res['rec_texts'], res['rec_boxes']):
            box = [int(num) for num in box]
            output.append((txt, box))
    return output


"""
Associate nodes with components, get text out of results object.
"""

def process_boxes(yolo_results) -> tuple[dict[dict[str, str]], list, int]:
    """
    Get boxes ready to be connected, also filter out text boxes.

    Structure:
    class_name, component_id (unique identifier), coords
    """

    boxes = results_to_boxes(yolo_results)
    names = yolo_results[0].names
    
    text = []
    components = {}
    labeling_dict = {}

    for box in boxes:
        class_id = int(box.cls[0])
        class_name = names[class_id]
        if class_name == 'text':
            text.append(box)
        else:
            # labeling logic
            if class_name not in labeling_dict:
                labeling_dict[class_name] = 0
            else:
                labeling_dict[class_name] += 1
            component_id = f'{class_name}_{labeling_dict[class_name]}'
            coords = tuple(box.xyxy.tolist()[0])
            components[component_id] = {"name": class_name, "id": component_id, "coords": coords}
    
    junctions = [comp for comp in components.values() if comp["name"] == "junction"]
    junction_count = len(junctions)

    return (components, text, junction_count + 1)

def connect_components(processed_classes, processed_lines, distance_threshold):
    """
    Use lines connecting components to make a connections dictionary.
    """
    connections = {}
    connecting_lines = {}

    for line in processed_lines:
        x1, y1, x2, y2 = line
        connects = []
        for class_id, item in processed_classes.items():
            coords = item["coords"]
            if point_close_to_box([x1, y1], coords, distance_threshold):
                connects.append(class_id)
                if class_id not in connecting_lines:
                    connecting_lines[class_id] = [line]
                else:
                    connecting_lines[class_id].append(line)
            elif point_close_to_box([x2, y2], coords, distance_threshold):
                connects.append(class_id)  
                if class_id not in connecting_lines:
                    connecting_lines[class_id] = [line]
                else:
                    connecting_lines[class_id].append(line)
        if len(connects) == 2:
            id_1, id_2 = connects
            if id_1 == id_2:
                continue
            if id_1 not in connections:
                connections[id_1] = {id_2}
            else:
                connections[id_1].add(id_2)
            if id_2 not in connections:
                connections[id_2] = {id_1}
            else:
                connections[id_2].add(id_1)
    return connections, connecting_lines

def connections_to_nodes(connections_dict: dict, junction_count):
    """
    Using a connections dict, filter and add nodes for component-component
    connections to assemble a component: nodes dict. 
    """
    # if create node between two, make sure to add the node to both in one step
    seen_pairs = set()
    result = {}
    for component, connections in connections_dict.items():
        # filter out junctions
        if component[0] == "junction":
            continue
        # initialize component in result
        if component not in result:
            result[component] = []
        # add nodes, if its to another component add a node.
        for connect in connections:
            if connect[0] == 'junction':
                result[component].append(connect)
            else:
                pair_key = tuple(sorted([component[1], connect[1]]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    new_node = ('junction', f'junction_{junction_count}', (-1, -1, -1, -1)) # filler until I create coords
                    junction_count += 1
                    result[component].append(new_node)
                    if connect not in result:
                        result[connect] = [new_node]
                    else:
                        result[connect].append(new_node)
                else:
                    pass
    return result




"""
Testing flow functions.
"""

def test_node_sorting(path_to_image) -> dict:
    """
    Test control flow so far. 
    """

    raw_yolo_results, lines = run_lines_algorithm('training/data/bestv2_june26.pt', path_to_image)
    components, text, junction_count = process_boxes(raw_yolo_results)
    connections_dict, connecting_lines = connect_components(components, lines, 3)
    #nodes_dict = connections_to_nodes(connections_dict, junction_count)

    return connections_dict

def run_algo(image):
    raw_yolo, lines, time = run_yolo("backend/training/data/bestv2_june26.pt", image)
    components, text, junction_count = process_boxes(raw_yolo)
    connections, connecting_lines_raw = connect_components(components, lines, 3)
    component_ids_raw = format_classes(components, connecting_lines_raw)
    component_ids, connecting_wires, _ = bfs_layout_v4(component_ids_raw, connecting_lines_raw)
    return {"componentIds": component_ids, "connections": connections, "lines": connecting_wires, "text": text, "time": time}

if __name__ == "__main__":
    print(test_node_sorting("training/test_images/test6.jpg"))

