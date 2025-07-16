"""
Circuit classes for breadMaker.
"""

from paddleocr import PaddleOCR
from collections import defaultdict
from lines import point_close_to_box, results_to_boxes, run_lines_algorithm


### TODO: Add a merge nodes function if there's repetitive stuff. --> If im just conecting things, its safe to always merge two nodes that are directly connected. 
### TODO: Figure out consistency for the class definitions, how to get to netlist.
### TODO: Add check for boxes that are overlapping and thus have no connecting line, need to implement in function that places lines. 
### TODO: Add error propogation for entire logic flow, where when somethign goes wrong I can either pivot or display appropriate message. 
# -- Figure out how much I wanna do for this, cause idk if i'm gonna keep this as backend. 
### TODO: Add case where for connecting boxes to lines, if both ends of line touch same box, get rid of line. 
### TODO: Keep track of case where there is a junction to junction connection, probably should just treat them as components. 

class Node():
    def __init__(self, coords, id):
        self.coords = coords
        self.id = id

    def __eq__(self, other: "Node"):
        if self.id == other.id: 
            return True
        return False
    
    def __repr__(self):
        return f'Node: {self.id}'

    def __hash__(self):
        return hash(self.id)

# change so value defaults to one.
class Component():
    def __init__(self, name: str, id: str, coords: list[int], value: str, nodes: list[Node, Node]):
        self.name = name # name of component --> use official yolo names.
        self.id = id
        self.coords = coords
        self.value = value
        self.nodes = nodes

    def add_value(self, value: str) -> None:
        self.value = value

    def __eq__(self, value: "Component"):
        if self.name == value.name:
            return True
        return False

    def __repr__(self):
       return f'type: {self.type}, ID: {self.name}'
    
    def __hash__(self):
        return hash(self.name)

class Circuit():
    connections: dict[Node, list[Component]]
    def __init__(self, name: str):
        self.name = name
        self.connections = {}
        
    def add_component(self, comps: Component | list[Component,]):
        """
        Add component to circuit, update components dictionary
        """
        if isinstance(comps, Component):
            nodes = comps.nodes
            for node in nodes:
                if node not in self.connections:
                    self.connections[node] = [comps]
                else:
                    self.connections[node].append(comps)
        else:
            for comp in comps:
                nodes = comp.nodes
                for node in nodes:
                    if node not in self.connections:
                        self.connections[node] = [comp]
                    else:
                        self.connections[node].append(comp)

    def __repr__(self):
        return str(self.connections)

    def components_to_netlist(self) -> list[str]:
        connects: dict[Node, list[Component]] = self.connections
        pairs: dict[Component, set[Node, Node]] = defaultdict(set)

        for node, components in connects.items():
            for comp in components:
                pairs[comp].add(node)

        result = []
        for component, nodes in pairs.items():
            
            if len(nodes) != 2:
                raise ValueError(f'{component.name} has incorrect node count')

            node1, node2 = sorted(nodes, key=lambda n: n.id)
            s = f'{component.name} {node1.id} {node2.id} {component.value}'
            result.append(s)

        return result

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

"""
Associate nodes with components, get text out of results object.
"""

def process_boxes(yolo_results):
    """
    Get boxes ready to be connected, also filter out text boxes.

    Structure:
    class_name, component_id (unique identifier), coords
    """

    boxes = results_to_boxes(yolo_results)
    names = yolo_results[0].names
    
    text = []
    rest = []

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

            rest.append((class_name, component_id, coords))
    
    junctions = [comp for comp in rest if comp[0] == "junction"]
    junction_count = len(junctions)

    return (rest, text, junction_count + 1)

def connect_components(processed_classes, processed_lines, distance_threshold):
    """
    Use lines connecting components to make a connections dictionary.
    """
    connections = {}

    for line in processed_lines:
        x1, y1, x2, y2 = line
        connects = []
        for obj in processed_classes:
            coords = obj[2]
            if (point_close_to_box([x1, y1], coords, distance_threshold) 
                or point_close_to_box([x2, y2], coords, distance_threshold)):
                connects.append(obj)  
        if len(connects) == 2:
            obja, objb = connects
            if obja not in connections:
                connections[obja] = [objb]
            else:
                connections[obja].append(objb)
            if objb not in connections:
                connections[objb] = [obja]
            else:
                connections[objb].append(obja)
    return connections

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

def merge_nodes(sorted_components):
    """
    Determine if repeat nodes exist and remove them.
    """
    raise NotImplementedError

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

def associate_text(component_list):
    """
    Use coords of text and components to place text to nearest component, use it 
    to also double check if we got the component right. 
    """
    raise NotImplementedError


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

def dct_to_circuit(component_dct, text, circuit_name):
    """
    Take processed components and put them into a circuit object. 
    """

"""
Testing flow functions.
"""

def test_node_sorting(path_to_image) -> dict:
    """
    Test control flow so far. 
    """

    raw_yolo_results, lines = run_lines_algorithm('training/data/bestv2_june26.pt', path_to_image)
    components, text, junction_count = process_boxes(raw_yolo_results)
    connections_dict = connect_components(components, lines, 3)
    nodes_dict = connections_to_nodes(connections_dict, junction_count)

    return nodes_dict


if __name__ == '__main__':
    dct = test_node_sorting('training/test_images/test7.jpg')
    for name, nodes in dct.items():
        print(name[1], ':')
        for node in nodes:
            print(node[1])
        print(" ")

