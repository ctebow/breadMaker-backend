"""
Circuit classes for breadMaker.
"""

from typing import Optional
from ultralytics import YOLO, settings
from PIL import Image
from collections import defaultdict

# also just update based on the YAML with class identifiers
IDS = {"R": "resistor", "L": "inductor", } # finish later


"""
Methods and functions I want for the classes:

***Functions***
-- Link to nodes
    - Links a component to closest/most probable nodes
    - Hough/Contour lines
-- Assert all components used
    - Checks if node list and everything is connected up
-- Make netlist
-- Separate Nodes
    - Separate and give each node an ID

***Circuit***

-- Print Components
-- Print Nodes
-- Add component
    - Just takes a component and places it inbetween two nodes
    - Node information IN component
-- Get netlist
    - Builds ltspice like netlist list 

"""

### TODO: Just make component name V1... ditch id altogether

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

    def __init__(self, name: str, coords: list[int,], value: str, nodes: list[Node, Node]):
        
        self.name = name # name of component
        self.coords = coords
        self.value = value
        self.nodes = nodes

    def __eq__(self, value: "Component"):
        if self.name == value.name:
            return True
        return False

    def __repr__(self):
        
       return f'{self.name}'
    
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
        
def separate_junctions(junction_list):
    raise NotImplementedError

def associate_with_junctions(component_list, junction_list):
    raise NotImplementedError

def package_to_components(sorted_components):
    raise NotImplementedError

def run_yolo(image):
    raise NotImplementedError

def make_circuit(components_list):
    raise NotImplementedError

def merge_nodes(sorted_components):
    """
    Determine if repeat nodes exist and remove them.
    """
    raise NotImplementedError


n0 = Node([0, 0, 0, 0], 0)
n1 = Node([1, 1, 1, 1], 1)
n2 = Node([2, 2, 2, 2], 2)
n3 = Node([3, 3, 3, 3], 3)

r1 = Component("R1", [0, 0, 0, 0], "5", [n0, n1])
v1 = Component("V1", [1, 1, 1, 1], "5", [n1, n2])
l1 = Component("L1", [2, 2, 2, 2], "5", [n2, n3])
c1 = Component("C1", [3, 3, 3, 3], "5", [n3, n0])

model = YOLO("training/data/bestv2_june26.pt")
results = model.predict(source=0, show=True)


