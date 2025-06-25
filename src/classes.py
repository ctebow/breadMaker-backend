"""
Circuit classes for breadMaker.
"""

from typing import Optional

IDS = {"R": "resistor", "L": "inductor", } # finish later

class NodeLvl():

    def __init__(self, n1, n2):
        self.n1 = n1
        self.n2 = n2

    def __eq__(self, other: "NodeLvl"):
        if {self.n1, self.n2} == {other.n1, other.n2}:
            return True
        return False
    
    def __repr__(self):
        print(f'{(self.n1, self.n2)}')

class Component():

    name: str
    id: str
    value: str | int # not sure yet
    pos: tuple[int, int]
    nodes: Optional[tuple[str, str]]

    def __init__(self, name: str, id: str, value: str, pos: tuple[int, int], nodes: NodeLvl):
        self.nodes = nodes
        self.name = name
        self.id = id
        self.value = value
        self.pos = pos

    def add_nodes(self, nodes: NodeLvl) -> None:
        """
        Add nodes after logic
        """
        self.nodes = nodes

    def __repr__(self):
        print(f'Type: {self.id} Name: {self.name} Value: {self.value} Nodes: {self.nodes}')

class Circuit():

    def __init__(self):

        self.connections = {}
        self.components = set()
        self.nodes = set()

    def add_component(self, component: Component) -> None:
        """
        Add component to the circuit.
        """
        self.components.add(component)
        self.nodes.add(component.nodes)

        n1, n2 = component.nodes.n1, component.nodes.n2
        if n1 not in self.connections:
            self.connections[n1] = component
        else:
            self.connections[n1].append(component)

        if n2 not in self.connections:
            self.connections[n2] = component
        else:
            self.connections[n2].append(component)

    def get_component(self, component_name: str) -> Component:
        """
        Extract component from circuit.
        """
        for comp in self.components:
            if comp.name == component_name:
                return comp
            
    def print_nodes(self) -> None:
        """
        Print the name of each node and how they are connected.
        """
        for node in self.nodes:
            print(node)