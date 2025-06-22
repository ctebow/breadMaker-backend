"""
Circuit classes for breadMaker.
"""

class Component():

    def __init__(self):
        self.connections = []

    def series_connect(self, next_component):
        self.connections.append(next_component)


class Resistor(Component):
    
    def __init__(self, component_id, resistance):
        
        self.component_id = component_id
        self.resistance = resistance


class Inductor(Component):
    ...

class Capacitor(Component):
    ...

class Capacitor_Polarized(Component):
    ...

class DC_Voltage(Component):
    ...

class AC_Voltage(Component):
    ...

class Switch(Component):
    ...

class Junction():
    ...

class Terminal():
    ...

class Diode(Component):
    ...

class Speaker(Component):
    ...

class Gate_AND(Component):
    ...

class Gate_OR(Component):
    ...

class Gate_XOR(Component):
    ...

class Gate_NOT(Component):
    ...

class Fuse(Component):
    ...

class Gnd(Component):
    ...