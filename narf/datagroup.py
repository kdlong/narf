class DataGroup:
    def __init__(self, name, label, members = [], is_data = False, draw = None):
        self.name = name
        self.label = label
        self.members = members
        self.is_data = is_data
        self.drawproperties = draw
