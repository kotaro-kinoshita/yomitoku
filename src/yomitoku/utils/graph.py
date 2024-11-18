class BiLink:
    def __init__(self):
        self.h_parent = []
        self.v_parent = []
        self.h_children = []
        self.v_children = []


class Node:
    def __init__(self, id, prop):
        self.id = id
        self.prop = prop
        self.is_merged_cell = False
        self.cell_group = []
        self.is_visited = False

        self.links = {
            "order": BiLink(),
        }

    def add_h_link(self, name, node):
        self.links[name].h_children.append(node)
        node.links[name].h_parent.append(self)

    def add_v_link(self, name, node):
        self.links[name].v_children.append(node)
        node.links[name].v_parent.append(self)

    def reset_h_link(self, name):
        for child in self.links[name].h_children:
            child.links[name].h_parent.remove(self)

        for parent in self.links[name].h_parent:
            parent.links[name].h_children.remove(self)

        self.links[name].h_children = []
        self.links[name].h_parent = []

    def reset_v_link(self, name):
        for child in self.links[name].v_children:
            child.links[name].v_parent.remove(self)

        for parent in self.links[name].v_parent:
            parent.links[name].v_children.remove(self)

        self.links[name].v_children = []
        self.links[name].v_parent = []

    def remove_h_link(self, name, node):
        if node in self.links[name].h_children:
            self.links[name].h_children.remove(node)

        if self in node.links[name].h_parent:
            node.links[name].h_parent.remove(self)

    def remove_v_link(self, name, node):
        if node in self.links[name].v_children:
            self.links[name].v_children.remove(node)

        if self in node.links[name].v_parent:
            node.links[name].v_parent.remove(self)

    def __repr__(self):
        return str(self.prop["content"])

    def root(self, name):
        node = self

        num_row = 1
        num_col = 1

        while True:
            while len(node.links[name].h_parent) > 0:
                node = node.links[name].h_parent[-1]
                num_col += 1

            while len(node.links[name].v_parent) > 0:
                node = node.links[name].v_parent[-1]
                num_row += 1

            if (
                len(node.links[name].h_parent) == 0
                and len(node.links[name].v_parent) == 0
            ):
                return node, num_row, num_col
