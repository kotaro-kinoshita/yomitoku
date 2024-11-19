class Node:
    def __init__(self, id, prop):
        self.id = id
        self.prop = prop
        self.parents = []
        self.children = []

    def add_link(self, node):
        if node in self.children:
            return

        self.children.append(node)
        node.parents.append(self)
