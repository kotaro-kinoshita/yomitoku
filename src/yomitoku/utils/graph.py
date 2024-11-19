class Node:
    def __init__(self, id, prop):
        self.id = id
        self.prop = prop
        self.parents = []
        self.children = []

        self.is_locked = False

    def add_link(self, node):
        if node in self.children:
            return

        self.children.append(node)
        node.parents.append(self)

    def __repr__(self):
        if "contents" in self.prop:
            return self.prop["contents"]
        return "table"
