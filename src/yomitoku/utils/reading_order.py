from .graph import Node
from .misc import (
    is_contained,
    is_intersected_horizontal,
    is_intersected_vertical,
)
from .union_find import UnionFind


def get_element_hierarchy(elements):
    """要素の階層関係を取得する

    Args:
        elements (List[Element]): 要素のリスト

    Returns:
        List[int]: 要素の階層関係
    """
    union_find = UnionFind(len(elements))

    for i, element in enumerate(elements):
        for j, other_element in enumerate(elements):
            if i == j:
                continue

            if is_contained(element.box, other_element.box):
                union_find.union(i, j)

    return union_find


def visit_v_children(node, visited, order, stack):
    parents = node.links["order"].v_parent
    if (
        all([visited[parent.id] for parent in parents])
        and not visited[node.id]
    ):
        visited[node.id] = True
        order.append(node.id)
        stack.append(node)


def scanning_nodes(nodes, root):
    visited = [False] * len(nodes)
    order = []
    start = root
    stack = [start]
    while not all(visited):
        while stack:
            current = stack.pop()
            visited[current.id] = True
            order.append(current.id)

            for child in current.links["order"].v_children:
                visit_v_children(child, visited, order, stack)

        if not all(visited):
            not_visited = [nodes[i] for i, v in enumerate(visited) if not v]
            not_visited = sorted(not_visited, key=lambda x: x.prop["distance"])
            stack.append(not_visited[0])

    return order


def reading_order(elements):
    nodes = [Node(i, element) for i, element in enumerate(elements)]

    root = None
    min_distance = None
    for i, node in enumerate(nodes):
        for j, other_node in enumerate(nodes):
            if i == j:
                continue

            if is_intersected_horizontal(
                node.prop["box"], other_node.prop["box"]
            ):
                node.add_h_link("order", other_node)

            if is_intersected_vertical(
                node.prop["box"], other_node.prop["box"]
            ):
                node.add_v_link("order", other_node)

            node_distance = node.prop["box"][0] + node.prop["box"][1]
            node.prop["distance"] = node_distance
            if min_distance is None or node_distance < min_distance:
                root = node
                min_distance = node_distance

    return scanning_nodes(nodes, root)
