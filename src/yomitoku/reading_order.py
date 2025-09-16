import cv2

from .utils.graph import Node
from .utils.misc import (
    is_intersected_vertical,
    is_intersected_horizontal,
)


def is_locked_node(node):
    return all([child.is_locked for child in node.children])


def _priority_dfs(nodes, direction):
    if len(nodes) == 0:
        return []

    pending_nodes = sorted(nodes, key=lambda x: x.prop["distance"])
    visited = [False] * len(nodes)
    start = pending_nodes.pop(0)
    stack = [start]

    order = []
    open_list = []

    while not all(visited):
        while stack:
            is_updated = False
            current = stack.pop()
            if not visited[current.id]:
                parents = current.parents
                if all([visited[parent.id] for parent in parents]) or len(parents) == 0:
                    visited[current.id] = True
                    order.append(current.id)
                    is_updated = True
                else:
                    if current not in open_list:
                        open_list.append(current)

            if is_updated:
                for open_node in reversed(open_list):
                    stack.append(open_node)
                    open_list.remove(open_node)

            if len(current.children) > 0:
                stack.append(current)

            if len(current.children) == 0:
                children = []
                for node in stack:
                    if current in node.parents:
                        children.append(node)
                        stack.remove(node)

                if direction in "top2bottom":
                    children = sorted(
                        children, key=lambda x: x.prop["box"][0], reverse=True
                    )
                elif direction in ["right2left", "left2right"]:
                    children = sorted(
                        children, key=lambda x: x.prop["box"][1], reverse=True
                    )

                stack.extend(children)
                continue

            child = current.children.pop(0)
            stack.append(child)

        for node in pending_nodes:
            if node in open_list:
                continue
            stack.append(node)
            pending_nodes.remove(node)
            break
        else:
            if not all(visited) and len(open_list) != 0:
                node = open_list.pop(0)
                visited[node.id] = True
                order.append(node.id)

    return order


def _exist_other_node_between_vertical(node, other_node, nodes):
    for search_node in nodes:
        if search_node == node or search_node == other_node:
            continue

        _, sy1, _, sy2 = search_node.prop["box"]
        _, oy1, _, oy2 = other_node.prop["box"]
        _, ny1, _, ny2 = node.prop["box"]

        if is_intersected_vertical(search_node.prop["box"], node.prop["box"]):
            if ny2 < sy1 < oy1 and ny2 < sy2 < oy1:
                return True

            if oy2 < sy1 < ny1 and oy2 < sy2 < ny1:
                return True

    return False


def _exist_other_node_between_horizontal(node, other_node, nodes):
    for search_node in nodes:
        if search_node == node or search_node == other_node:
            continue

        sx1, _, sx2, _ = search_node.prop["box"]
        ox1, _, ox2, _ = other_node.prop["box"]
        nx1, _, nx2, _ = node.prop["box"]

        if is_intersected_horizontal(search_node.prop["box"], node.prop["box"]):
            if nx2 < sx1 < ox1 and nx2 < sx2 < ox1:
                return True

            if ox2 < sx1 < nx1 and ox2 < sx2 < nx1:
                return True

    return False


def _create_graph_top2bottom(nodes):
    for i, node in enumerate(nodes):
        for j, other_node in enumerate(nodes):
            if i == j:
                continue

            if is_intersected_vertical(node.prop["box"], other_node.prop["box"]):
                ty = node.prop["box"][1]
                oy = other_node.prop["box"][1]

                if _exist_other_node_between_vertical(node, other_node, nodes):
                    continue

                if ty < oy:
                    node.add_link(other_node)
                else:
                    other_node.add_link(node)

            node_distance = node.prop["box"][0] + node.prop["box"][1]
            node.prop["distance"] = node_distance

    for node in nodes:
        node.children = sorted(node.children, key=lambda x: x.prop["box"][0])


def _create_graph_right2left(nodes):
    max_x = max([node.prop["box"][2] for node in nodes])

    for i, node in enumerate(nodes):
        for j, other_node in enumerate(nodes):
            if i == j:
                continue

            if is_intersected_horizontal(node.prop["box"], other_node.prop["box"]):
                tx = node.prop["box"][2]
                ox = other_node.prop["box"][2]

                if _exist_other_node_between_horizontal(node, other_node, nodes):
                    continue

                if tx < ox:
                    other_node.add_link(node)
                else:
                    node.add_link(other_node)

            node.prop["distance"] = (max_x - node.prop["box"][2]) + node.prop["box"][1]

    for node in nodes:
        node.children = sorted(node.children, key=lambda x: x.prop["box"][1])


def _create_graph_left2right(nodes, x_weight=1, y_weight=5):
    for i, node in enumerate(nodes):
        for j, other_node in enumerate(nodes):
            if i == j:
                continue

            if is_intersected_horizontal(node.prop["box"], other_node.prop["box"]):
                tx = node.prop["box"][2]
                ox = other_node.prop["box"][2]

                if _exist_other_node_between_horizontal(node, other_node, nodes):
                    continue

                if ox < tx:
                    other_node.add_link(node)
                else:
                    node.add_link(other_node)

            node_distance = (
                node.prop["box"][0] * x_weight + node.prop["box"][1] * y_weight
            )
            node.prop["distance"] = node_distance

    for node in nodes:
        node.children = sorted(node.children, key=lambda x: x.prop["box"][1])


def prediction_reading_order(elements, direction, img=None):
    if len(elements) < 2:
        return elements

    nodes = [Node(i, element.dict()) for i, element in enumerate(elements)]
    if direction == "top2bottom":
        _create_graph_top2bottom(nodes)
    elif direction == "right2left":
        _create_graph_right2left(nodes)
    elif direction == "left2right":
        _create_graph_left2right(nodes)
    else:
        raise ValueError(f"Invalid direction: {direction}")

    # For debugging
    # if img is not None:
    #    visualize_graph(img, nodes)

    order = _priority_dfs(nodes, direction)
    for i, index in enumerate(order):
        elements[index].order = i

    return elements


def visualize_graph(img, nodes):
    out = img.copy()
    for node in nodes:
        for child in node.children:
            nx1, ny1, nx2, ny2 = node.prop["box"]
            cx1, cy1, cx2, cy2 = child.prop["box"]

            node_center = nx1 + (nx2 - nx1) // 2, ny1 + (ny2 - ny1) // 2
            child_center = cx1 + (cx2 - cx1) // 2, cy1 + (cy2 - cy1) // 2

            cv2.arrowedLine(
                out,
                (int(node_center[0]), int(node_center[1])),
                (int(child_center[0]), int(child_center[1])),
                (0, 0, 255),
                2,
            )

    cv2.imwrite("graph.jpg", out)


def prediction_reading_order_simple(elements, direction="left2right"):
    if len(elements) < 2:
        return elements

    # Create a list of tuples (element, original_index)
    indexed_elements = list(enumerate(elements))

    # Sort elements by vertical position first
    indexed_elements.sort(key=lambda item: item[1].box[1])

    lines = []
    if len(indexed_elements) > 0:
        current_line = [indexed_elements[0]]
        for item in indexed_elements[1:]:
            element = item[1]
            # Check if the vertical center of the new element is close to the vertical center of the current line
            line_y_center = sum(e.box[1] + e.box[3] for _, e in current_line) / (
                2 * len(current_line)
            )
            element_y_center = (element.box[1] + element.box[3]) / 2
            element_height = element.box[3] - element.box[1]

            if abs(line_y_center - element_y_center) < element_height / 2:
                current_line.append(item)
            else:
                lines.append(current_line)
                current_line = [item]
        lines.append(current_line)

    # Sort lines by their average vertical position
    lines.sort(key=lambda line: sum(e.box[1] for _, e in line) / len(line))

    # Sort elements within each line
    for line in lines:
        if direction == "left2right":
            line.sort(key=lambda item: item[1].box[0])
        elif direction == "right2left":
            line.sort(key=lambda item: item[1].box[0], reverse=True)
        elif direction == "top2bottom":
            # For vertical text, sort columns by x-coordinate
            line.sort(key=lambda item: item[1].box[0])

    # Assign order
    order_counter = 0
    for line in lines:
        for original_index, element in line:
            elements[original_index].order = order_counter
            order_counter += 1

    return elements
