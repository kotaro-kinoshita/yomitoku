from .graph import Node
from .misc import (
    is_intersected_horizontal,
)


def dfs_horizontal_with_priority(nodes):
    """優先度付き深さ優先探索を行い、ルールに従いノードの順序を返す"""

    if len(nodes) == 0:
        return []

    # 未訪問のノードをdistanceの昇順でソート
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
                parents = current.links["order"].h_parent

                # 親ノードが全て訪問済みの場合にチェック
                if all([visited[parent.id] for parent in parents]) or len(parents) == 0:
                    visited[current.id] = True
                    order.append(current.id)
                    is_updated = True
                else:
                    # 一度訪問してみて、まだチェックできないノードはopen_listに追加
                    open_list.append(current)

            if is_updated:
                # 更新があった場合、open_listにあるノードを優先的にスタックに積む
                for open_node in reversed(open_list):
                    stack.append(open_node)
                    open_list.remove(open_node)

            if len(current.links["order"].h_children) > 0:
                stack.append(current)

            if len(current.links["order"].h_children) == 0:
                continue

            child = current.links["order"].h_children.pop(0)
            if child not in open_list:
                stack.append(child)

        # 未訪問のノードの中で、最もdistanceが小さいノードかつopen_listにないノードをスタックに積む
        for node in pending_nodes:
            if node in open_list:
                continue
            stack.append(node)
            pending_nodes.remove(node)
            break
        else:
            # 強制的に全てのノードを訪問済みにする
            if not all(visited) and len(pending_nodes) != 0:
                node = open_list.pop(0)
                visited[node.id] = True
                order.append(node.id)

    return order


def exist_other_node_between_horizontal(node, other_node, nodes):
    for search_node in nodes:
        if search_node == node or search_node == other_node:
            continue

        sx1, _, sx2, _ = search_node.prop["box"]
        ox1, _, ox2, _ = other_node.prop["box"]
        nx1, _, nx2, _ = node.prop["box"]

        if is_intersected_horizontal(search_node.prop["box"], node.prop["box"]):
            if nx1 < sx1 < ox1 or nx2 < sx2 < ox2:
                return True

            if ox1 < sx1 < nx1 or ox2 < sx2 < nx2:
                return True

    return False


def reading_order_vertical(elements):
    nodes = [Node(i, element) for i, element in enumerate(elements)]
    max_x = max([node.prop["box"][2] for node in nodes])

    for i, node in enumerate(nodes):
        for j, other_node in enumerate(nodes):
            if i == j:
                continue

            if is_intersected_horizontal(node.prop["box"], other_node.prop["box"]):
                tx = node.prop["box"][2]
                ox = other_node.prop["box"][2]

                if exist_other_node_between_horizontal(node, other_node, nodes):
                    continue

                if tx < ox:
                    other_node.add_h_link("order", node)

            node.prop["distance"] = (max_x - node.prop["box"][2]) * 5 + node.prop[
                "box"
            ][1]

    for node in nodes:
        node.links["order"].h_children = sorted(
            node.links["order"].h_children, key=lambda x: x.prop["box"][3]
        )

    return dfs_horizontal_with_priority(nodes)
