from .graph import Node
from .misc import (
    is_intersected_vertical,
)


def dfs_vertical_with_priority(nodes):
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
                parents = current.links["order"].v_parent
                # 親ノードが全て訪問済みの場合にチェック
                if all([visited[parent.id] for parent in parents]) or len(parents) == 0:
                    visited[current.id] = True
                    order.append(current.id)
                    is_updated = True
                else:
                    # 一度訪問してみて、まだチェックできないノードはopen_listに追加
                    if current not in open_list:
                        open_list.append(current)

            if is_updated:
                # 更新があった場合、open_listにあるノードをスタックに積む
                for open_node in reversed(open_list):
                    stack.append(open_node)
                    open_list.remove(open_node)

            if len(current.links["order"].v_children) > 0:
                stack.append(current)

            if len(current.links["order"].v_children) == 0:
                continue

            child = current.links["order"].v_children.pop(0)
            if child not in open_list:
                stack.append(child)

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


def exist_other_node_between_vertical(node, other_node, nodes):
    for search_node in nodes:
        if search_node == node or search_node == other_node:
            continue

        _, sy1, _, sy2 = search_node.prop["box"]
        _, oy1, _, oy2 = other_node.prop["box"]
        _, ny1, _, ny2 = node.prop["box"]

        if is_intersected_vertical(search_node.prop["box"], node.prop["box"]):
            if ny1 < sy1 < oy1 and ny2 < sy2 < oy2:
                return True

            if oy1 < sy1 < ny1 and oy2 < sy2 < ny2:
                return True

    return False


def reading_order_horizontal(elements):
    nodes = [Node(i, element) for i, element in enumerate(elements)]
    for i, node in enumerate(nodes):
        for j, other_node in enumerate(nodes):
            if i == j:
                continue

            if is_intersected_vertical(node.prop["box"], other_node.prop["box"]):
                ty = node.prop["box"][1]
                oy = other_node.prop["box"][1]

                if exist_other_node_between_vertical(node, other_node, nodes):
                    continue

                if ty < oy:
                    node.add_v_link("order", other_node)
                else:
                    other_node.add_v_link("order", node)

            node_distance = node.prop["box"][0] + node.prop["box"][1] * 5
            node.prop["distance"] = node_distance

    for node in nodes:
        node.links["order"].v_children = sorted(
            node.links["order"].v_children, key=lambda x: x.prop["box"][0]
        )

        # print([child.prop["html"] for child in node.links["order"].v_children])

    return dfs_vertical_with_priority(nodes)
