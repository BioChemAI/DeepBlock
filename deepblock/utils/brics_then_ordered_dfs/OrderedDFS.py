from typing import List, Tuple, Union

TypeEdgeList = List[Tuple[int, int, int, int]]
TypeSeq = List[Union[int, str]]


def parse_edge(edge_list: TypeEdgeList):
    edge_dict = {}
    for edge in edge_list:
        edge_dict[edge[:2]], edge_dict[edge[2:]] = edge[2:], edge[:2]
    return edge_dict


def serialize(num_child_list: List[int], edge_list: TypeEdgeList,
              start_node: int = 0, mirror_sym_node_list: List[int] = [], 
              abbr = True, silent = True) -> TypeSeq:
    """An improved depth traversal method with child node order for fragment networks"""

    edge_dict = parse_edge(edge_list)
    if not silent: print(edge_dict)
    seq = []

    def f(node: int, parent_node: int = None):
        seq.append(node)

        for child_idx in range(num_child_list[node]):
            assert (node, child_idx) in edge_dict.keys(), Exception(
                f"Missing edge {(node, child_idx)} -> ?")

        # Reduce the number of placeholders
        is_reversed = all((
            num_child_list[node] > 1, 
            node in mirror_sym_node_list, 
            parent_node == edge_dict[(node, 0)][0],
            abbr
        ))
        if not silent: print(is_reversed)

        for iter_idx in range(num_child_list[node]):
            child_idx = num_child_list[node] - 1 - iter_idx if is_reversed else iter_idx
            child_node = edge_dict[(node, child_idx)][0]
            if child_node == parent_node:
                if iter_idx != num_child_list[node] - 1 or not abbr:
                    seq.append('.')
            else:
                if child_node in seq:
                    raise Exception(
                        f"Cannot form a ring: seq: {seq}, "
                        f"{(node, child_idx)} -> {edge_dict[(node, child_idx)]}, "
                        f"iter_idx: {iter_idx}, is_reversed: {is_reversed}")
                f(child_node, node)
    f(start_node)
    return seq


def deserialize(seq: TypeSeq, num_child_list: List[int], abbr=True) -> TypeEdgeList:
    edge_list = []

    def f(cursor, child_idx: int = None, parent_node: int = None):
        node = seq[cursor]
        if node != '.':
            child_node_list = []
            while cursor < len(seq):

                if abbr and len(child_node_list) == num_child_list[node] - 1:
                    if '.' not in child_node_list and parent_node is not None:
                        child_node_list.append('.')

                if len(child_node_list) == num_child_list[node]:
                    break

                cursor, child_node = f(
                    cursor + 1, len(child_node_list), node)
                child_node_list.append(child_node)

            if parent_node is not None:
                assert child_node_list.count('.') == 1, Exception(
                    f"Illegal sequence at {cursor}")
                edge_list.append(
                    (parent_node, child_idx, node, child_node_list.index('.')))
        return cursor, node
    cursor, _ = f(0)
    assert cursor == len(
        seq) - 1, Exception(f"Deserialization failed at {cursor}")

    return edge_list


if __name__ == "__main__":
    num_child_list = [4, 2, 1, 3, 1, 1, 1, 1]
    edge_list = [
        (0, 0, 5, 0),
        (0, 1, 3, 1),
        (0, 2, 1, 1),
        (0, 3, 6, 0),
        (3, 0, 7, 0),
        (3, 2, 4, 0),
        (1, 0, 2, 0)
    ]
    start_node = 0

    seq = serialize(num_child_list, edge_list, start_node)

    print(f"Serialize: {seq}")

    edge_list_deserialize = deserialize(seq, num_child_list)
    print(f"Deserialize: {edge_list_deserialize}")

    if sorted(edge_list) == sorted(edge_list_deserialize):
        print("Equal!")
    else:
        print("Unequal!")
