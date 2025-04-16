
import os
import sys
import json
from anytree import Node, RenderTree
from anytree.exporter import DotExporter

def iterate_tree(tree: Node, moves: list[str]) -> Node:

    currTree = tree
    for idx, move in enumerate(moves):

        children = [child for child in currTree.children if child.name == move]
        if not children:
            currTree = Node(move, parent=currTree, endsHere = 0)
        else:
            currTree = children[0]

        if idx == len(moves) - 1:
            currTree.endsHere += 1
        
    return tree

def collect_paths_with_endsHere(node: Node, path=None, results=None):
    if path is None:
        path = []
    if results is None:
        results = []

    path.append(node.name)

    if node.endsHere > 0:
        results.append((" -> ".join(path), node.endsHere))

    for child in node.children:
        collect_paths_with_endsHere(child, path[:], results)

    return results

if __name__ == "__main__":

    args = sys.argv[1:]
    if len(args) != 1:
        print(f"Invalid Usage. Valid Usage: python3 {sys.argv[0]} dir")
        exit()

    trees = {} # tree root value: tree

    outs = os.listdir(args[0])
    for out in outs:
        if out.endswith(".json"):
            with open(f"{args[0]}/{out}", "r") as fd:
                fc = json.load(fd)

                moves = fc.get("moves", None)
                if not moves:
                    print("Invalid File format: Expected moves, but not found")

                mvs = []
                for idx, move in enumerate(moves):
                    mv = move.get("move", None)
                    if not mv:
                        print(f"Invalid File format: Expected move in moves but not found in {idx}th iteration")

                    mvs.append(mv)

                if mvs[0] in trees.keys():
                    trees[mvs[0]] = iterate_tree(trees[mvs[0]], mvs)
                else:
                    # create a tree and then
                    trees[mvs[0]] = Node(mvs[0], endsHere = 0)
                    trees[mvs[0]] = iterate_tree(trees[mvs[0]], mvs)
                
    # print the trees and their paths with endsHere > 0, sorted descendingly
    paths = []
    for tree in trees.values():
        # Collect and sort paths
        paths.extend(collect_paths_with_endsHere(tree))

    print("Paths with endsHere > 0:")
    print(len(paths))
    paths.sort(key=lambda x: x[1], reverse=True)
    # for path, endsHere in paths:
        # print(f"{path} ({endsHere})")
    print("-" * 40)
