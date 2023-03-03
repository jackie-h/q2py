from _ast import Add, BinOp, Constant
from ast import unparse

from tree_sitter import Language, Parser, Node
from collections import deque


def convert_expr_list(node: Node, tail, out: deque):
    l = list(node.children)
    while len(l) > 0:
        node = l.pop()
        transpile(node, l, out)


def convert_number(node: Node, tail, out: deque):
    out.append(Constant(float(node.text.decode('utf-8'))))


def convert_operator(node: Node, tail, out: deque):
    lhs = out.pop()
    transpile(tail.pop(), tail, out)
    rhs = out.pop()
    if node.text == b'+':
        op = BinOp(lhs, Add(), rhs)
        out.append(op)


def  transpile(node: Node, tail, out: deque):
    print(node.type)
    if node.type == "source_file":
        for child in node.children:
            transpile(child, [], out)
    elif node.type == "expr_list":
        convert_expr_list(node, tail, out)
    elif node.type == "number":
        convert_number(node, [], out)
    elif node.type == "operator":
        convert_operator(node, tail, out)
    else:
        raise NotImplementedError
    return out


def main():
    print("Hello World!")

    Language.build_library(
        # Store the library in the `build` directory
        'build/my-languages.so',

        # Include one or more languages
        [
            '../../tree-sitter-q',
        ]
    )

    Q_LANGUAGE = Language('build/my-languages.so', 'q')

    parser = Parser()
    parser.set_language(Q_LANGUAGE)

    tree = parser.parse(bytes("""
        1+2
        """, "utf8"))

    root_node = tree.root_node
    print(root_node.sexp())
    out = deque()
    transpile(root_node, [], out)
    print(unparse(out.pop()))


if __name__ == "__main__":
    main()
