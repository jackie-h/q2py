from _ast import Add, BinOp, Constant, FunctionDef
from ast import unparse

from tree_sitter import Language, Parser, Node
from collections import deque


def convert_expr_list(node: Node, tail, out: deque):
    l = list(node.children)
    while len(l) > 0:
        node = l.pop()
        transpile(node, l, out)


def convert_expr_seq(node: Node, tail, out: deque):
    l = list(node.children)
    while len(l) > 0:
        node = l.pop()
        transpile(node, l, out)

def convert_number(node: Node, tail, out: deque):
    out.append(Constant(float(node.text.decode('utf-8'))))

def convert_string(node: Node, tail, out: deque):
    out.append(Constant(node.text.decode('utf-8')))

def convert_operator(node: Node, tail, out: deque):
    lhs = out.pop()
    transpile(tail.pop(), tail, out)
    rhs = out.pop()
    if node.text == b'+':
        op = BinOp(lhs, Add(), rhs)
        out.append(op)


def convert_local_var(node: Node, tail, out: deque):
    assert node.child_count == 3
    var_name = node.children[0].text.decode('utf-8')
    transpile(node.children[2], tail, out)
    value = out.pop()
    if node.children[2].type == "function_body":
        func = FunctionDef(var_name, [], body=[value],
                           decorator_list=[],lineno=0)
        out.append(func)
    print(node)


def convert_function_body(node: Node, tail, out: deque):
    assert node.child_count == 3
    transpile(node.children[1], tail, out)
    print(node)
def convert_namespace(node: Node, tail, out: deque):
    assert node.child_count == 2
    convert_string(node.children[1], tail, out)

def convert_entity_name(node: Node, tail, out: deque):
    assert node.child_count == 0
    convert_string(node, tail, out)

def convert_table(node: Node, tail, out: deque):
    assert node.child_count == 3
    transpile(node.children[1], tail, out)

def  transpile(node: Node, tail, out: deque):
    print(node.type)
    if node.type == "source_file":
        for child in node.children:
            transpile(child, [], out)
    elif node.type == "expr_list":
        convert_expr_list(node, tail, out)
    elif node.type == "expr_seq":
        convert_expr_seq(node, tail, out)
    elif node.type == "number":
        convert_number(node, [], out)
    elif node.type == "string":
        convert_string(node, [], out)
    elif node.type == "operator":
        convert_operator(node, tail, out)
    elif node.type == "local_variable_assignment":
        convert_local_var(node, tail, out)
    elif node.type == "function_body":
        convert_function_body(node, tail, out)
    elif node.type == "namespace":
        convert_namespace(node, tail, out)
    elif node.type == "entity_name":
        convert_entity_name(node, tail, out)
    elif node.type == "table":
        convert_table(node, tail, out)
    elif node.type == ";":
        None
    elif node.type == ":":
        None
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
    .test.test_add:{
      AEQ[3;1+2;""]
    };
        """, "utf8"))

    root_node = tree.root_node
    print(root_node.sexp())
    out = deque()
    transpile(root_node, [], out)
    print(unparse(out.pop()))


if __name__ == "__main__":
    main()
