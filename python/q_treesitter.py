import ast
import typing
from _ast import Add, BinOp, Constant, FunctionDef, Call, Name, Module, ClassDef, arguments, arg, Attribute
from pathlib import Path
from ast import unparse

from tree_sitter import Language, Parser, Node
from collections import deque


def convert_expr_list(node: Node, tail, out: deque, named: dict):
    l = list(node.children)
    while len(l) > 0:
        node = l.pop()
        transpile(node, l, out, named)


def convert_expr_seq(node: Node, tail, out: deque, named: dict):
    l = list(node.children)
    while len(l) > 0:
        node = l.pop()
        transpile(node, l, out, named)


def convert_number(node: Node, tail, out: deque):
    out.append(Constant(float(node.text.decode('utf-8'))))


def convert_string(node: Node, tail, out: deque):
    str_val_with_quotes: str = node.text.decode('utf-8')
    str_val = str_val_with_quotes.removesuffix("\"").removeprefix("\"")
    out.append(Constant(str_val))


def convert_operator(node: Node, tail, out: deque, named: dict):
    lhs = out.pop()
    transpile(tail.pop(), tail, out, named)
    rhs = out.pop()
    if node.text == b'+':
        op = BinOp(lhs, Add(), rhs)
        out.append(op)


def convert_local_var(node: Node, tail, out: deque, named: dict):
    assert node.child_count == 3
    names = deque()
    transpile(node.children[0], tail, names, named)
    names.reverse()
    parent = named
    while len(names) > 1:
        name = names.pop().id
        if name not in parent:
            parent[name] = {}
        parent = named[name]

    var_name: Name = names.pop()
    transpile(node.children[2], tail, out, named)
    value = out.pop()
    if node.children[2].type == "function_body":
        func = FunctionDef(var_name.id, [], body=[value], decorator_list=[], lineno=0)
        out.append(func)
        parent[var_name.id] = func


def convert_function_call(node: Node, tail, out: deque, named: dict):
    assert node.child_count == 2
    lhs = node.children[0].text.decode('utf-8')
    args_node = node.children[1]
    transpile(args_node, tail, out, named)
    args = []
    for _ in range(args_node.child_count):
        args.append(out.pop())
    call = Call(Name(lhs), args, [])
    out.append(call)


def convert_function_body(node: Node, tail, out: deque, named: dict):
    assert node.child_count == 3
    transpile(node.children[1], tail, out, named)


def convert_namespace(node: Node, tail, out: deque, named: dict):
    assert len(out) == 0
    for child in node.children:
        transpile(child, tail, out, named)


def convert_entity_name(node: Node, tail, out: deque):
    assert node.child_count == 0
    out.append(Name(node.text.decode('utf-8')))


def convert_table(node: Node, tail, out: deque, named: dict):
    assert node.child_count == 3
    transpile(node.children[1], tail, out, named)


def transpile(node: Node, tail, out: deque, named: dict):
    print(node.type)
    if node.type == "source_file":
        for child in node.children:
            transpile(child, [], out, named)
    elif node.type == "expr_list":
        convert_expr_list(node, tail, out, named)
    elif node.type == "expr_seq":
        convert_expr_seq(node, tail, out, named)
    elif node.type == "number":
        convert_number(node, [], out)
    elif node.type == "string":
        convert_string(node, [], out)
    elif node.type == "operator":
        convert_operator(node, tail, out, named)
    elif node.type == "local_variable_assignment":
        convert_local_var(node, tail, out, named)
    elif node.type == "function_body":
        convert_function_body(node, tail, out, named)
    elif node.type == "function_call":
        convert_function_call(node, tail, out, named)
    elif node.type == "namespace":
        convert_namespace(node, tail, out, named)
    elif node.type == "entity_name":
        convert_entity_name(node, tail, out)
    elif node.type == "table":
        convert_table(node, tail, out, named)
    elif node.type == ";":
        None
    elif node.type == ".":
        None
    else:
        raise NotImplementedError
    return out


def get_parser() -> Parser:
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
    return parser


def parse_and_transpile(parser:Parser, text:str, module_name:str) -> Module:
    tree = parser.parse(bytes(text, "utf8"))
    root_node = tree.root_node
    print(root_node.sexp())
    out = deque()
    named = {}
    transpile(root_node, [], out, named)
    if 'test' in named:
        tests: dict = named['test']
        test_funcs:typing.List[FunctionDef] = list(tests.values())
        for test_func in test_funcs:
            test_func.args=arguments(posonlyargs=[],args=[arg(arg='self')],defaults=[],kwonlyargs=[])
            for stmt in test_func.body:
                if isinstance(stmt, Call):
                    test_fn_call:Call = stmt
                    if isinstance(test_fn_call.func, Name) and test_fn_call.func.id == 'AEQ':
                        test_fn_call.func = Attribute(value=Name(id='self'), attr='assertEqual')

        test_class_name = module_name[0].upper() + module_name[1:]
        test_class = ClassDef(test_class_name,[Name(id='unittest.TestCase')],[],test_funcs,[])
        import_st = ast.parse('import unittest')
        main = ast.parse('if __name__ == \'__main__\':unittest.main()')
        mod = Module([import_st, test_class, main], [])
        return mod
    else:
        mod = Module([out.pop()],[])
        return mod


def parse_and_transpile_file(parser, file_name) -> Module:
    f = open(file_name, "r")
    path:Path = Path(file_name)
    return parse_and_transpile(parser, f.read(), path.stem)

def main():
    print("Hello Q!")

    parser:Parser = get_parser()
    mod = parse_and_transpile(parser, "1+2", "example")
    print(unparse(mod))
    mod = parse_and_transpile(parser,"""
        .test.test_add:{
          AEQ[3;1+2;"1+2 should equal 3"]
        };
            """, "example")
    print(unparse(mod))
    mod = parse_and_transpile_file(parser, "../q/testExample.q")
    print(unparse(mod))


if __name__ == "__main__":
    main()
