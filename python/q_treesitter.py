import ast
import typing
from _ast import Add, BinOp, Constant, FunctionDef, Call, Name, Module, ClassDef, arguments, arg, Attribute, Dict, \
    operator, Sub, Mult, Div, And, Or, boolop, BoolOp, For, Tuple, Assign, Raise, Subscript, Slice, Compare, cmpop, Eq, \
    Gt, Lt, LtE, GtE, NotEq, List, IfExp, Return, If, Expression, Expr, Lambda
from datetime import datetime
from pathlib import Path

from tree_sitter import Language, Parser, Node
from collections import deque
from astunparse import unparse

SHOULD_ERROR = True

def convert_expr_list(node: Node, tail, out: deque, named: dict):
    l = list(node.children)
    while len(l) > 0:
        child = l.pop()
        transpile(child, l, out, named)


def convert_expr_seq(node: Node, tail, out: deque, named: dict):
    l = list(node.children)
    while len(l) > 0:
        node = l.pop()
        seq = deque()
        transpile(node, l, seq, named)
        if len(seq) > 0:
            out.append(seq.pop())


def convert_long(node: Node, tail, out: deque):
    val:str = node.text.decode('utf-8')
    if val.endswith('j'):
        val = val[:len(val)-1]
    out.append(Constant(int(val), ""))

def convert_int(node: Node, tail, out: deque):
    val:str = node.text.decode('utf-8')
    if val.endswith('i'):
        val = val[:len(val)-1]
    out.append(Constant(int(val), ""))

def convert_short(node: Node, tail, out: deque):
    val:str = node.text.decode('utf-8')
    if val.endswith('h'):
        val = val[:len(val)-1]
    out.append(Call(Name('numpy.short'), [Constant(int(val), "")], []))

def convert_float(node: Node, tail, out: deque):
    val:str = node.text.decode('utf-8')
    if val.endswith('f'):
        val = val[:len(val)-1]
    out.append(Constant(float(val), ""))

def convert_string(node: Node, tail, out: deque):
    str_val: str = node.text.decode('utf-8')
    if str_val.startswith("\""):
        str_val = str_val[len("\""):]
    if str_val.endswith("\""):
        str_val = str_val[:-len("\"")]
    # str_val = str_val.removesuffix("\"").removeprefix("\"") - Python 3.9 only
    out.append(Constant(str_val, ""))

def convert_boolean(node: Node, tail, out: deque):
    val = node.text.decode('utf-8')
    if val == '1b':
        out.append(Constant(True, ""))
    elif val == '0b':
        out.append(Constant(False, ""))
    else:
        error(node.type, out)

def convert_timestamp(node: Node, tail, out: deque):
    val = node.text.decode('utf-8')
    #todo - Should this use numpy.datetime64 instead ?
    if val.endswith('p'):
        val = val[:len(val)-1]
        t = int(val)
        t2000 = t + 946728000
        out.append(Constant(datetime.fromtimestamp(t2000), ""))
    else:
        out.append(Constant(datetime.fromtimestamp(val), ""))

def convert_operator(node: Node, tail, out: deque, named: dict):
    if node.text == b'+':
        convert_bin_op(Add(), tail, out, named)
    elif node.text == b'-':
        convert_bin_op(Sub(), tail, out, named)
    elif node.text == b'*':
        convert_bin_op(Mult(), tail, out, named)
    elif node.text == b'%':
        convert_bin_op(Div(), tail, out, named)
    elif node.text == b'=':
        convert_compare_op(Eq(), tail, out, named)
    elif node.text == b'<>':
        convert_compare_op(NotEq(), tail, out, named)
    elif node.text == b'>':
        convert_compare_op(Gt(), tail, out, named)
    elif node.text == b'<':
        convert_compare_op(Lt(), tail, out, named)
    elif node.text == b'>=':
        convert_compare_op(GtE(), tail, out, named)
    elif node.text == b'<=':
        convert_compare_op(LtE(), tail, out, named)
    elif node.text == b'&':
        convert_bool_op(And(), tail, out, named)
    elif node.text == b'|':
        convert_bool_op(Or(), tail, out, named)
    elif node.text == b',':
        #join
        #todo - replace with something like itertools.chain
        if len(out) > 0:
            convert_bin_op(Add(), tail, out, named)
        else:
            error('failed to convert ,', out)
    elif node.text == b'!': #dictionary
        keys = []
        while len(out) > 0:
            keys.append(out.pop())
        while len(tail) > 0:
            transpile(tail.pop(), tail, out, named)
        values = []
        while len(out) > 0:
            values.append(out.pop())
        op = Dict(keys, values)
        out.append(op)
    elif node.text == b'\'': #each or signal an error
        lhs_args = []
        while len(out) > 0:
            lhs_args.append(out.pop())
        if len(tail) > 0: #this is an each and we have an operator
            each_op = tail.pop()
            while len(tail) > 0:
                transpile(tail.pop(), tail, out, named)
            rhs_args = []
            while len(out) > 0:
                rhs_args.append(out.pop())
            out.append(Call(Name('numpy.add'), [lhs_args,rhs_args], []))
        else: #this is a signal (exception)
            out.append(Raise(lhs_args, []))
    elif node.text == b'#':  # take or set attribute
        lhs_args = []
        while len(out) > 0:
            lhs_args.append(out.pop())
        while len(tail) > 0:
            transpile(tail.pop(), tail, out, named)
        rhs_args = []
        while len(out) > 0:
            rhs_args.append(out.pop())
        lhs = None
        if len(lhs_args) > 1:
            lhs = Call(Name('numpy.array'), lhs_args, [])
        elif len(lhs_args) == 1:
            lhs = lhs_args[0]
        if lhs is None:
            error('subscript with no lhs', out)
        else:
            out.append(Subscript(lhs,Slice([],rhs_args,[])))
    elif node.text == b':': #Return
        rhs_args = []
        while len(out) > 0:
            rhs_args.append(out.pop())
        out.append(Return(rhs_args))
    else:
        error('operator is ' + node.text.decode('utf-8'), out)


def convert_bin_op(op: operator, tail, out: deque, named: dict):
    if len(out) > 0:
        lhs = out.pop()
    else:
        lhs = Constant(None, "")
        error('No lhs for bin op', out)
    transpile(tail.pop(), tail, out, named)
    if len(out) > 0:
        rhs = out.pop()
    else:
        rhs = Constant(None, "")
        error('No rhs for bin op', out)
    opn = BinOp(lhs, op, rhs)
    out.append(opn)

def convert_bool_op(op: boolop, tail, out: deque, named: dict):
    if len(out) > 0:
        lhs = out.pop()
    else:
        lhs = Constant(None, "")
        error('No lhs for bool op', out)
    transpile(tail.pop(), tail, out, named)
    if len(out) > 0:
        rhs = out.pop()
    else:
        rhs = Constant(None, "")
        error('No rhs for bool op', out)
    opn = BoolOp(op, [lhs,rhs])
    out.append(opn)

def convert_compare_op(op: cmpop, tail, out: deque, named: dict):
    if len(out) > 0:
        rhs = out.pop()
    else:
        rhs = Constant(None, "")
        error('No rhs for compare op', out)
    transpile(tail.pop(), tail, out, named)
    if len(out) > 0:
        lhs = out.pop()
    else:
        lhs = Constant(None, "")
        error('No lhs for compare op', out)
    opn = Compare(lhs, [op], [rhs])
    out.append(opn)

def convert_local_var(node: Node, tail, out: deque, named: dict):
    assert node.child_count == 3
    names = deque()
    transpile(node.children[0], tail, names, named)
    name_node = names.pop()
    parent_node = name_node
    f_name = None
    namespace = []

    while isinstance(parent_node,Attribute):
        if f_name is None:
            f_name = parent_node.attr
        else:
            namespace.append(parent_node.attr)
        parent_node = parent_node.value

    if isinstance(parent_node, Name):
        namespace.append(parent_node.id)

    parent = {}
    if len(namespace) > 0:
        name = namespace.pop()
        if name not in named:
            named[name] = {}
        parent = named[name]

        while len(namespace) > 0:
            name = namespace.pop()
            if name not in parent:
                parent[name] = {}
            parent = parent[name]

    if f_name is None:
        f_name = name_node.id

    rhs = node.children[2]
    transpile(rhs, tail, out, named)
    if rhs.type == "expr_list" and rhs.children[0].type == "function_definition":
        if len(out) == 1:
            val = out.pop()
            if isinstance(val,Lambda):
                func = FunctionDef(f_name, val.args, body=val.body, decorator_list=[], lineno=0)
                out.append(func)
                parent[f_name] = func
            else:
                error('Unexpected function def', out)
        else:
            error('Unexpected function def', out)
    elif len(out) >= 1:
        if len(out) > 1:
            args = []
            while len(out) > 0:
                args.append(out.pop())
            rhs = Call(Name('numpy.array'), args, [])
        else:
            rhs = out.pop()
        assign = Assign([name_node], rhs)
        parent[f_name] = assign
        out.append(assign)
    else:
        error('local var assign no rhs', out)


def convert_function_call(node: Node, tail, out: deque, named: dict):
    assert node.child_count > 0
    lhs = node.children[0].text.decode('utf-8')
    args = []
    if node.child_count == 2:
        args_node = node.children[1]
        transpile(args_node, tail, out, named)
        for _ in range(len(out)):
            args.append(out.pop())
        args.reverse()
    else:
        error('unexpected args for function', out)
    call = Call(Name(lhs), args, [])
    out.append(call)


def convert_builtin_function_call(node: Node, tail, out: deque, named: dict):
    convert_function_call(node, tail, out, named)


def convert_function_definition(node: Node, tail, out: deque, named: dict):
    args_node = node.child_by_field_name('args')
    fn_args = []
    if args_node is not None:
        arg_out = deque()
        args = []
        convert_index(args_node, tail, arg_out, named)
        while len(arg_out) > 0:
            arg_name_node = arg_out.popleft()
            arg_name = ''
            if isinstance(arg_name_node, Name):
                arg_name = arg_name_node.id
            else:
                error('unknown func arg', out)
            args.append(arg(arg=arg_name, annotation=[]))
        fn_args = arguments(posonlyargs=[], args=args, defaults=[], kwonlyargs=[], vararg=[], kwarg=[])
    body_node = node.child_by_field_name('body')
    body_exprs = []
    if body_node is not None:
        transpile(body_node, tail, out, named)
        while len(out) > 0:
            body_exprs.append(Expr(out.pop()))
    out.append(Lambda(fn_args,body_exprs))


def convert_namespace(node: Node, tail, out: deque, named: dict):
    nsp_out = deque()
    for child in node.children:
        transpile(child, tail, nsp_out, named)
    attrs = []
    while len(nsp_out) > 1:
        attrs.append(nsp_out.pop())
    name = nsp_out.pop()
    val = attrs.pop()
    attr = Attribute(name, attr=val.id)
    while len(attrs) > 0:
        val = attrs.pop()
        attr = Attribute(attr, attr=val.id)
    out.append(attr)


def convert_entity_name(node: Node, tail, out: deque):
    assert node.child_count == 0
    out.append(Name(node.text.decode('utf-8')))


def convert_entity_namespace(node: Node, tail, out: deque):
    assert node.child_count == 0
    out.append(Name(node.text.decode('utf-8')[1:]))

def convert_index(node: Node, tail, out: deque, named: dict):
    for child in node.children:
        transpile(child, tail, out, named)


def convert_table(node: Node, tail, out: deque, named: dict):
    inits = []
    for child in node.children:
        if child.type == 'variable_assign':
            column_name = child.children[0]
            transpile(column_name, tail, out, named)
            keys = [ out.pop() ]
            value_node = child.children[2]
            transpile(value_node, tail, out, named)
            values = []
            while len(out) > 0:
                values.append(out.pop())
            op = Dict(keys, [List(values)])
            inits.append(op)
    out.append(Call(Name('numpy.array'), inits, []))


def convert_list(node: Node, tail, out: deque, named: dict):
    for child in node.children:
        transpile(child, tail, out, named)
    args = []
    while len(out) > 0:
        args.append(out.pop())
    out.append(Call(Name('numpy.array'), args, []))


def convert_symbol(node: Node, tail, out: deque, named: dict):
    symbol_name: str = node.text.decode('utf-8')
    if len(symbol_name) > 1:
        symbol_name = symbol_name[len("`"):]
        name = Name(symbol_name)
        out.append(name)

    if len(out) == 0:
        out.append(Constant(None, ""))

def convert_if(node: Node, tail, out: deque, named: dict):
    vals = []
    for child in node.children:
        if child.text == b';':
            while len(out) > 0:
                vals.append(out.pop())
        elif child.text != b'$[' and child.text != b'if':
            transpile(child, tail, out, named)
    while len(out) > 0:
        vals.append(out.pop())
    if len(vals) >= 3 and len(vals) % 2 != 0: #odd number of args
        while len(vals) >= 3:
            arg3 = vals.pop()
            arg2 = vals.pop()
            arg1 = vals.pop()
            vals.append(If(arg1, [arg2], [arg3]))
        out.append(vals.pop())
    else:
        error('Unsupported if', out)

def convert_cast(node: Node, tail, out: deque, named: dict):
    assert node.child_count == 2
    rhs_args = []
    if len(out) > 0:
        rhs_args.append(out.pop())
    transpile(node.children[0], tail, out, named)
    out.append(Call(out.pop(), rhs_args, []))

def transpile(node: Node, tail, out: deque, named: dict):
    if node.type == "source_file":
        c_out = deque()
        for child in node.children:
            transpile(child, [], c_out, named)
        while len(c_out) > 0:
            nd = c_out.pop()
            if not isinstance(nd,FunctionDef) and not isinstance(nd,Assign):
                nd = Expr(nd)
            out.append(nd)
    elif node.type == "expr_list":
        convert_expr_list(node, tail, out, named)
    elif node.type == "expr_seq":
        convert_expr_seq(node, tail, out, named)
    elif node.type == "long":
        convert_long(node, [], out)
    elif node.type == "int":
        convert_int(node, [], out)
    elif node.type == "short":
        convert_short(node, [], out)
    elif node.type == "float":
        convert_float(node, [], out)
    elif node.type == "string":
        convert_string(node, [], out)
    elif node.type == "boolean":
        convert_boolean(node, [], out)
    elif node.type == "timestamp":
        convert_timestamp(node, [], out)
    elif node.type == "operator":
        convert_operator(node, tail, out, named)
    elif node.type == "variable_assign":
        convert_local_var(node, tail, out, named)
    elif node.type == "function_definition":
        convert_function_definition(node, tail, out, named)
    elif node.type == "builtin_function_call":
        convert_builtin_function_call(node, tail, out, named)
    elif node.type == "entity_with_index":
        convert_function_call(node, tail, out, named)
    elif node.type == "namespace":
        convert_namespace(node, tail, out, named)
    elif node.type == "entity_name":
        convert_entity_name(node, tail, out)
    elif node.type == "entity_nsp":
        convert_entity_namespace(node, tail, out)
    elif node.type == "index":
        convert_index(node, tail, out, named)
    elif node.type == "table":
        convert_table(node, tail, out, named)
    elif node.type == "list":
        convert_list(node, tail, out, named)
    elif node.type == "symbol":
        convert_symbol(node, tail, out, named)
    elif node.type == "if_else":
        convert_if(node, tail, out, named)
    elif node.type == "cast":
        convert_cast(node, tail, out, named)
    elif node.type == "comment":
        None
    elif node.type == ";":
        None
    elif node.type == ".":
        None
    elif node.type == "[":
        None
    elif node.type == "]":
        None
    elif node.type == "(":
        None
    elif node.type == ")":
        None
    elif node.type == "ERROR":
        error("Parse ERROR for node start:" + str(node.start_point) + " end:" + str(node.end_point) + " " + node.text.decode('utf-8'), out)
        # if we are just logging errors try to continue
        for child in node.children:
            transpile(child, [], out, named)
    else:
        error(node.type, out)
    return out

def error(message:str, out):
    if SHOULD_ERROR:
        raise NotImplementedError(message)
    else:
        out.clear() #clear the current stack and try to continue
        print('ERROR not implemented:' + message)

def get_parser(path:str) -> Parser:
    Language.build_library(
        # Store the library in the `build` directory
        'build/my-languages.so',

        # Include one or more languages
        [
            path,
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
        test_funcs.reverse()
        for test_func in test_funcs:
            test_func.args=arguments(posonlyargs=[],args=[arg(arg='self',annotation=[])],defaults=[],kwonlyargs=[],vararg=[],kwarg=[])
            for stmt in test_func.body:
                if isinstance(stmt, Expr):
                    value = stmt.value
                    if isinstance(value, Call):
                        test_fn_call:Call = value
                        if isinstance(test_fn_call.func, Name) and test_fn_call.func.id == 'AEQ':
                            test_fn_call.func = Attribute(value=Name(id='self'), attr='assertEqual')

        test_class_name = module_name[0].upper() + module_name[1:]
        test_class = ClassDef(test_class_name,[Name(id='unittest.TestCase')],[],test_funcs,[])
        import_st = ast.parse('import unittest')
        main = ast.parse('if __name__ == \'__main__\':unittest.main()')
        mod = Module([import_st, test_class, main], [])
        return mod
    else:
        out_nodes = []
        while len(out) > 0:
            out_nodes.append(out.pop())
        mod = Module(out_nodes,[])
        return mod


def parse_and_transpile_file(parser, file_name) -> Module:
    f = open(file_name, "r")
    path:Path = Path(file_name)
    mod = parse_and_transpile(parser, f.read(), path.stem)
    f.close()
    return mod

def main():
    print("Running q2Py!")

    parser:Parser = get_parser('../../tree-sitter-q')
    mod = parse_and_transpile_file(parser, "../q/testExample.q")
    print(unparse(mod))


if __name__ == "__main__":
    main()
