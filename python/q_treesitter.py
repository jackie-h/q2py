from tree_sitter import Language, Parser


def convert_expr_list(node, out):
    for child in node.children:
        out.append(transpile(child, out))


def convert_number(node, out):
    print(node)


def convert_operator(node, out):
    print(node)


def transpile(node, out):
    print(node.type)
    if node.type == "source_file":
        for child in node.children:
            transpile(child, out)
    elif node.type == "expr_list":
        convert_expr_list(node, out)
    elif node.type == "number":
        convert_number(node,out)
    elif node.type == "operator":
        convert_operator(node,out)
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
    transpile(root_node, [])


if __name__ == "__main__":
    main()
