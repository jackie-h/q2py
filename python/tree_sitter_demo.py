from tree_sitter import Language, Parser


def main():
    print("Hello World!")

    Language.build_library(
        # Store the library in the `build` directory
        'build/my-languages.so',

        # Include one or more languages
        [
            '../vendor/tree-sitter-python',
        ]
    )

    PY_LANGUAGE = Language('build/my-languages.so', 'python')

    parser = Parser()
    parser.set_language(PY_LANGUAGE)

    # tree = parser.parse(bytes("""
    # def foo():
    #     if bar:
    #         baz()
    # """, "utf8"))

    tree = parser.parse(bytes("""
        1+2
        """, "utf8"))

    root_node = tree.root_node
    print(root_node.sexp())


if __name__ == "__main__":
    main()
