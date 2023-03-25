import unittest

from python.q_treesitter import get_parser, parse_and_transpile
from astunparse import unparse


class TestStringMethods(unittest.TestCase):
    def test_symbol(self):
        self.assertEqual(self.parse('`x'), 'x\n')

    def test_symbol_with_namespace(self):
        self.assertEqual(self.parse('`.x.y'), 'x.y\n')

    def parse(self, input):
        parser = get_parser()
        module = parse_and_transpile(parser, input, 'test')
        return unparse(module)
