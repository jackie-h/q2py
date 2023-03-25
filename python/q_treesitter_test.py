import unittest

from python.q_treesitter import get_parser, parse_and_transpile, parse_and_transpile_file
from astunparse import unparse


class TestStringMethods(unittest.TestCase):
    def test_boolean(self):
        self.assertEqual(self.parse('1b'), 'True\n')
        self.assertEqual(self.parse('0b'), 'False\n')

    def test_symbol(self):
        self.assertEqual(self.parse('`x'), 'x\n')

    def test_symbol_with_namespace(self):
        self.assertEqual(self.parse('`.x.y'), 'x.y\n')

    def test_op_add(self):
        self.assertEqual(self.parse('1+2'), '(2 + 1)\n')

    def test_test(self):
        self.assertEqual(self.parse_file("../q/testExample.q"), '''
import unittest

class TestExample(unittest.TestCase):

    def test_add(self):self.assertEqual(3, (2 + 1), '1+2 should equal 3')
if (__name__ == '__main__'):
    unittest.main()
''')

    def test_test_mocks(self):
            self.assertEqual(self.parse_file("../q/testMocks.q"), '''
import unittest

class TestMocks(unittest.TestCase):

    def test_mocks(self):MOCK(e.DEBUG, False)MOCK(e.ENV.utest)MOCK(get(s.doX, z), x, y, z)MOCK(s.assertOk, True)
if (__name__ == '__main__'):
    unittest.main()
''')

    def parse(self, input):
        parser = get_parser()
        module = parse_and_transpile(parser, input, 'test')
        return unparse(module)

    def parse_file(self, input):
        parser = get_parser()
        module = parse_and_transpile_file(parser, input)
        return unparse(module)