import unittest

from python.q_treesitter import get_parser, parse_and_transpile, parse_and_transpile_file
from astunparse import unparse


class TestQ2Py(unittest.TestCase):

    def test_parse_error(self):
        with self.assertRaises(NotImplementedError) as cm:
            self.parse(':0p')
        self.assertEqual(cm.exception.args[0], 'Parse ERROR for node start:(0, 0) end:(0, 1) :')

    def test_comment(self):
        self.assertEqual(self.parse('/ a comment'), '\n')

    def test_long(self):
        self.assertEqual(self.parse('1'), '1\n')
        self.assertEqual(self.parse('1j'), '1\n')

    def test_int(self):
        self.assertEqual(self.parse('2i'), '2\n')

    def test_float(self):
        self.assertEqual(self.parse('1.2'), '1.2\n')
        self.assertEqual(self.parse('1f'), '1.0\n')

    def test_boolean(self):
        self.assertEqual(self.parse('1b'), 'True\n')
        self.assertEqual(self.parse('0b'), 'False\n')

    def test_symbol(self):
        self.assertEqual(self.parse('`x'), 'x\n')

    def test_null_symbol(self):
        self.assertEqual(self.parse('`'), 'None\n')

    def test_symbol_with_namespace(self):
        self.assertEqual(self.parse('`.x.y'), 'x.y\n')
        self.assertEqual(self.parse('`.x.y.z'), 'x.y.z\n')

    def test_list(self):
        self.assertEqual(self.parse('(1;2;3)'), 'numpy.array(1, 2, 3)\n')

    def test_empty_list(self):
        self.assertEqual(self.parse('()'), 'numpy.array()\n')

    def test_op_add(self):
        self.assertEqual(self.parse('1+2'), '(2 + 1)\n')

    #    def test_op_subtract(self):
    #        self.assertEqual(self.parse('1-2'), '(2 - 1)\n')

    def test_op_multiply(self):
        self.assertEqual(self.parse('1*2'), '(2 * 1)\n')

    def test_op_divide(self):
        self.assertEqual(self.parse('1%2'), '(2 / 1)\n')

    def test_op_and(self):
        self.assertEqual(self.parse('1&2'), '(2 and 1)\n')

    def test_op_or(self):
        self.assertEqual(self.parse('1|2'), '(2 or 1)\n')

    def test_op_equal(self):
        self.assertEqual(self.parse('1=2'), '(2 == 1)\n')

    def test_op_join(self):
        self.assertEqual(self.parse('"abc", "de"'), '(\'de\' + \'abc\')\n')

    def test_op_each_join(self):
        self.assertEqual(self.parse('("abc"; "uv"),\'("de"; "xyz")'),
                         "numpy.add(numpy.array('de', 'xyz'), numpy.array('abc', 'uv'))\n")

    def test_signal_error(self):
        self.assertEqual(self.parse('\'"abc"'), "\nraise 'abc'\n")

    def test_op_take(self):
        # takes the first 5 elements of the list
        self.assertEqual(self.parse('5#0 1 2 3 4 5 6 7 8'), 'numpy.array(0, 1, 2, 3, 4, 5, 6, 7, 8)[:5]\n')

    def test_dictionary(self):
        self.assertEqual(self.parse('10 20 30!1.1 2.2 3.3'), '{1.1: 10, 2.2: 20, 3.3: 30}\n')

    def test_variable(self):
        self.assertEqual(self.parse('p: 10000'), '\np = 10000\n')

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
        parser = get_parser('../../tree-sitter-q')
        module = parse_and_transpile(parser, input, 'test')
        return unparse(module)

    def parse_file(self, input):
        parser = get_parser('../../tree-sitter-q')
        module = parse_and_transpile_file(parser, input)
        return unparse(module)
