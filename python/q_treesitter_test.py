import unittest

from python.q_treesitter import get_parser, parse_and_transpile, parse_and_transpile_file
from astunparse import unparse


class TestQ2Py(unittest.TestCase):

    def test_parse_error(self):
        with self.assertRaises(NotImplementedError) as cm:
            self.parse('}{ "Hello World"')
        self.assertEqual('Parse ERROR for node start:(0, 0) end:(0, 16) }{ "Hello World"', cm.exception.args[0])

    def test_comment(self):
        self.assertEqual('\n', self.parse('/ a comment'))

    def test_long(self):
        self.assertEqual('\n1\n', self.parse('1'))
        self.assertEqual('\n1\n', self.parse('1j'))

    def test_int(self):
        self.assertEqual('\n2\n', self.parse('2i'))

    def test_short(self):
        self.assertEqual('\nnumpy.short(2)\n', self.parse('2h'))

    def test_float(self):
        self.assertEqual('\n1.2\n', self.parse('1.2'))
        self.assertEqual('\n1.0\n', self.parse('1f'))
        self.assertEqual('\n0.3\n', self.parse('.3'))

    def test_boolean(self):
        self.assertEqual('\nTrue\n', self.parse('1b'))
        self.assertEqual('\nFalse\n', self.parse('0b'))

    def test_timestamp(self):
        self.assertEqual('\ndatetime.datetime(2000, 1, 1, 7, 0)\n', self.parse('0p'))

    def test_symbol(self):
        self.assertEqual('\nx\n', self.parse('`x'))

    def test_null_symbol(self):
        self.assertEqual('\nNone\n', self.parse('`'))

    def test_symbol_with_namespace(self):
        self.assertEqual('\nx.y\n', self.parse('`.x.y'), )
        self.assertEqual('\nx.y.z\n', self.parse('`.x.y.z'))

    def test_list(self):
        self.assertEqual('\nnumpy.array(1, 2, 3)\n', self.parse('(1;2;3)'))

    def test_empty_list(self):
        self.assertEqual('\nnumpy.array()\n', self.parse('()'))

    def test_op_add(self):
        self.assertEqual('\n(2 + 1)\n', self.parse('1+2'))

    #    def test_op_subtract(self):
    #        self.assertEqual(self.parse('1-2'), '(2 - 1)\n')

    def test_op_multiply(self):
        self.assertEqual('\n(2 * 1)\n', self.parse('1*2'))

    def test_op_divide(self):
        self.assertEqual('\n(2 / 1)\n', self.parse('1%2'))

    def test_op_and(self):
        self.assertEqual('\n(2 and 1)\n', self.parse('1&2'))

    def test_op_or(self):
        self.assertEqual('\n(2 or 1)\n', self.parse('1|2'))

    def test_op_equal(self):
        self.assertEqual('\n(1 == 2)\n', self.parse('1=2'))

    def test_op_not_equal(self):
        self.assertEqual('\n(1 != 2)\n', self.parse('1<>2'))

    def test_op_gt(self):
        self.assertEqual('\n(1 > 2)\n', self.parse('1>2'))

    def test_op_lt(self):
        self.assertEqual('\n(1 < 2)\n', self.parse('1<2'))

    def test_op_gte(self):
        self.assertEqual('\n(1 >= 2)\n', self.parse('1>=2'))

    def test_op_lte(self):
        self.assertEqual('\n(1 <= 2)\n', self.parse('1<=2'))

    def test_op_join(self):
        self.assertEqual('\n(\'de\' + \'abc\')\n', self.parse('"abc", "de"'))

    def test_op_each_join(self):
        self.assertEqual("\nnumpy.add(numpy.array('de', 'xyz'), numpy.array('abc', 'uv'))\n",
                         self.parse('("abc"; "uv"),\'("de"; "xyz")'))

    def test_signal_error(self):
        self.assertEqual("\n\nraise 'abc'\n", self.parse('\'"abc"'))

    def test_op_take(self):
        # takes the first 5 elements of the list
        self.assertEqual('\nnumpy.array(0, 1, 2, 3, 4, 5, 6, 7, 8)[:5]\n', self.parse('5#0 1 2 3 4 5 6 7 8'))

    def test_dictionary(self):
        self.assertEqual('\n{1.1: 10, 2.2: 20, 3.3: 30}\n', self.parse('10 20 30!1.1 2.2 3.3'))

    def test_variable(self):
        self.assertEqual('\np = 10000\n', self.parse('p: 10000'))

    def test_variable_namespace(self):
        self.assertEqual('\na.b.c.d = 10000\n', self.parse('.a.b.c.d: 10000'))

    def test_variable_multiple_values(self):
        self.assertEqual('\np = numpy.array(0, 10, 5)\n', self.parse('p:0 10 5'))

    def test_variable_multiple_symbols(self):
        self.assertEqual('\nn = numpy.array(x, y, x)\n', self.parse('n:`x`y`x'))

    def test_table(self):
        self.assertEqual('\nt = numpy.array({n: [x, y, x, z, z, y]}, {p: [0, 15, 12, 20, 25, 14]})\n',
                         self.parse('t:([]n:`x`y`x`z`z`y;p:0 15 12 20 25 14)'))
    #?[t;();0b;()]

    def test_if_else_dollar(self):
        self.assertEqual('''

if (a > 10):
    a = 20
else:
    r = 'true'
''', self.parse('$[a > 10;a:20;r: "true"]'))

    def test_if_else(self):
            self.assertEqual('''

if (a > 10):
    a = 20
else:
    r = 'true'
''', self.parse('if[a > 10;a:20;r: "true"]'))

    def test_if_elif(self):
        self.assertEqual('''

if (a > 2):2
elif (a > 5):5
else:1
''', self.parse('$[a>2;2;a>5;5;1]'))

    def test_cast(self):
        self.assertEqual('\ndate(datetime.datetime(2000, 1, 1, 7, 0))\n', self.parse('`date$0p'))

    def test_return(self):
        self.assertEqual('\n\nreturn 10000\n', self.parse(':10000;'))

    def test_function_def(self):
        self.assertEqual('''

def f():
    (x + 2)
''', self.parse('f:{2+x}'))

    def test_function_def_with_args(self):
        self.assertEqual('''

def f(a, b, c):
    (((c * 2) + b) + a)
''', self.parse('f:{[a;b;c] a+b+2*c}'))

    def test_function_def_multiline(self):
        self.assertEqual('''

def f():
    
    y = (x + 2)
    y
''', self.parse('f:{y:2+x;y}'))

    def test_namespace_mixed_with_symbol_exprs(self):
        self.assertEqual('''
b
a
j.j
''', self.parse('.j.j`a`b'))

    def test_test(self):
        self.assertEqual('''
import unittest

class TestExample(unittest.TestCase):

    def test_add(self):
        self.assertEqual(3, (2 + 1), '1+2 should equal 3')

    def test_subtract(self):
        self.assertEqual(1, -1, 2, '2-1 should equal 1')
if (__name__ == '__main__'):
    unittest.main()
''', self.parse_file("../q/testExample.q"))

    def test_test_mocks(self):
        self.assertEqual('''
import unittest

class TestMocks(unittest.TestCase):

    def test_mocks(self):
        MOCK(e.DEBUG, False)
        MOCK(e.ENV, utest)
        MOCK((lambda x, y, z: 
        get(s.doX, z)))
        MOCK((lambda : 
        True
        s.assertOk))
if (__name__ == '__main__'):
    unittest.main()
''', self.parse_file("../q/testMocks.q"))

    def parse(self, input):
        parser = get_parser('../../tree-sitter-q')
        module = parse_and_transpile(parser, input, 'test')
        return unparse(module)

    def parse_file(self, input):
        parser = get_parser('../../tree-sitter-q')
        module = parse_and_transpile_file(parser, input)
        return unparse(module)
