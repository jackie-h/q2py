from pyq import q
from ast import *

def t_int(elem, k_tail):
    return Constant(int(elem))


def t_bin_op(elem, k_tail):
    lhs = transpile(k_tail.pop(), k_tail)
    rhs = transpile(k_tail.pop(), k_tail)
    return BinOp(lhs, Add(), rhs)


def transpile(elem, k_tail):
    type = q.type(elem)
    str_type = str(type)
    if str_type == '102h':
        out = t_bin_op(elem, k_tail)
    elif str_type == '-7h':
        out = t_int(elem, k_tail)
    else:
        raise NotImplementedError
    return out


def parse(input_q_code):
    text = 'parse \"{}\"'.format(input_q_code)
    return list(q(text))

def main():
    print("Hello World!")
    a = parse("1+2")
    a.reverse()
    elem = a.pop()
    b = transpile(elem, a)
    print(unparse(b))


if __name__ == "__main__":
    main()