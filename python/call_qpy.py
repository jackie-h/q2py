from pyq import q

def main():
    print("Hello World!")
    a = q("parse \"1+2\"")
    b = list(a)
    c = q.type(b[0])
    d = c.inspect(b'g')
    print(c)



if __name__ == "__main__":
    main()