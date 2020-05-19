cdef extern from "examples.hpp":
    void hello(const char *name)

def py_hello(name: bytes) -> None:
    hello(name)
