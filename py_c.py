r"""This file load c++ library and deal with the interface
"""
import numpy as np
import time
from ctypes import cdll, POINTER, Array, cast
from ctypes import c_int


class CPPLib:
    """Class for operating CPP library

    Attributes:
        lib_path: (str) the path of a library, e.g. 'lib.so.6'
    """
    def __init__(self, lib_path):
        self.lib = cdll.LoadLibrary(lib_path)

        self.lib.test.argtypes = None
        self.lib.test.restype = (c_int)

        # neighbor_edge sampler
        IntArray = IntArrayType()
        self.lib.build_graph_c.argtypes = (IntArray, IntArray, IntArray, c_int, c_int)
        self.lib.build_graph_c.restype = None
        self.lib.sample_edges_c.argtypes = [c_int, POINTER(c_int)]
        self.lib.sample_edges_c.restype = None

    def build_graph(self, src, rel, dst, num_node, num_edge):
        self.lib.build_graph_c(src, rel, dst, num_node, num_edge)

    def neighbor_edge_sample(self, sample_size):
        result = (c_int * sample_size)()

        self.lib.sample_edges_c(sample_size, result)

        return np.array(list(result))

    def check_state(self):
        return self.lib.test()


class IntArrayType:
    # Define a special type for the 'int *' argument
    def from_param(self, param):
        typename = type(param).__name__
        if hasattr(self, 'from_' + typename):
            return getattr(self, 'from_' + typename)(param)
        elif isinstance(param, Array):
            return param
        else:
            raise TypeError("Can't convert %s" % typename)

    # Cast from array.array objects
    def from_array(self, param):
        if param.typecode != 'i':
            raise TypeError('must be an array of doubles')
        ptr, _ = param.buffer_info()
        return cast(ptr, POINTER(c_int))

    # Cast from lists/tuples
    def from_list(self, param):
        val = ((c_int) * len(param))(*param)
        return val

    from_tuple = from_list

    # Cast from a numpy array
    def from_ndarray(self, param):
        return param.ctypes.data_as(POINTER(c_int))


if __name__ == '__main__':
    lib = CPPLib('cpp/libsampler.so')

    state = lib.check_state()
    print(state)

    num_node = 14541
    num_rels = 237
    num_edge = 272115
    batch_size = 30000

    src = np.random.choice(np.arange(num_node), size=num_edge)
    rel = np.random.choice(np.arange(num_rels), size=num_edge)
    dst = np.random.choice(np.arange(num_node), size=num_edge)

    lib.build_graph(src, rel, dst, num_node, num_edge)

    while True:
        start = time.time()
        edges = lib.neighbor_edge_sample(batch_size)
        print('cost time: {} s'.format(time.time() - start))
