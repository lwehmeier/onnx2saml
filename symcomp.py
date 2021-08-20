from sympy import symbols
import sympy
from sympy.plotting import plot3d
import sys


def buildLayerEq(w, b, layer, nneurons, inputs):
    outs = []
    import multiprocessing
    pool = multiprocessing.Pool()
    args = [(inputs, b[layer], w[layer], n) for n in range(nneurons)]
    #for a in args:
    #    outs.append(buildneq(a))
    y_vars = sympy.symbols("%s" % ', '.join(map(str, ["l"+str(layer)+"x"+str(i) for i in range(nneurons)])))
    outs = pool.map(buildneq, args)
    outs = [sympy.Eq(y_vars[i], outs[i]) for i in range(nneurons)]
    return outs, y_vars
def buildneq(f):
    sys.setrecursionlimit(1024*64)
    inputs, b, w, n = f
    weights = w[n]
    bias = b[n]
    ysyms = [inputs[k] * weights[k] for k in range(len(inputs))]
    ysym = bias
    for eq in ysyms: ysym += eq
    # apply relu
    ysym = sympy.Piecewise((0,ysym<0),(ysym,ysym >= 0))
    print(ysym)
    return ysym