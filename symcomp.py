from sympy import symbols
import sympy
from sympy.plotting import plot3d
import sys


def buildRnnLayerEq(w_x, w_h,b_x,b_h,layer,nneurons,inputs, hiddeninpts):
    outs_x, outs_h = [],[]
    import multiprocessing
    pool = multiprocessing.Pool()
    args = [(inputs, hiddeninpts, b_x[layer], b_h[layer], w_x[layer], w_h[layer], n) for n in range(nneurons)]
    y_vars = sympy.symbols("%s" % ', '.join(map(str, ["l"+str(layer)+"x"+str(i) for i in range(nneurons)])))
    h_vars = sympy.symbols("%s" % ', '.join(map(str, ["l"+str(layer)+"hy"+str(i) for i in range(nneurons)])))
    ret = pool.map(buildneq_rnn, args)
    outs_x, outs_h = zip(*ret)
    outs_x = [sympy.Eq(y_vars[i], outs_x[i]) for i in range(nneurons)]
    outs_h = [sympy.Eq(h_vars[i], outs_h[i]) for i in range(nneurons)]
    return outs_x, outs_h, y_vars, h_vars
def buildneq_rnn(f):
    sys.setrecursionlimit(1024*64)
    inputs, hiddeninpts, b_x, b_h, w_x, w_h, n = f
    weights_x, weights_h = w_x[n], w_h[n]
    bias_x, bias_h = b_x[n], b_h[n]
    hsyms = [inputs[k]*weights_x[k] for k in range(len(inputs))] + [hiddeninpts[k]*weights_h[k] for k in range(len(hiddeninpts))]
    #ysyms = [inputs[k] * weights_x[k] for k in range(len(inputs))]
    #ysym = bias_x
    hsym = bias_x + bias_h
    #for eq in ysyms: ysym += eq
    for eq in hsyms: hsym += eq
    # apply relu
    #ysym = sympy.Piecewise((0,ysym<0),(ysym,ysym >= 0))
    hsym = sympy.Piecewise((0,hsym<0),(hsym,hsym >= 0))
    #print(ysym)
    print(hsym)
    return (hsym, hsym,)

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