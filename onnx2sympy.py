import onnx
from onnx import numpy_helper
import onnxruntime
import numpy as np
import symcomp
import sympy
import json
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
from enum import Enum

class ANNType(Enum):
    FFDense = 1
    RNNDense = 2
    # add further ann types here

class CalDR(CalibrationDataReader):
    def __init__(self,quantDataPath, augmented_model_path='augmented_model.onnx'):
        self.augmented_model_path = augmented_model_path
        self.preprocess_flag = True
        self.enum_data_dicts = []
        self.datasize = 0
        self.quantDataPath = quantDataPath

    def get_next(self):
        if self.preprocess_flag:
            self.preprocess_flag = False
            session = onnxruntime.InferenceSession(self.augmented_model_path, None)
            (_, nin) = session.get_inputs()[0].shape
            with open(self.quantDataPath, "r") as f:
                js = json.load(f)
                nhwc_data_list = np.array(dict["data"]).astype(np.float32)
            input_name = session.get_inputs()[0].name
            self.datasize = len(nhwc_data_list)
            self.enum_data_dicts = iter([{input_name: nhwc_data} for nhwc_data in nhwc_data_list])
        return next(self.enum_data_dicts, None)

def quant_model(modelpath, quantDataPath):
    dr = CalDR(quantDataPath)
    quantize_static(modelpath,
                    modelpath+".quant.onnx",
                    dr)
    print('Calibrated and quantized model saved.')
# https://github.com/onnx/onnx/blob/master/docs/Operators.md#RNN for rnn computation as used by onnx (elman-like RNN)
def rnnEquationsFromWB(w_x, w_h,b_x, b_h):
    nneurons = [len(w_x[i]) for i in range(len(w_x))]
    inpts = sympy.symbols("%s" % ', '.join(map(str, ["x"+str(i) for i in range(len(w_x[0][0]))])))
    hiddeninpts = []
    for n in range(len(nneurons)):
        hiddeninpts.append(sympy.symbols("%s" % ', '.join(map(str, ["h"+str(i)+"l"+str(n) for i in range(nneurons[n])]))))

    l0eq_x, l0eq_h, l0vars_x, l0vars_h = symcomp.buildRnnLayerEq(w_x, w_h,b_x,b_h,0,nneurons[0],inpts, hiddeninpts[0])
    layereqs_x,layereqs_h = [l0eq_x],[l0eq_h]
    layerouts_x,layerouts_h = [l0vars_x],[l0vars_h]
    for l in range(1,len(nneurons)):
        leq_x, leq_h, lvar_x, lvar_h = symcomp.buildRnnLayerEq(w_x, w_h,b_x,b_h,l,nneurons[l],layerouts_x[l-1], hiddeninpts[l])
        layereqs_x.append(leq_h)
        layerouts_x.append(lvar_h)
        layereqs_h.append(leq_h)
        layerouts_h.append(lvar_h)
    return layereqs_x, layereqs_h, layerouts_x, layerouts_h, hiddeninpts, list(inpts)
def annEquationsFromWB(weights, bias):
    w,b = weights, bias
    nneurons = [len(w[i]) for i in range(len(w))]
    inpts = sympy.symbols("%s" % ', '.join(map(str, ["x"+str(i) for i in range(len(w[0][0]))])))

    l0eq, l0vars = symcomp.buildLayerEq(w,b,0,nneurons[0],inpts)
    layereqs = [l0eq]
    layerouts = [l0vars]
    for l in range(1,len(nneurons)):
        leq, lvar = symcomp.buildLayerEq(w,b,l,nneurons[l],layerouts[-1])
        layereqs.append(leq)
        layerouts.append(lvar)
    return layereqs, layerouts, list(inpts)
def ann2Equations(onnxpath, quantize=False, quantData=None, annType=ANNType.FFDense):
    if quantize:
        quant_model(onnxpath, quantData)
        onnx_model = onnx.load_model(onnxpath+".quant.onnx")
        onnx.checker.check_model(onnx_model)
    else:
        onnx_model = onnx.load_model(onnxpath)
        onnx.checker.check_model(onnx_model)
    if annType==ANNType.FFDense:
        INTIALIZERS=onnx_model.graph.initializer
        Weight=[]
        for initializer in INTIALIZERS: # for dense ff: weight, bias, weight, bias ...
                                        # for rnn: weight_x, weight_hidden, bias, ...
            W= numpy_helper.to_array(initializer)
            Weight.append(W)
        w = Weight[0::2]
        b = Weight[1::2]
        layereqs, layerouts, modelins = annEquationsFromWB(w,b)
        return layereqs, layerouts, modelins
    elif annType == ANNType.RNNDense:
        INTIALIZERS=onnx_model.graph.initializer
        Weight=[]
        for initializer in INTIALIZERS: # for dense ff: weight, bias, weight, bias ...
                                        # for rnn: weight_x, weight_hidden, bias, ...
            W= numpy_helper.to_array(initializer)
            W = np.squeeze(W)
            Weight.append(W)
        w_x = Weight[0::3]
        w_h = Weight[1::3]
        b = Weight[2::3]
        b = np.array([np.split(np.array(c),2) for c in b])
        b_x = np.squeeze(b[:,::2])
        b_h = np.squeeze(b[:,1::2])
        layereqs_x, layereqs_h, layerouts_x, layerouts_h, layerins_h, modelins = rnnEquationsFromWB(w_x, w_h,b_x, b_h)
        return layereqs_x, layereqs_h, layerouts_x, layerouts_h, layerins_h, modelins
    return None