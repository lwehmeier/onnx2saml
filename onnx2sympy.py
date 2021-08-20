import onnx
from onnx import numpy_helper
import onnxruntime
import numpy as np
import symcomp
import sympy
import json
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

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
def ann2Equations(onnxpath, quantize=False, quantData=None):
    if quantize:
        quant_model(onnxpath, quantData)
        onnx_model = onnx.load_model(onnxpath+".quant.onnx")
        onnx.checker.check_model(onnx_model)
    else:
        onnx_model = onnx.load_model(onnxpath)
        onnx.checker.check_model(onnx_model)
    INTIALIZERS=onnx_model.graph.initializer
    Weight=[]
    for initializer in INTIALIZERS:
        W= numpy_helper.to_array(initializer)
        Weight.append(W)
    w = Weight[0::2]
    b = Weight[1::2]
    layereqs, layerouts, modelins = annEquationsFromWB(w,b)
    return layereqs, layerouts, modelins