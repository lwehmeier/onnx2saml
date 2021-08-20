import sympy2saml
import onnx2sympy
import argparse

def onnx2saml(onnxpath, inputVarNames, outputvarNames, quantize=False, quantData=None):
    layereqs, layerouts, modelins = onnx2sympy.ann2Equations(onnxpath, quantize=quantize, quantData=quantData)
    saml = sympy2saml.annEqsToSaml(layereqs, layerouts, modelins, inputVarNames, outputvarNames, isInt=quantize)
    return saml

if __name__ == '__main__':
    # set up argparse
    parser = argparse.ArgumentParser(description="Converts onnx ANN models for embedding in saml"+
                                                 "models for mode checking and fault injection. "+
                                                 "Currently limited to dense/linear layers with relu activation")
    parser.add_argument("-i", dest="filename", required=True, help="path to onnx model file",
                        metavar="FILE", type=str)
    parser.add_argument("-p", dest="outprefix", required=True, help="prefix for the generated output formulas"+
                        "(resulting saml model will use $(prefix)I for the output names, where I is the index of"+
                        "the output variable)", metavar="FILE", type=str)
    parser.add_argument("-q", dest="inprefix", required=True, help="prefix for the generated input formulas"+
                        "(resulting saml model will use $(prefix)I for referencing the input variables for the ann,"+
                        " where I is the index of the output variable)", metavar="FILE", type=str)
    parser.add_argument("-o", dest="outfile", required=False, help="optional path to file fr dumping the generated saml",
                        metavar="FILE", type=str)
    parser.add_argument("-Q", dest="quantize", required=False, help="whether to perform model quantization to uint8. " +
                        "Requires specification of quantisation data set as json in {\"data\": [batch, data]} format."
                        "Currently broken, see TODO",
                        type=bool)
    parser.add_argument("-z", dest="quantData", required=False, help="File to data for static quantization",
                        metavar="FILE", type=str)
    args = parser.parse_args()
    #setup config
    if args.quantize is not None and args.quantData is None:
        print("Static quantization requested, but no quantData file has been specified")

    quantize=args.quantize is not None and args.quantize
    #TODO: currently broken, the parameter layout in the onnx graph changes after quantization

    onnxpath = args.filename
    inputVarNames=[(args.inprefix + str(i)) for i in range(11)]
    outputvarNames=[(args.outprefix + str(i)) for i in range(5)]

    #run conversion

    saml = onnx2saml(onnxpath, inputVarNames, outputvarNames, quantize=quantize, quantData=args.quantData)
    print("PROCESSING OK")

    if args.outfile is None:
        print("Copy the following code into the ann stub in your generated saml model")
        print("//============================================================")
        print("//============================================================")
        print("//============================================================")
        print(saml)
        print("//============================================================")
        print("//============================================================")
        print("//============================================================")
    else:
        with open(args.outfile, "w") as f:
            f.write(saml)
            print("Writing saml to file: ", args.outfile)
