import sympy
import numpy as np

def createSaml_pre(inpts, inptmap):
    assert len(inpts) == len(inptmap)
    s = ""
    for i in range(len(inpts)):
        s+="formula double "+str(inpts[i])+" := " + str(inptmap[i]) +";\n"
    return s
def createSaml_rnnStates(h_ins, h_outs):
    s=""
    for i in range(len(h_ins)):
        for j in range(len(h_ins[i])):
            # create and initialise states
            s+=str(h_ins[i][j])+" : [ 0.0 .. 1.0 ] init 0.0"+";\n"
            # create update rules
    s += "true ->"
    for i in range(len(h_ins)):
        for j in range(len(h_ins[i])):
            s+=" ("+str(h_ins[i][j])+"' = "+str(h_outs[i][j])+") &"
    return s[:-2]+"\n;"
def createSaml_post(outps, outptmap):
    assert len(outps) == len(outptmap)
    s = ""
    for i in range(len(outps)):
        s+="formula double "+str(outptmap[i])+" := " + str(outps[i]) +";\n"
    return s
def rnnEqsToSaml(layereqs_x, layereqs_h, layerouts_x, layerouts_h, layerins_h, modelins, inputVarNames, outputvarNames, isInt=False):
    saml = createSaml_pre(modelins, inputVarNames)
    saml += createSaml_rnnStates(layerins_h, layerouts_h)
    for i in range(len(layereqs_x)):
        saml += "//begin layer "+str(i)+"\n"
        saml += layerToSaml(layereqs_x[i],layerouts_x[i])
        saml += layerToSaml(layereqs_h[i],layerouts_h[i])
        saml += "//end layer "+str(i)+"\n"
    saml += createSaml_post(layerouts_x[-1], outputvarNames)
    if isInt: #slightly hacke, pass data type argument to string generation
        # functions via class member in a future revision
        saml.replace("formula double ", "formula int ")
    return saml

def annEqsToSaml(eqs, vars, inpts, inptmap, outptmap, isInt=False):
    saml = createSaml_pre(inpts, inptmap)
    for i in range(len(eqs)):
        saml += "//begin layer "+str(i)+"\n"
        saml += layerToSaml(eqs[i],vars[i])
        saml += "//end layer "+str(i)+"\n"
    saml += createSaml_post(vars[-1], outptmap)
    if isInt: #slightly hacke, pass data type argument to string generation
        # functions via class member in a future revision
        saml.replace("formula double ", "formula int ")
    return saml
def layerToSaml(equations, vars):
    layerStr = ""
    for eq in equations:
        eqStr = ""
        assert eq.class_key()[2] == "Equality"
        assert eq.lhs in vars or eq.rhs in vars
        lhs=eq.lhs
        rhs = eq.rhs
        if eq.lhs in vars:
            eqStr+=eqToSaml(lhs,rhs)
        else:
            eqStr+=eqToSaml(rhs,lhs)
        layerStr += eqStr+";\n"
    return layerStr


def exprToSaml(exp):
#    assert exp.class_key()[2] in ["Piecewise","Equality","Add","Mul","Symbol","Number",
#                                  "StrictGreaterThan","GreaterThan","StrictLessThan",
#                                  "LessThan","",]
# python 3.10 only..
#    match exp.class_key()[2]:
#        case "Piecewise":
#            pass
#        case _:
#            assert False
    print("processing expression: ", exp.class_key()[2], exp)
    match = exp.class_key()[2]
    if match == "Piecewise":
        return piecewiseToSaml(exp)
    if match == "ITE":
        return iteToSaml(exp)
    elif match == "Equality":
        return eqToSaml(exp)
    elif match == "Add":
        return addToSaml(exp)
    elif match == "Mul":
        return mulToSaml(exp)
    elif match == "Symbol":
        return symbToSaml(exp)
    elif match == "Number":
        return numToSaml(exp)
    elif match == "StrictGreaterThan":
        return relToSaml(exp,">=")
    elif match == "GreaterThan":
        return relToSaml(exp,">")
    elif match == "StrictLessThan":
        return relToSaml(exp,"<=")
    elif match == "LessThan":
        return relToSaml(exp,"<")
    else:
        print(exp.class_key(),exp)
        assert False

def eqToSaml(exp):
    return eqToSaml(exp.lhs, exp.rhs)
def eqToSaml(lhs, rhs):
    return "formula double "+str(lhs)+" := " + exprToSaml(rhs)
def iteToSaml(exp):
    args=exp.args
    return "if ("+exprToSaml(args[0])+") then ("+exprToSaml(args[1])+") else ("+exprToSaml(args[2])+")"
def piecewiseToSaml(exp):
    #find default
    defaultArg = None
    for arg in exp.args:
        if arg[1].class_key()[2] == "BooleanTrue":
            defaultArg=arg
            break
    args = set(exp.args)
    args.remove(defaultArg)
    expStr = ""
    nbrackets = 0
    for arg in args:
        expStr += "if ("+exprToSaml(arg[1])+") then ("+exprToSaml(arg[0])+") else ("
        nbrackets+=1
    expStr+=exprToSaml(defaultArg[0])+("".join(")" for i in range(nbrackets)))
    return expStr
def addToSaml(exp):
    return "("+exprToSaml(exp.as_two_terms()[0])+")+("+exprToSaml(exp.as_two_terms()[1])+")"
def mulToSaml(exp):
    return "("+exprToSaml(exp.as_two_terms()[0])+")*("+exprToSaml(exp.as_two_terms()[1])+")"
def symbToSaml(exp):
    return str(exp)
def numToSaml(exp):
    return str(exp)
def relToSaml(exp, opSym):
    return exprToSaml(exp.lhs)+opSym+exprToSaml(exp.rhs)