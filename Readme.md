# ONNX2SAML converter

Simple tool to parse onnx ANN models and transform them into SAML for use in verification and mc-based fault injection.
Currently, onnx2saml encodes the translated ANN as saml formulas (int or double/real, depending on the chosen quantization option) which can be copied or imported in a saml project to have access to the ANN in the formal (system) model.
When used with the simulink verification framework, the generated formulas can be copy&pasted into the generated "stub" cmponents for the ANN reference blocks. To allow for seamless integration, set the input and output naming parameters accordingly, in the case of dive a typical onnx2saml call should look like this:
`onnx2saml.py -i path-to-model.onnx -q INPUT -p OUTPUT -o generated.saml`

## Demo
* use `create_model_pytorch.py` to generate and export a relatively simple 3-layer, 23 neuron model with 11 inputs and 5 outputs to generate a test model
* call `onnx2saml.py -i data/demo_model.onnx -q INPUT -p OUTPUT` to print the generated saml lines
* if you have access to dive, you can use the included simulink model (data/super.slx) to create a complete verifiably system model in which the ANN code can be pasted
 * currently, the n-d lookup is used as placeholder, as the custom ann blocks are non-public
* Alternatively, a pre-generated saml-file with an ANN-stub component is included in the data folder
