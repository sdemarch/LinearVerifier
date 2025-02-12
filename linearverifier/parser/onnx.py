"""
This module reads ONNX files representing neural networks
"""

import onnx
from mpmath import mp

from linearverifier.core.layer import LinearLayer, Layer


def to_nn(onnx_path: str) -> list[Layer]:
    """Procedure to read a ONNX network as a list of Layer objects"""

    # Open model
    onnx_net = onnx.load(onnx_path)

    # Read data
    parameters = {}
    for initializer in onnx_net.graph.initializer:
        parameters[initializer.name] = onnx.numpy_helper.to_array(initializer)

    # Create layers
    net = []

    for node in onnx_net.graph.node:
        if node.op_type == 'Gemm':

            weight = parameters[node.input[1]]
            for att in node.attribute:
                if (att.name == 'transA' or att.name == 'transB') and att.i == 0:
                    weight = parameters[node.input[1]].T

            neurons = weight.shape[0]
            weight = mp.matrix(weight)

            if len(node.input) <= 2:
                bias = mp.zeros(neurons, 1)
            else:
                bias = mp.matrix(parameters[node.input[2]])

            net.append(LinearLayer(weight, bias))

    assert (len(net)) == 1
    return net
