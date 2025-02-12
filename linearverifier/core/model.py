"""
This module defines the behavior of a neural network model

"""

from mpmath import mp

from linearverifier.core import ops
from linearverifier.core.layer import LinearLayer
from linearverifier.parser import onnx
from linearverifier.parser import vnnlib


class ModelOptions:
    pass


class LinearModel:
    def __init__(self, onnx_path: str, options: ModelOptions = None):
        self.onnx_path = onnx_path
        self.options = options

        self.layer = self.parse_layer()

    @staticmethod
    def check_robust(lbs: mp.matrix, ubs: mp.matrix, label: int) -> bool:
        """Procedure to check whether the robustness specification holds"""
        correct = lbs[label]
        others = ops.get_other_ubs(ubs, label)

        # Naive check - very unlikely
        if ops.max_upper(others) < correct:
            return True
        else:
            # Create the matrices of the disjunctions
            out_props = ops.create_disjunction_matrix(len(lbs), label)
            bounds = {
                'lower': lbs,
                'upper': ubs
            }

            # For each disjunction in the output property, check none is satisfied by output_bounds.
            # If one disjunction is satisfied, then it represents a potential counter-example.
            for i in range(len(others)):
                result = ops.check_unsafe(bounds, out_props[i])

                if result == 1:
                    return False
                elif result == -1:
                    return True
                else:
                    print('Unknown')
                    return False

            return True

    def parse_layer(self) -> LinearLayer:
        """Procedure to read the first layer of a ONNX network"""
        nn = onnx.to_nn(self.onnx_path)
        return nn[0]

    def propagate(self, lbs: mp.matrix, ubs: mp.matrix) -> tuple[mp.matrix, mp.matrix]:
        """Procedure to compute the numeric interval bounds of a linear layer"""
        weights_plus = ops.get_positive(self.layer.weight)
        weights_minus = ops.get_negative(self.layer.weight)

        low = weights_plus * lbs + weights_minus * ubs + self.layer.bias
        upp = weights_plus * ubs + weights_minus * lbs + self.layer.bias

        return low, upp

    def verify(self, vnnlib_path: str) -> bool:
        # 1: Read VNNLIB bounds
        in_lbs, in_ubs, label = vnnlib.read_vnnlib(vnnlib_path)

        # 2: Propagate input through linear layer
        out_lbs, out_ubs = self.propagate(in_lbs, in_ubs)

        # 4: Check output
        return LinearModel.check_robust(out_lbs, out_ubs, label)
