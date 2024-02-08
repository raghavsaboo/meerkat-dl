from typing import Tuple
from itertools import zip_longest
import numpy as np

def unbroadcast(tensor: np.ndarray, original_tensor_shape: Tuple[int, ...], broadcast_shape: Tuple[int, ...]):
    def get_axes_to_be_summed(original_tensor_shape, broadcast_shape):
        axes_to_be_summed = []
        zipped = list(zip_longest(tuple(reversed(broadcast_shape)), tuple(reversed(original_tensor_shape)), fillvalue=None))
        for dim, (dim_broadcasted, dim_orig) in enumerate(reversed(zipped)):
            if dim_broadcasted!=dim_orig:
                axes_to_be_summed.append(dim)
        return tuple(axes_to_be_summed)

    if broadcast_shape is not None:
        axes_to_be_summed = get_axes_to_be_summed(original_tensor_shape, broadcast_shape)
        unbroadcasted_tensor = np.sum(tensor, axis=axes_to_be_summed)
    else:
        unbroadcasted_tensor = tensor
    return unbroadcasted_tensor