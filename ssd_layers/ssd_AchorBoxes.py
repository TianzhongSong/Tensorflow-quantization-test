"""
reference: https://github.com/pierluigiferrari/ssd_keras/blob/master/keras_layers/keras_layer_AnchorBoxes.py
"""
import tensorflow as tf
import numpy as np
from .bounding_box_utils import convert_coordinates
from keras import backend as K


def AnchorBoxes(x,
                img_height,
                img_width,
                this_scale,
                next_scale,
                aspect_ratios=[0.5, 1.0, 2.0],
                two_boxes_for_ar1=True,
                this_steps=None,
                this_offsets=None,
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                coords='centroids',
                normalize_coords=False,
                batch_size=1):
    '''
        A TensorFlow layer to create an output tensor containing anchor box coordinates
        and variances based on the input tensor and the passed arguments.
        A set of 2D anchor boxes of different aspect ratios is created for each spatial unit of
        the input tensor. The number of anchor boxes created per unit depends on the arguments
        `aspect_ratios` and `two_boxes_for_ar1`, in the default case it is 4. The boxes
        are parameterized by the coordinate tuple `(xmin, xmax, ymin, ymax)`.
        The logic implemented by this layer is identical to the logic in the module
        `ssd_box_encode_decode_utils.py`.
        The purpose of having this layer in the network is to make the model self-sufficient
        at inference time. Since the model is predicting offsets to the anchor boxes
        (rather than predicting absolute box coordinates directly), one needs to know the anchor
        box coordinates in order to construct the final prediction boxes from the predicted offsets.
        If the model's output tensor did not contain the anchor box coordinates, the necessary
        information to convert the predicted offsets back to absolute coordinates would be missing
        in the model output. The reason why it is necessary to predict offsets to the anchor boxes
        rather than to predict absolute box coordinates directly is explained in `README.md`.
        Input shape:
            4D tensor of shape `(batch, channels, height, width)` if `dim_ordering = 'th'`
            or `(batch, height, width, channels)` if `dim_ordering = 'tf'`.
        Output shape:
            5D tensor of shape `(batch, height, width, n_boxes, 8)`. The last axis contains
            the four anchor box coordinates and the four variance values for each box.
    '''
    if (this_scale < 0) or (next_scale < 0) or (this_scale > 1):
        raise ValueError(
            "`this_scale` must be in [0, 1] and `next_scale` must be >0, but `this_scale` == {}, `next_scale` == {}".format(
                this_scale, next_scale))

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    # Compute the number of boxes per cell
    if (1 in aspect_ratios) and two_boxes_for_ar1:
        n_boxes = len(aspect_ratios) + 1
    else:
        n_boxes = len(aspect_ratios)

    # Compute box width and height for each aspect ratio
    # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.
    size = min(img_height, img_width)

    wh_list = []
    for ar in aspect_ratios:
        if ar == 1:
            # Compute the regular anchor box for aspect ratio 1.
            box_height = box_width = this_scale * size
            wh_list.append((box_width, box_height))
            if two_boxes_for_ar1:
                # Compute one slightly larger version using the geometric mean of this scale value and the next.
                box_height = box_width = np.sqrt(this_scale * next_scale) * size
                wh_list.append((box_width, box_height))
        else:
            box_height = this_scale * size / np.sqrt(ar)
            box_width = this_scale * size * np.sqrt(ar)
            wh_list.append((box_width, box_height))
    wh_list = np.array(wh_list)

    # We need the shape of the input tensor
    feature_map_height = x.get_shape().as_list()[1]
    feature_map_width = x.get_shape().as_list()[2]
    # Compute the grid of box center points. They are identical for all aspect ratios.
    # Compute the step sizes, i.e. how far apart the anchor box center points will be vertically and horizontally.
    if (this_steps is None):
        step_height = img_height / feature_map_height
        step_width = img_width / feature_map_width
    else:
        if isinstance(this_steps, (list, tuple)) and (len(this_steps) == 2):
            step_height = this_steps[0]
            step_width = this_steps[1]
        elif isinstance(this_steps, (int, float)):
            step_height = this_steps
            step_width = this_steps
    # Compute the offsets, i.e. at what pixel values the first anchor box center point will be from the top and from the left of the image.
    if (this_offsets is None):
        offset_height = 0.5
        offset_width = 0.5
    else:
        if isinstance(this_offsets, (list, tuple)) and (len(this_offsets) == 2):
            offset_height = this_offsets[0]
            offset_width = this_offsets[1]
        elif isinstance(this_offsets, (int, float)):
            offset_height = this_offsets
            offset_width = this_offsets

    # Now that we have the offsets and step sizes, compute the grid of anchor box center points.
    cy = np.linspace(offset_height * step_height, (offset_height + feature_map_height - 1) * step_height, feature_map_height)
    cx = np.linspace(offset_width * step_width, (offset_width + feature_map_width - 1) * step_width, feature_map_width)
    cx_grid, cy_grid = np.meshgrid(cx, cy)
    cx_grid = np.expand_dims(cx_grid, -1) # This is necessary for np.tile() to do what we want further down
    cy_grid = np.expand_dims(cy_grid, -1) # This is necessary for np.tile() to do what we want further down

    # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
    # where the last dimension will contain `(cx, cy, w, h)`
    boxes_tensor = np.zeros((feature_map_height, feature_map_width, n_boxes, 4))

    boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes))  # Set cx
    boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes))  # Set cy
    boxes_tensor[:, :, :, 2] = wh_list[:, 0]  # Set w
    boxes_tensor[:, :, :, 3] = wh_list[:, 1]  # Set h

    # Convert `(cx, cy, w, h)` to `(xmin, xmax, ymin, ymax)`
    boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='centroids2corners')

    # If `clip_boxes` is enabled, clip the coordinates to lie within the image boundaries
    if clip_boxes:
        x_coords = boxes_tensor[:, :, :, [0, 2]]
        x_coords[x_coords >= img_width] = img_width - 1
        x_coords[x_coords < 0] = 0
        boxes_tensor[:, :, :, [0, 2]] = x_coords
        y_coords = boxes_tensor[:, :, :, [1, 3]]
        y_coords[y_coords >= img_height] = img_height - 1
        y_coords[y_coords < 0] = 0
        boxes_tensor[:, :, :, [1, 3]] = y_coords

    # If `normalize_coords` is enabled, normalize the coordinates to be within [0,1]
    if normalize_coords:
        boxes_tensor[:, :, :, [0, 2]] /= img_width
        boxes_tensor[:, :, :, [1, 3]] /= img_height

    if coords == 'centroids':
        # Convert `(xmin, ymin, xmax, ymax)` back to `(cx, cy, w, h)`.
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2centroids',
                                           border_pixels='half')
    elif coords == 'minmax':
        # Convert `(xmin, ymin, xmax, ymax)` to `(xmin, xmax, ymin, ymax).
        boxes_tensor = convert_coordinates(boxes_tensor, start_index=0, conversion='corners2minmax',
                                           border_pixels='half')

    # Create a tensor to contain the variances and append it to `boxes_tensor`. This tensor has the same shape
    # as `boxes_tensor` and simply contains the same 4 variance values for every position in the last axis.
    variances_tensor = np.zeros_like(boxes_tensor)  # Has shape `(feature_map_height, feature_map_width, n_boxes, 4)`
    variances_tensor += variances  # Long live broadcasting
    # Now `boxes_tensor` becomes a tensor of shape `(feature_map_height, feature_map_width, n_boxes, 8)`
    boxes_tensor = np.concatenate((boxes_tensor, variances_tensor), axis=-1)

    # Now prepend one dimension to `boxes_tensor` to account for the batch size and tile it along
    # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 8)`
    boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
    # revised
    boxes_tensor = tf.tile(tf.constant(boxes_tensor, dtype='float32'), [batch_size, 1, 1, 1, 1])
    return boxes_tensor
