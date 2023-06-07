import gc
import os
from copy import deepcopy
from typing import Union, List

import numpy as np
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from batchgenerators.utilities.file_and_folder_operations import load_json, isfile, save_pickle
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager


def export_prediction_from_softmax(predicted_array_or_file: Union[np.ndarray, str], properties_dict: dict,
                                   configuration_manager: ConfigurationManager,
                                   plans_manager: PlansManager,
                                   dataset_json_dict_or_file: Union[dict, str], output_file_truncated: str,
                                   save_probabilities: bool = False):
    if isinstance(predicted_array_or_file, str):
        tmp = deepcopy(predicted_array_or_file)
        if predicted_array_or_file.endswith('.npy'):
            predicted_array_or_file = np.load(predicted_array_or_file)
        elif predicted_array_or_file.endswith('.npz'):
            predicted_array_or_file = np.load(predicted_array_or_file)['softmax']
        os.remove(tmp)
    print("origin unique:", np.max(predicted_array_or_file), np.min(predicted_array_or_file))
    predicted_array_or_file = predicted_array_or_file.astype(np.float16)
    print("origin unique:", np.max(predicted_array_or_file), np.min(predicted_array_or_file))
    print("PAF shape:", predicted_array_or_file.shape)
    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    # resample to original shape
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *configuration_manager.spacing]

    predicted_array_or_file = configuration_manager.resampling_fn_probabilities(predicted_array_or_file,
                                            properties_dict['shape_after_cropping_and_before_resampling'],
                                            current_spacing,
                                            properties_dict['spacing'])
    print("After getting prob:", predicted_array_or_file.shape)
    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    print("predicted unique:", np.max(predicted_array_or_file), np.min(predicted_array_or_file))

    # segmentation = predicted_array_or_file
    # put result in bbox (revert cropping)
    print(properties_dict['shape_before_cropping'])
    segmentation_reverted_cropping = np.zeros((3,) + properties_dict['shape_before_cropping'][:], dtype=np.float16)
    print("rc shape is:", segmentation_reverted_cropping.shape)
    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    print("slicer is:", slicer)
    for i in range(0,3):
        segmentation = label_manager.convert_logits_to_segmentation(predicted_array_or_file[i:i+1,:,:,:])
        print("Segmentation unique:", np.max(segmentation), np.min(segmentation))
        segmentation_reverted_cropping[i][slicer] = segmentation
        # revert transpose
        # segmentation_reverted_cropping[i] = segmentation_reverted_cropping[i].transpose(plans_manager.transpose_backward)

    # save
    if save_probabilities:
        # probabilities are already resampled

        # apply nonlinearity
        predicted_array_or_file = label_manager.apply_inference_nonlin(predicted_array_or_file)

        # revert cropping
        probs_reverted_cropping = label_manager.revert_cropping(predicted_array_or_file,
                                                                properties_dict['bbox_used_for_cropping'],
                                                                properties_dict['shape_before_cropping'])
        # $revert transpose
        probs_reverted_cropping = probs_reverted_cropping.transpose([0] + [i + 1 for i in
                                                                           plans_manager.transpose_backward])
        np.savez_compressed(output_file_truncated + '.npz', probabilities=probs_reverted_cropping)
        save_pickle(properties_dict, output_file_truncated + '.pkl')
        del probs_reverted_cropping
        gc.collect()
    print("Final prediction:", np.max(predicted_array_or_file), np.min(predicted_array_or_file))
    del predicted_array_or_file
    gc.collect()
    rw = plans_manager.image_reader_writer_class()
    rw.write_seg(segmentation_reverted_cropping, output_file_truncated + dataset_json_dict_or_file['file_ending'],
                 properties_dict)
    del segmentation_reverted_cropping, segmentation
    gc.collect()


def resample_and_save(predicted: Union[str, np.ndarray], target_shape: List[int], output_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager, properties_dict: dict,
                      dataset_json_dict_or_file: Union[dict, str], next_configuration: str) -> None:
    # needed for cascade
    if isinstance(predicted, str):
        assert isfile(predicted), "If isinstance(segmentation_softmax, str) then " \
                                  "isfile(segmentation_softmax) must be True"
        del_file = deepcopy(predicted)
        predicted = np.load(predicted)
        os.remove(del_file)

    predicted = predicted.astype(np.float16)

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)

    # resample to original shape
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *configuration_manager.spacing]
    target_spacing = configuration_manager.spacing if len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [properties_dict['spacing'][0], *configuration_manager.spacing]
    print()
    predicted_array_or_file = configuration_manager.resampling_fn_probabilities(predicted,
                                                                                target_shape,
                                                                                current_spacing,
                                                                                target_spacing)
    del predicted
    gc.collect()

    # create segmentation (argmax, regions, etc)
    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    segmentation = label_manager.convert_logits_to_segmentation(predicted_array_or_file)
    #segmentation = predicted_array_or_file
    np.savez_compressed(output_file, seg=segmentation.astype(np.float16))
    del segmentation
    gc.collect()
