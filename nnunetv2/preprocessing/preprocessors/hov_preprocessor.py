#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import shutil
from typing import Union, Tuple

import nnunetv2
import numpy as np
from acvl_utils.miscellaneous.ptqdm import ptqdm
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import get_identifiers_from_splitted_dataset_folder, \
    create_lists_from_splitted_dataset_folder
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor

class HovPreprocessor(DefaultPreprocessor):

    def run_case(self, image_files: List[str], seg_files: Union[str, None], focus_files: Union[str, None], plans_manager: PlansManager,
                 configuration_manager: ConfigurationManager,
                 dataset_json: Union[dict, str]):
        """
        seg file can be none (test cases)

        order of operations is: transpose -> crop -> resample
        so when we export we need to run the following order: resample -> crop -> transpose (we could also run
        transpose at a different place, but reverting the order of operations done during preprocessing seems cleaner)
        """
        if isinstance(dataset_json, str):
            dataset_json = load_json(dataset_json)

        rw = plans_manager.image_reader_writer_class()

        # load image(s)
        data, data_properites = rw.read_images(image_files)
        data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        original_spacing = [data_properites['spacing'][i] for i in plans_manager.transpose_forward]
        shape_before_cropping = data.shape[1:]
        data_properites['shape_before_cropping'] = shape_before_cropping
        data, seg_worng, bbox = crop_to_nonzero(data)
        data_properites['bbox_used_for_cropping'] = bbox
        # print(data.shape, seg.shape)
        data_properites['shape_after_cropping_and_before_resampling'] = data.shape[1:]
        # resample
        target_spacing = configuration_manager.spacing  # this should already be transposed
        if len(target_spacing) < len(data.shape[1:]):
            # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
            # in 3d we do not change the spacing between slices
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)
        #print(new_shape)
        # normalize
        # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
        # longer fitting the images perfectly!
        data = self._normalize(data, None, configuration_manager,
                               plans_manager.foreground_intensity_properties_per_channel)

        # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
        #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
        old_shape = data.shape[1:]
        data = configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)
        print(data.shape)
        if self.verbose:
            print(f'old shape: {old_shape}, new_shape: {new_shape}, old_spacing: {original_spacing}, '
                  f'new_spacing: {target_spacing}, fn_data: {configuration_manager.resampling_fn_data}')
        # if possible, load seg
        seg_res = np.empty(data.shape)
        print("seg files are:" ,seg_files)
        if seg_files is not None:
            for idx, seg_file in enumerate(seg_files):
                if seg_file is not None:
                    seg, _ = rw.read_seg(seg_file)
                else:
                    seg = None
                print(seg.shape)
                print(_.keys())
                # apply transpose_forward, this also needs to be applied to the spacing!
                if seg is not None:
                    seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])

                # crop, remember to store size before cropping!
                print("start doing configuration")
                # this command will generate a segmentation. This is important because of the nonzero mask which we may need
                seg = configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing)
                print(seg.shape)

                # if we have a segmentation, sample foreground locations for oversampling and add those to properties
                if seg_file is not None:
                    # reinstantiating LabelManager for each case is not ideal. We could replace the dataset_json argument
                    # with a LabelManager Instance in this function because that's all its used for. Dunno what's better.
                    # LabelManager is pretty light computation-wise.
                    label_manager = plans_manager.get_label_manager(dataset_json)
                    collect_for_this = label_manager.foreground_regions if label_manager.has_regions \
                        else label_manager.foreground_labels

                    # when using the ignore label we want to sample only from annotated regions. Therefore we also need to
                    # collect samples uniformly from all classes (incl background)
                    if label_manager.has_ignore_label:
                        collect_for_this.append(label_manager.all_labels)
                    # no need to filter background in regions because it is already filtered in handle_labels
                    # print(all_labels, regions)
                    data_properites['class_locations'] = self._sample_foreground_locations(seg, collect_for_this,
                                                                                           verbose=self.verbose)
                    seg = self.modify_seg_fn(seg, plans_manager, dataset_json, configuration_manager)

                seg = seg.astype(np.float16)
                print(seg.shape)
                seg_res[idx] = seg
                print("seg_res: shape:", seg_res.shape)


        focus_res = np.empty(data.shape)
        print("Focus file is:", focus_files)
        if focus_files is not None:
            for idx, focus_file in enumerate(focus_files):
                if focus_file is not None:
                    focus, _ = rw.read_seg(focus_file)
                else:
                    focus = None
                print("1", focus.shape)
                print(_.keys())
                # apply transpose_forward, this also needs to be applied to the spacing!
                if focus is not None:
                    focus = focus.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])

                # crop, remember to store size before cropping!
                print("start doing configuration")
                # this command will generate a segmentation. This is important because of the nonzero mask which we may need
                focus = configuration_manager.resampling_fn_seg(focus, new_shape, original_spacing, target_spacing)
                print("2", focus.shape)

                # if we have a segmentation, sample foreground locations for oversampling and add those to properties
                if focus_file is not None:
                    # reinstantiating LabelManager for each case is not ideal. We could replace the dataset_json argument
                    # with a LabelManager Instance in this function because that's all its used for. Dunno what's better.
                    # LabelManager is pretty light computation-wise.
                    label_manager = plans_manager.get_label_manager(dataset_json)
                    collect_for_this = label_manager.foreground_regions if label_manager.has_regions \
                        else label_manager.foreground_labels

                    # when using the ignore label we want to sample only from annotated regions. Therefore we also need to
                    # collect samples uniformly from all classes (incl background)
                    if label_manager.has_ignore_label:
                        collect_for_this.append(label_manager.all_labels)
                    # no need to filter background in regions because it is already filtered in handle_labels
                    # print(all_labels, regions)
                    data_properites['class_locations'] = self._sample_foreground_locations(focus, collect_for_this,
                                                                                           verbose=self.verbose)
                    focus = self.modify_seg_fn(focus, plans_manager, dataset_json, configuration_manager)

                focus = focus.astype(np.int16)
                print("3", focus.shape)
                print("One loop done")
            focus_res[0] = focus
            focus_res[1] = focus
            focus_res[2] = focus
            print("focus_res: shape:", focus_res.shape)

            print(data.shape)
            print(seg_res.shape)
            print(focus_res.shape)
        return data, seg_res, focus_res, data_properites

    def run_case_save(self, output_filename_truncated: str, image_files: List[str], seg_file: str, focus_files: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                      dataset_json: Union[dict, str]):
        data, seg, focus, properties = self.run_case(image_files, seg_file, focus_files, plans_manager, configuration_manager, dataset_json)
        # print('dtypes', data.dtype, seg.dtype)
        print("Start Compressed")
        print(seg.shape)
        # tif.imwrite("C:/Users/17914/phd/nnUNet-master/nnunetv2/media/fabian/nnUNet_raw/cur/Glom_000_beforecompress.tif", seg)
        np.savez_compressed(output_filename_truncated + '.npz', data=data, seg=seg, focus=focus)
        del data, seg, focus
        import gc
        gc.collect()
        print("Done compressed")
        write_pickle(properties, output_filename_truncated + '.pkl')

    def _normalize(self, data: np.ndarray, seg: np.ndarray, configuration_manager: ConfigurationManager,
                   foreground_intensity_properties_per_channel: dict) -> np.ndarray:
        for c in range(data.shape[0]):
            scheme = configuration_manager.normalization_schemes[c]
            normalizer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "preprocessing", "normalization"),
                                                           scheme,
                                                           'nnunetv2.preprocessing.normalization')
            if normalizer_class is None:
                raise RuntimeError('Unable to locate class \'%s\' for normalization' % scheme)
            normalizer = normalizer_class(use_mask_for_norm= False,
                                          intensityproperties=foreground_intensity_properties_per_channel[str(c)])
            data[c] = normalizer.run(data[c], None)
        return data

    def run(self, dataset_name_or_id: Union[int, str], configuration_name: str, plans_identifier: str,
            num_processes: int):
        """
        data identifier = configuration name in plans. EZ.
        """
        dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)

        assert isdir(join(nnUNet_raw, dataset_name)), "The requested dataset could not be found in nnUNet_raw"

        plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + '.json')
        assert isfile(plans_file), "Expected plans file (%s) not found. Run corresponding nnUNet_plan_experiment " \
                                   "first." % plans_file
        plans = load_json(plans_file)
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration(configuration_name)

        if self.verbose:
            print(f'Preprocessing the following configuration: {configuration_name}')
        if self.verbose:
            print(configuration_manager)

        dataset_json_file = join(nnUNet_preprocessed, dataset_name, 'dataset.json')
        dataset_json = load_json(dataset_json_file)

        identifiers = get_identifiers_from_splitted_dataset_folder(join(nnUNet_raw, dataset_name, 'imagesTr'),
                                                               dataset_json['file_ending'])
        output_directory = join(nnUNet_preprocessed, dataset_name, configuration_manager.data_identifier)

        if isdir(output_directory):
            shutil.rmtree(output_directory)

        maybe_mkdir_p(output_directory)

        output_filenames_truncated = [join(output_directory, i) for i in identifiers]

        file_ending = dataset_json['file_ending']
        # list of lists with image filenames
        image_fnames = create_lists_from_splitted_dataset_folder(join(nnUNet_raw, dataset_name, 'imagesTr'), file_ending,
                                                                 identifiers)
        print(image_fnames)
        # list of segmentation filenames
        seg_fnames = [[join(nnUNet_raw, dataset_name, 'labelsTr', i + j + file_ending) for j in ['_x','_y','_z']] for i in identifiers]
        print(seg_fnames)
        focus_fnames = [[join(nnUNet_raw, dataset_name, 'focus', i + file_ending)] for i in identifiers]
        print(focus_fnames)

        _ = ptqdm(self.run_case_save, (output_filenames_truncated, image_fnames, seg_fnames, focus_fnames),
                  processes=num_processes, zipped=True, plans_manager=plans_manager,
                  configuration_manager=configuration_manager,
                  dataset_json=dataset_json, disable=self.verbose)