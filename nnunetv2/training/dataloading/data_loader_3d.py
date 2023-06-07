import gc

import numpy as np
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset


class nnUNetDataLoader3D(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        print("Start generate batch")
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float16)
        seg_all = np.zeros(self.seg_shape, dtype=np.float16)
        focus_all = np.zeros(self.focus_shape, dtype=np.float16)
        case_properties = []
        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            print("current key:", j, i)
            force_fg = self.get_do_oversample(j)
            print("Done oversample")
            data, seg, focus, properties = self._data.load_case(i)
            #print(self._data)
            print("Fistr focues: ", np.max(focus), np.min(focus))
            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            print("Done getting box")
            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]


            print("Start padding")
            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]
            print("Start data")
            data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            print("After data")
            del data
            gc.collect()
            this_slice = tuple([slice(0, seg .shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            #print("Seg shape slice1:", seg.shape)
            #print(this_slice)
            seg = seg[this_slice]
            print("Seg shape slice2:", seg.shape)
            seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=0)
            del seg, this_slice
            gc.collect()
            this_slice = tuple([slice(0, focus.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            print("before focues: ", np.max(focus), np.min(focus))
            # print("focus slice:", this_slice)
            focus = focus[this_slice]
            #print("After focues: ", np.max(focus), np.min(focus))
            print(focus.shape)
            focus_all[j] = np.pad(focus, ((0, 0), *padding), 'constant', constant_values=0)
            print("After padding: ", np.max(focus), np.min(focus))
            del padding, focus
            gc.collect()
            del shape,this_slice
            gc.collect()
            # print(np.unique(seg_all))
            # print("Done all the data")

        return {'data': data_all, 'seg': seg_all, 'focus': focus_all, 'properties': case_properties, 'keys': selected_keys}


if __name__ == '__main__':
    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/3d_fullres'
    ds = nnUNetDataset(folder, 0)  # this should not load the properties!
    dl = nnUNetDataLoader3D(ds, 5, (16, 16, 16), (16, 16, 16), 0.33, None, None)
    a = next(dl)
