import numpy as np
import tifffile as tif
import cv2
from scipy import ndimage
from scipy.ndimage import filters, measurements
from skimage.segmentation import watershed
from misc.utils import get_bounding_box, remove_small_objects
import gc
from scipy.stats import norm
import warnings
from skimage import feature
from scipy.ndimage.morphology import (
    binary_dilation,
    binary_fill_holes,
    distance_transform_cdt,
    distance_transform_edt,
)


def noop(*args, **kargs):
    pass


####
def __proc_np_hv(focus, hov):
    """Process Nuclei Prediction with XY Coordinate Map.

    Args:
        pred: prediction output, assuming
              channel 0 contain probability map of nuclei
              channel 1 containing the regressed X-map
              channel 2 containing the regressed Y-map

    """
    print("Start proc np hv")
    blb_raw = focus.astype(np.int16)
    hov = hov.astype(float)
    #hov2 = hov2.astype(float)
    #hov3 = hov3.astype(float)
    alpha = 0
    beta = 1
    h_dir_raw = hov[0, ...]
    h_dir = alpha + (h_dir_raw - h_dir_raw.min()) * (beta - alpha) / (h_dir_raw.max() - h_dir_raw.min())
    print("H get")
    sobelh = ndimage.sobel(h_dir, axis=2)
    del h_dir, h_dir_raw
    gc.collect()
    print("after h sobel")
    sobelh = 1 - (alpha + (sobelh - sobelh.min()) * (beta - alpha) / (sobelh.max() - sobelh.min()))
    print("Done h sobel")
    v_dir_raw = hov[1, ...]
    v_dir = alpha + (v_dir_raw - v_dir_raw.min()) * (beta - alpha) / (v_dir_raw.max() - v_dir_raw.min())
    print("V get")
    sobelv = ndimage.sobel(v_dir, axis=1)
    del v_dir, v_dir_raw
    gc.collect()
    print("after v sobel")

    sobelv = 1 - (alpha + (sobelv - sobelv.min()) * (beta - alpha) / (sobelv.max() - sobelv.min()))
    print("Done v sobel")
    f_dir_raw = hov[2, ...]
    f_dir = alpha + (f_dir_raw - f_dir_raw.min()) * (beta - alpha) / (f_dir_raw.max() - f_dir_raw.min())
    print("f get")

    sobelf = ndimage.sobel(f_dir, axis=0)
    del f_dir, f_dir_raw
    gc.collect()
    print("after f sobel")
    sobelf = 1 - (alpha + (sobelf - sobelf.min()) * (beta - alpha) / (sobelf.max() - sobelf.min()))
    print("Done f sobel")
    # processing
    blb = np.array(blb_raw == 1, dtype=np.int16)

    print("Done blb")

    overall = np.maximum(sobelh, sobelv, sobelf)
    tif.imwrite("/data/zucksliu/Glom-Segmnentation/media/fabian/post/G5overall.tif", overall)
    """mu = np.mean(overall)
    sigma = np.std(overall)

    # Create a Gaussian distribution object
    gaussian_dist = norm(mu, sigma)
    # Apply the Gaussian distribution to the array
    overall = gaussian_dist.cdf(overall)"""
    
    del sobelv, sobelh
    gc.collect()
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    #tif.imwrite("/data/zucksliu/Glom-Segmnentation/media/fabian/post/G35overall3.tif", overall)
    print("Done overall")
    dist = (1.0 - overall) * blb
    ## nuclei values form mountains so inverse to get basins
    print("Start guassian fileter")
    dist = -ndimage.gaussian_filter(dist, sigma=3, mode='constant', cval=0)
    #tif.imwrite("/data/zucksliu/Glom-Segmnentation/media/fabian/post/G3dist.tif", dist)
    print(np.count_nonzero(overall[overall >= 0.5]))
    print(np.count_nonzero(overall[overall >= 0.4]))
    print(np.count_nonzero(overall[overall >= 0.3]))
    print(np.count_nonzero(overall[overall >= 0.2]))
    overall = np.array(overall >= 0.47, dtype=np.int16)

    marker = blb - overall
    #tif.imwrite("/data/zucksliu/Glom-Segmnentation/media/fabian/post/G35marker1.tif", marker)
    marker[marker < 0] = 0

    print("Start marker 1")
    marker = binary_fill_holes(marker).astype("uint8")
    print("Start marker 2")
    pred_inst = np.zeros(marker.shape)
    radius = 10  # Approximate equivalent to a 5x5 elliptical structuring element
    for i, m in enumerate(marker):
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        print(m.shape)
        # marker[i] = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        selem = ndimage.generate_binary_structure(2, 1)
        kernel = ndimage.iterate_structure(selem, radius)
        # erforming morphology operation
        # marker = ndimage.binary_opening(marker, structure=kernel)
        # kernel = ndimage.generate_binary_structure(2, 1)
        marker[i] = ndimage.binary_opening(m, structure=kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=50)
    print("Start Final pred")
    pred_inst = watershed(dist, markers=marker, mask=blb, watershed_line=True)

    return pred_inst

print("Start tif read")
hov = tif.imread("/data/zucksliu/Glom-Segmnentation/media/fabian/nnuNet_results/Dataset043_6Hov/nnUNetTrainer_20epochs__nnUNetPlans__3d_lowres/fold_all/validation/Glom_002.tif")
#hov1 = tif.imread('D:/Glom_001.tif')
#hov2 = tif.imread('D:/Glom_001y.tif')
#hov3 = tif.imread('D:/Glom_001z.tif')
print("Start tif read2")
focus = tif.imread("/data/zucksliu/Glom-Segmnentation/media/fabian/nnuNet_results/Dataset041_GlomFullStack6Sample/nnUNetTrainer_20epochs__nnUNetPlans__3d_cascade_fullres/fold_all/validation/Glom_002.tif")
#focus = tif.imread('D:/Glom_001_raw.tif')
pred = __proc_np_hv(focus, hov)
print("Start tif write")
tif.imwrite("/data/zucksliu/Glom-Segmnentation/media/fabian/post/Glom_003_new.tif",pred)
#tif.imwrite('D:/Glom_test1.tif', pred)