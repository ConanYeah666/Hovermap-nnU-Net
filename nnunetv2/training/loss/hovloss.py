import tifffile as tif
import torch
from nnunetv2.training.loss.dice import SoftDiceLoss, MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss, TopKLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1
from torch import nn
import numpy as np
from typing import Callable
import torch.nn.functional as F

class HovLoss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, weight_ce=1, weight_dice=1, ignore_label=None):
        """
        Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(HovLoss, self).__init__()
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label

        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.ignore_label = ignore_label

        self.mse = MSE_LOSS(**ce_kwargs)
        self.msge = MSGE_LOSS(**soft_dice_kwargs)



    def forward(self, net_output: torch.Tensor, target: torch.Tensor, focus: torch.Tensor):

        mse_loss = self.mse(target, net_output)
        msge_loss = self.msge(target, net_output, focus)

        result =  mse_loss + 2* msge_loss
        print("Current batch mse is:", result)
        return result

class MSGE_LOSS(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True,
                 smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(MSGE_LOSS, self).__init__()

        #im = tif.imread('/Users/17914/phd/nnUNet-master/nnunetv2/media/fabian/nnUNet_raw/label_new/Glom_2_label.tif')
        #self.focus = torch.from_numpy(im[im == 1])
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, true, pred, focus):
        """Calculate the mean squared error of the gradients of
        horizontal and vertical map predictions. Assumes
        channel 0 is Vertical and channel 1 is Horizontal.

        Args:
            true:  ground truth of combined horizontal
                   and vertical maps
            pred:  prediction of combined horizontal
                   and vertical maps
            focus: area where to apply loss (we only calculate
                    the loss within the nuclei)

        Returns:
            loss:  mean squared error of gradients

        """

        def get_3d_sobel_kernel():
            """Get sobel kernel with a given size."""

            base_filter = np.multiply(1 / 4, np.array([
                [0, 0, 0],
                [-1, 0, 0],
                [-1, 0, 0]
            ]))

            sobel_x = np.multiply(1 / 4, np.array([
                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],
                [[0, 0, 0],
                 [-4, 0, 4],
                 [0, 0, 0]],
                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]]
            ]))

            sobel_y = np.multiply(1 / 4, np.array([
                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],
                [[0, -4, 0],
                 [0, 0, 0],
                 [0, 4, 0]],
                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]]
            ]))

            sobel_z = np.multiply(1 / 4, np.array([
                [[0, 0, 0],
                 [0, -4, 0],
                 [0, 0, 0]],
                [[0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]],
                [[0, 0, 0],
                 [0,4, 0],
                 [0, 0, 0]]
            ]))

            return  torch.tensor(sobel_x, device='cuda:0',dtype=torch.float32), \
                    torch.tensor(sobel_y, device='cuda:0',dtype=torch.float32), \
                    torch.tensor(sobel_z, device='cuda:0',dtype=torch.float32)

        ####
        def get_gradient_hvf(hvf):
            """For calculating gradient."""
            kernel_h, kernel_v, kernel_f = get_3d_sobel_kernel()
            kernel_h = kernel_h.view(1, 1, *kernel_h.shape)  # constant
            kernel_v = kernel_v.view(1, 1, *kernel_v.shape)  # constant
            kernel_f = kernel_v.view(1, 1, *kernel_f.shape)  # constant

            h_ch = hvf[:,0:1,:,:,:]
            v_ch = hvf[:,1:2,:,:,:]
            f_ch = hvf[:,2:3,:,:,:]
            print(h_ch.shape)
            print(v_ch.shape)
            print(f_ch.shape)

            # can only apply in NCHW mode
            h_dh_ch = F.conv3d(h_ch, kernel_h, padding=1)
            v_dv_ch = F.conv3d(v_ch, kernel_v, padding=1)
            f_df_ch = F.conv3d(f_ch, kernel_f, padding=1)
            dhvf = torch.cat([h_dh_ch, v_dv_ch, f_df_ch], dim=1)
            dhvf = dhvf.permute(0, 2, 3, 4, 1).contiguous()  # to NHWC
            return dhvf

        shp_x, shp_y = true.shape, pred.shape
        """print(shp_x, shp_y)
        print(torch.unique(pred[:, 0, :, :, :]))
        print(torch.unique(pred[:, 1, :, :, :]))
        print(torch.unique(pred[:, 2, :, :, :]))
        print(torch.unique(true[:, 0, :, :, :]))
        print(torch.unique(true[:, 1, :, :, :]))
        print(torch.unique(true[:, 2, :, :, :]))"""

        print("before cat shape:", focus.shape)
        """true1 = true.cpu().numpy()
        focus1 = focus.cpu().numpy()
        pred1 = pred.cpu().detach().numpy()
        import tifffile as tif
        tif.imwrite(
            "/data/zucksliu/Glom-Segmnentation/media/fabian/nnuNet_results/Dataset040_GlomHovTest/nnUNetTrainer_20epochs__nnUNetPlans__3d_fullres/test/loss_seg1.tif",
            true1)
        tif.imwrite(
            "/data/zucksliu/Glom-Segmnentation/media/fabian/nnuNet_results/Dataset040_GlomHovTest/nnUNetTrainer_20epochs__nnUNetPlans__3d_fullres/test/loss_focus1.tif",
            focus1)
        tif.imwrite(
            "/data/zucksliu/Glom-Segmnentation/media/fabian/nnuNet_results/Dataset040_GlomHovTest/nnUNetTrainer_20epochs__nnUNetPlans__3d_fullres/test/loss_pred1.tif",
            pred1)
        exit()"""
        focus = focus.permute(0, 2, 3, 4, 1).contiguous()
        print("After cat shape:",focus.shape)

        true_grad = get_gradient_hvf(true)
        pred_grad = get_gradient_hvf(pred)
        print("loss shape:", true_grad.shape)
        print("focus shape:", focus.shape)
        loss = pred_grad - true_grad
        loss = focus * (loss * loss)
        print(torch.max(focus), torch.min(focus))
        print(torch.max(loss), torch.min(loss))
        print("Focus.sum is:", focus.sum())
        lam = 2
        # artificial reduce_mean with focused region
        loss = loss.sum() / (focus.sum() + 1.0e-8)
        return lam * loss

class MSE_LOSS(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True,
                 smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(MSE_LOSS, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, true, pred):
        """Calculate mean squared error loss.

        Args:
            true: ground truth of combined horizontal
                  and vertical maps
            pred: prediction of combined horizontal
                  and vertical maps

        Returns:
            loss: mean squared error

        """
        loss = pred - true
        loss = (loss * loss).mean()
        return loss

# TODO: add mes loss
# add msge loss, currently hovernet use conv2d, you need to change it to conv3d, and need to check the
# corresponding sobel filter,
# e.g., in 2d cases, the sobel filter in x direction, is [[0, 0, 0], [-1, 0, 1], [0,0,0]]
# you need to expand to 3d, you can ask GPT to help. Also, you need to add the axial direction's sobel filter, they
# should be similar, but need to check.

# then your Hover loss class should take (x, y, focus) as input.
# x (or prediction) shape: torch.Size([bs, 3, x, y, z]), need to check why x channel is currently 4.
# y should be same shape.

# loss = mse_loss(x, y) + # you can directly calculate for all three channel in one function
# lambda * msge(x, y, focus) # here you need to operate sobel filter for each directions, and also, you need to generate a mask
# called focus where the Nuclei position should be 1 and the other position should be zero. You only calculate
# msge loss on the region that focus[position] = 1. You can check the hovernet implementation.

# The final return loss should be a scalar, you may check the lambad choice of the original Hovernet, and follow their
# option.
# If this is done, you can then think about how to insert your hoverloss back to the nnUNet.
