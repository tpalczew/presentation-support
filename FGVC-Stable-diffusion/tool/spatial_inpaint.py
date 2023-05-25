from __future__ import absolute_import, division, print_function, unicode_literals
import os
import cv2
import numpy as np
import torch
from PIL import Image


def spatial_inpaint(deepfill, mask, video_comp):
    keyFrameInd = np.argmax(np.sum(np.sum(mask, axis=0), axis=0))
    with torch.no_grad():
        img_res = deepfill.forward(video_comp[:, :, :, keyFrameInd] * 255., mask[:, :, keyFrameInd]) / 255.
    video_comp[mask[:, :, keyFrameInd], :, keyFrameInd] = img_res[mask[:, :, keyFrameInd], :]
    mask[:, :, keyFrameInd] = False

    return mask, video_comp


def sd_inpaint(pipe, mask, video_comp):
    keyFrameInd = np.argmax(np.sum(np.sum(mask, axis=0), axis=0))
    prompt = "high quality dslr photo, street, sidewalk, city"
    mask_tmp = mask[:, :, keyFrameInd]
    final_mask_tmp = np.where(np.expand_dims(mask_tmp, 2), [255, 255, 255], [0,0,0] ) 
    frame_tmp = video_comp[:, :, :, keyFrameInd] * 255.
    frame_tmp = np.array(frame_tmp).astype(np.uint8)
    final_mask = Image.fromarray( np.array(final_mask_tmp).astype(np.uint8)  ).convert("RGB").resize( (512,512) )
    final_frame = Image.fromarray( np.array(frame_tmp).astype(np.uint8)  ).convert("RGB").resize( (512,512) )
    inpt = pipe(prompt=prompt, image=final_frame, mask_image=final_mask).images[0]
    inpt = inpt.resize( (1920, 1080) )
    inpt = np.array(inpt)
    video_comp[mask[:,:, keyFrameInd], :, keyFrameInd] = inpt[mask[:, :, keyFrameInd], :]
    mask[:, :, keyFrameInd] = False
    return mask, video_comp