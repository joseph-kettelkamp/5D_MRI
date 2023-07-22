# authors : Joseph Kettelkamp <joseph-kettelkamp@uiowa.edu>
# license : The University of Iowa
from typing import * 

import sys
import time
import os
import SimpleITK as itk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

gpu_id=1
device = torch.device(gpu_id)

class SpatialTransformer(nn.Module):
    #initializer
    def __init__(self,inshape,mode='bilinear'):
        super().__init__()
        
        self.mode = mode
        
        device = torch.device(gpu_id)
        vectors = [torch.arange(0, s) for s in inshape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.grid = grid.to(device)
        
    def forward(self,inp,flow):
        #print(inp.shape)
        #print(flow.shape)
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
            
            
        
        return F.grid_sample(inp, new_locs, align_corners=True, mode=self.mode)


class Interface_X(nn.Module):
    def __init__(self, x_init: Optional[torch.Tensor] = torch.zeros(1, 256, 256, 256).double()) -> None:
        super().__init__()
        
        print(x_init.shape)
        
        
        self.x = nn.Parameter(torch.view_as_real(torch.complex(torch.norm(x_init, p=2.0, dim=0).reshape([1, 256, 256, 256]), torch.zeros(1, 256, 256, 256).double())).permute([0, 4, 1, 2, 3]).float())
        
    def forward(self) -> torch.Tensor:
        return self.x
device = torch.device(gpu_id)

device = torch.device(gpu_id)

class Interface_gen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        d = 24
        
        self.conv_blocks = nn.Sequential(
            nn.Conv3d(3, 30, 1, 1, 0),
            #nn.BatchNorm3d(30),
            nn.Tanh(), # Finish Layer 1 --> 1x1x1
            nn.Conv3d(30, d*8, 3, 1, 1),
            #nn.BatchNorm3d(d*8),
            nn.LeakyReLU(0.2, inplace=True), # Finish Layer 2 --> 1x1x1
            nn.Upsample(scale_factor=2, mode='trilinear'), # NN --> 2x2x2
            nn.Conv3d(d*8, d*4, 3, 1, 1),
            #nn.BatchNorm3d(d*4),
            nn.LeakyReLU(0.2, inplace=True), # Finish Layer 3 --> 2x2x2
            nn.Upsample(scale_factor=2, mode='trilinear'), # NN --> 4x4x4
            nn.Conv3d(d*4, d*4, 3, 1, 1),
            #nn.BatchNorm3d(d*4),
            nn.LeakyReLU(0.2, inplace=True), # Finish Layer 4 --> 4x4x4
            nn.Upsample(scale_factor=2, mode='trilinear'), # NN --> 8x8x8
            nn.Conv3d(d*4, d*2, 3, 1, 1),
            #nn.BatchNorm3d(d*2),
            nn.LeakyReLU(0.2, inplace=True), # Finish Layer 5 --> 8x8x8
            nn.Upsample(scale_factor=2, mode='trilinear'), # NN --> 16x16x16
            nn.Conv3d(d*2, d, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True), # Finish Layer 6 --> 16x16x16
            nn.Upsample(scale_factor=2, mode='trilinear'), # NN --> 32x32x32
            nn.Conv3d(d, d, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True), # Finish Layer 7 --> 32x32x32
            nn.Upsample(scale_factor=2, mode='trilinear'), # NN --> 64x64x64
            nn.Conv3d(d, d, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True), # Finish Layer 7 --> 32x32x32
            nn.Conv3d(d, d, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True), # Finish Layer 7 --> 32x32x32
            nn.Conv3d(d, 3, 3, 1, 1),
            nn.Tanh()
            )

            
        self.trans = SpatialTransformer([256, 256, 256], mode='bilinear')
        self.flow = None
        self.x = nn.Parameter(torch.zeros([1, 2, 256, 256, 256]))
        self.upsample = nn.Upsample(size=[256, 256, 256], mode='trilinear')
        
    def weight_init(self):
    
        def init(layer):
            if isinstance(layer, nn.Conv3d):
                torch.nn.init.kaiming_normal_(layer.weight, a=0.2)
            elif isinstance(layer, nn.ConvTranspose3d):
                torch.nn.init.kaiming_normal_(layer.weight, a=0.2)
            
        self.conv_blocks.apply(init)
        
    def resize_output(self, size):
        self.trans = SpatialTransformer([size, size, size], mode='bilinear')
        self.upsample = nn.Upsample(size=[size, size, size])
        self.trans.to(torch.device(gpu_id))
        
    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        
        flow = (self.conv_blocks(z))
        flow = self.upsample(flow)
        flow = flow*4.0
        self.flow = flow
        #print(self.flow.shape)
        return self.trans(x, flow)

class gen_new(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        d = 18
        
        
        self.trans = SpatialTransformer([256, 256, 256], mode='bilinear')
        self.flow = None
        self.x = nn.Parameter(torch.zeros([1, 2, 256, 256, 256]))
        grid_x, grid_y, grid_z = torch.torch.meshgrid(torch.linspace(0, 1.0, 64), torch.linspace(0, 1.0, 64), torch.linspace(0, 1.0, 64))
        self.grid = torch.stack([grid_x, grid_y, grid_z], axis=0).to(torch.device(gpu_id)).reshape(1, 3, 64, 64, 64).to(torch.device(gpu_id))
        self.grid.requires_grad = True
        self.ln1 = nn.Linear(6, 256)
        self.ln2 = nn.Linear(256, 256)
        self.ln3 = nn.Linear(256, 256)
        self.ln4 = nn.Linear(256, 3)
        self.upsample = nn.Upsample(size=[256, 256, 256])
        
    def weight_init(self):
        pass
        
    def resize_output(self, size):
        self.trans = SpatialTransformer([size, size, size], mode='bilinear')
        self.upsample = nn.Upsample(size=[size, size, size])
        self.trans.to(torch.device(gpu_id))
        
    def forward(self, z: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        bz = z.shape[0]
        z_br = torch.ones([bz, 1, 64, 64, 64]).to(torch.device(gpu_id)) * z
        x = torch.stack([z_br, self.grid.tile((bz, 1, 1, 1, 1))], axis=1).reshape(bz, 6, -1).permute([2, 0, 1]).reshape(-1, 6)
        flow = F.relu(self.ln1(x))
        flow = F.relu(self.ln2(flow))
        flow = F.relu(self.ln3(flow))
        flow = F.tanh(self.ln4(flow))
        flow = flow*8.0
        self.flow = flow.reshape([-1, bz, 3]).permute([1, 2, 0]).reshape([bz, 3, 64, 64, 64])
        self.flow = self.upsample(self.flow)
        #print(self.flow.shape)
        return self.trans(img, self.flow)

GEN = Interface_gen()
TEMP = Interface_X()
    
def LOAD_GLOBALS(params)-> None:
    GEN.to(device)
    TEMP.to(device)
    GEN.load_state_dict(torch.load(params['generator_file']))
    GEN.eval()
    TEMP.load_state_dict(torch.load(params['template_file']))
    TEMP.eval()

class DICOMWriter:
    def __init__(self, number_of_frames: int, metadata_tags: List[Tuple[str, str]], dummy_Img: itk.Image, output_folder: str) -> None:
       
        self.frame_count = number_of_frames
        self.output_dir = output_folder
        self.metadata_tags = metadata_tags
        self.dummy_img = dummy_Img
        self.temp = None
        self.gen = None
        
    def load(self, params: Optional[List[Any]] = None):
        pass

    def unload(self):
        pass
    
    def write(self, t: int, latent_sub_space: torch.Tensor) -> int:
        # Set the necessary DICOM attributes for this volume
        
        volume_data = abs(torch.view_as_complex(GEN(latent_sub_space.unsqueeze(0).to(device), TEMP()).reshape([2, 256, 256, 256]).permute([1, 2, 3, 0]).contiguous())).detach().cpu().numpy().reshape([256, 256, 256])

        img_volume_data = itk.GetImageFromArray(volume_data)
        
        series_tag_values = self.metadata_tags
        
        # Tags shared by the series.
        list(
            map(
                lambda tag_value: img_volume_data.SetMetaData(
                    tag_value[0], tag_value[1]
                ),
                series_tag_values,
            )
        )

        # Slice specific tags.
        #   Instance Creation Date
        img_volume_data.SetMetaData("0008|0012", time.strftime("%Y%m%d"))
        #   Instance Creation Time
        img_volume_data.SetMetaData("0008|0013", time.strftime("%H%M%S"))

        # Setting the type to MR so that the slice location is preserved and
        # the thickness is carried over.
        img_volume_data.SetMetaData("0008|0060", "MR")

        # (0020, 0032) image position patient determines the 3D spacing between
        # slices.
        #   Image Position (Patient)
        img_volume_data.SetMetaData(
            "0020|0032",
            "\\".join(map(str, self.dummy_img.TransformIndexToPhysicalPoint((0, 0, 0, t)))),
        )
        #   Instance Number
        img_volume_data.SetMetaData("0020|0013", str(t))

        # Write to the output directory and add the extension dcm, to force
        # writing in DICOM format.
        
        writer = itk.ImageFileWriter()
        # Use the study/series/frame of reference information given in the meta-data
        # dictionary and not the automatically generated information from the file IO
        writer.KeepOriginalImageUIDOn()

        writer.SetFileName(os.path.join(self.output_dir, str(t) + ".dcm"))
        writer.Execute(img_volume_data)     
        
        return 0

import ipyparallel
import ipyparallel.datapub as datapub
import ipyparallel.client.asyncresult as asyncresult
import ipyparallel.util

def write_dicom(t, latent_sub_space: torch.Tensor, params) -> int:
    
    LOAD_GLOBALS(params)
    
    dicom_writer = None
    
    # Create a DICOMWriter object
    dummy_Img = itk.Image([1, 1, 1, params['frame_count']], itk.sitkFloat64)
    dummy_Img.SetSpacing([1.0, 1.0, 1.0, params['scan_time'] / float(params['frame_count'])])
    
    dicom_writer = DICOMWriter(params['frame_count'], params['metadata'], dummy_Img, params['output_folder'])

    dicom_writer.load()
    results =  dicom_writer.write(t, latent_sub_space)
    dicom_writer.unload()
    return results

class write_pool():
    def __init__(self, workers) -> None:
        super().__init__()
        self.workers = workers
        
    def run(self, z: torch.Tensor, params) -> None:

        # Create a processing pool with n workers
        dummy_Img = itk.Image([1, 1, 1, params['frame_count']], itk.sitkFloat64)
        dummy_Img.SetSpacing([1.0, 1.0, 1.0, params['scan_time'] / float(params['frame_count'])])
        pool = self.workers.load_balanced_view()
    
        # Create a list of tasks to be parallelized
        
        writer = itk.ImageFileWriter()
        # Use the study/series/frame of reference information given in the meta-data
        # dictionary and not the automatically generated information from the file IO
        writer.KeepOriginalImageUIDOn()

        modification_time = time.strftime("%H%M%S")
        modification_date = time.strftime("%Y%m%d")

        # Copy some of the tags and add the relevant tags indicating the change.
        # For the series instance UID (0020|000e), each of the components is a number,
        # cannot start with zero, and separated by a '.' We create a unique series ID
        # using the date and time. Tags of interest:
        direction = dummy_Img.GetDirection()
        series_tag_values = [
            ("0008|0031", modification_time),  # Series Time
            ("0008|0021", modification_date),  # Series Date
            ("0008|0008", "DERIVED\\SECONDARY"),  # Image Type
            (
                "0020|000e",
                "1.2.826.0.1.3680043.2.1125."
                + modification_date
                + ".1"
                + modification_time,
            ),  # Series Instance UID
            (
                "0020|0037",
                "\\".join(
                    map(
                        str,
                        (
                            direction[0],
                            direction[4],
                            direction[7],
                            direction[1],
                            direction[5],
                            direction[8],
                        ),
                    )
                ),
            ),  # Image Orientation
            ("0010|0010", "Doe, John"), # (Patient)
            ("0010|0020", "196883"), # a totally "random number"
            ("0008|103e", "4D mri Siemens Data reconstrcution using moco-storm"),  # Series Description
        ]

        # If we want to write floating point values, we need to use the rescale
        # slope, "0028|1053", to select the number of digits we want to keep. We
        # also need to specify additional pixel storage and representation
        # information.
        rescale_slope = 0.001  # keep three digits after the decimal point
        series_tag_values = series_tag_values + [
            ("0028|1053", str(rescale_slope)),  # rescale slope
            ("0028|1052", "0"),  # rescale intercept
            ("0028|0100", "16"),  # bits allocated
            ("0028|0101", "16"),  # bits stored
            ("0028|0102", "15"),  # high bit
            ("0028|0103", "1"),
        ]  # pixel representation

        # Write slices to output directory
        #list(
        #    map(
        #        lambda i: writeSlices(series_tag_values, dummy_Img, sys.argv[1], i),
        #        range(dummy_Img.GetDepth()),
        #    )
        #)
        params['metadata'] = series_tag_values
        tasks = []
        for t in range(z.shape[0]):
            latent_sub_space = z[t]  # generate latent sub space for frame t
            task = pool.apply_async(write_dicom, t, latent_sub_space, params)
            tasks.append(task)
    
        # Wait for all tasks to complete
        results = [task.get() for task in tasks]
        if np.sum(results) != 0:
            raise Exception("Parallel write failed.")

