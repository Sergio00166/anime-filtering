#############################
#    Seishun Buta Yarou     #   
#      S02 filtering        #  
#       JPBD source         #  
#     by Sergio00166        #
#############################
# Use FGS at 10
from vstools import initialize_clip, finalize_clip
from vsdenoise import bm3d, nl_means
from vsdehalo.mask import dehalo_alpha
from vsaa.deinterlacers import NNEDI3
from vsmasktools.edge import Kirsch
from vssource import BestSource
import vapoursynth as vs
from os import cpu_count

core = vs.core
core.num_threads = cpu_count() // 2
src = BestSource.source("__SRC__")
src = initialize_clip(src)
nnedi3 = NNEDI3(opencl=True, nns=3)

def bm3d_luma(clip, **kwargs):
    y,u,v = core.std.SplitPlanes(clip)
    y = core.resize.Point(y, format=vs.GRAYS)
    denoise = bm3d(y, **kwargs)
    final = core.resize.Point(denoise, format=clip.format.id)
    yuv = core.std.ShufflePlanes([final, u, v], planes=[0, 0, 0], colorfamily=vs.YUV)
    return yuv

def process_lineart(clip, strenght=0.5, sharpness=1):
    y,u,v = core.std.SplitPlanes(clip)
    mask = Kirsch().edgemask(y)

    # Make it sharp, remove blur
    sharp = core.cas.CAS(y, sharpness=sharpness)
    # Remove aliasing from source
    aa = nnedi3.antialias(sharp)
    # Remove small haloing from CAS
    deh = dehalo_alpha(aa, ss=1)

    # Weighted mix to avoid plastiline look
    mixed = core.std.Merge(y, deh, weight=strenght)
    y = core.std.MaskedMerge(y, mixed, mask=mask, planes=0)

    # Merge the luma with the untouched chroma
    final = core.resize.Point(y, format=clip.format.id)
    yuv = core.std.ShufflePlanes([final, u, v], planes=[0, 0, 0], colorfamily=vs.YUV)
    return yuv

filtered = process_lineart(src, strenght=0.75, sharpness=1)
denoise = bm3d_luma(filtered, sigma=2, tr=2, profile=bm3d.Profile.FAST, refine=0)
denoise = nl_means(denoise, h=0.3, tr=2, planes=[1, 2])
deband = core.placebo.Deband(denoise, planes=1, threshold=1, grain=1)
finalize_clip(deband).set_output()

 
