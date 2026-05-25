#######################
#     Roshidere       #   
#     WEB source      #  
#   by Sergio00166    #
#######################
# Use FGS at 10
from vstools import initialize_clip, finalize_clip
from vsdehalo.mask import dehalo_alpha, fine_dehalo
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

# The source has very strong haloing
deh = fine_dehalo(src, ss=1, planes=0)
filtered = process_lineart(deh, strenght=0.5, sharpness=0.75)
deband = core.placebo.Deband(filtered, planes=1, threshold=1, grain=0)
finalize_clip(deband).set_output()

 