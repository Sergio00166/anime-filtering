#############################
#        KonoSuba           #   
#     S03 filtering         #  
#       JPBD source         #  
#     by Sergio00166        #
#############################
# Use FGS at 10
from vstools import initialize_clip, finalize_clip
from vsdenoise import bm3d, nl_means
from vskernels import Bilinear, Hermite
from vodesfunc import RescaleBuilder
from vssource import BestSource
from vsscale import ArtCNN
import vapoursynth as vs
from os import cpu_count

core = vs.core
core.num_threads = cpu_count() // 2
src = BestSource.source("__SRC__")
src = initialize_clip(src)

def bm3d_luma(clip, **kwargs):
    y,u,v = core.std.SplitPlanes(clip)
    y = core.resize.Point(y, format=vs.GRAYS)
    denoise = bm3d(y, **kwargs)
    final = core.resize.Point(denoise, format=clip.format.id)
    yuv = core.std.ShufflePlanes([final, u, v], planes=[0, 0, 0], colorfamily=vs.YUV)
    return yuv

_, rescale = (
    RescaleBuilder(src)
    # I know it was upscaled with Catrom,
    # but Bilinear looks better for me
    .descale(Bilinear(border_handling=2), 1488, 837)
    .double(ArtCNN.R8F64)
    .errormask(0.0275)
    .linemask() # Default is good
    .downscale(Hermite(linear=True))
    .final()
)
denoise = bm3d_luma(rescale, sigma=2, tr=2, profile=bm3d.Profile.FAST, refine=0)
denoise = nl_means(denoise, h=0.3, tr=2, planes=[1, 2])
deband = core.placebo.Deband(denoise, threshold=1, planes=1, grain=1)
finalize_clip(deband).set_output()

 