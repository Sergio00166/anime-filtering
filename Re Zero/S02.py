#############################
#         Re:Zero           #   
#      S02 filtering        #  
#       JPBD source         #  
#     by Sergio00166        #
#############################
# Use FGS at 15
from vsdenoise import bm3d, nl_means, deblock as dbk
from vstools import initialize_clip, finalize_clip
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

# Default wrapper is an fucking dumbass
def bm3d_luma(clip, **kwargs):
    y,u,v = core.std.SplitPlanes(clip)
    y = core.resize.Point(y, format=vs.GRAYS)
    denoise = bm3d(y, **kwargs)
    final = core.resize.Point(denoise, format=clip.format.id)
    yuv = core.std.ShufflePlanes([final, u, v], planes=[0, 0, 0], colorfamily=vs.YUV)
    return yuv

# Remove the heavy blocking on luma
deblock = dbk.deblock_qed(src, beta=(8,8), planes=0)

_, rescale = (
    RescaleBuilder(deblock)
    .descale(Bilinear(border_handling=1), 1280, 720)
    .double(ArtCNN.R8F64)
    .errormask(0.0275)
    .linemask() # Default is good
    .post_double(lambda clip: core.cas.CAS(clip, sharpness=0.25))
    .downscale(Hermite(linear=True))
    .final()
)
# Minor (fast) cleanup, encoder will do the rest
denoise = bm3d_luma(rescale, sigma=1, tr=2, profile=bm3d.Profile.FAST, refine=0)
denoise = nl_means(denoise, h=0.2, tr=2, planes=[1, 2])
deband = core.placebo.Deband(denoise, threshold=1.5, planes=1, grain=1)
finalize_clip(deband).set_output()

 
