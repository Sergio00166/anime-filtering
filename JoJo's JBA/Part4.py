#############################
#        JoJo JBA           #   
#     Part 4 filtering      #  
#       JPBD source         #  
#     by Sergio00166        #
#############################
# Use FGS at 20
from vodesfunc.rescale_ext.mixed_rescale import MixedRB
from vstools import initialize_clip, finalize_clip
from vskernels import BicubicSharp, Lanczos, Catrom
from vsdehalo.mask import fine_dehalo, dehalo_alpha
from vsdenoise import bm3d, nl_means
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

# For some reason studio sometimes uses lanczos
def build_rescale(src, kernel):
    rb = RescaleBuilder(src)
    rb = rb.descale(kernel, 1280, 720)
    rb = rb.double(ArtCNN.R8F64())
    rb = rb.errormask(mask=0.0275)
    rb = rb.linemask() # Default is good
    # Keep at least similar sharpness
    return rb.downscale(Catrom(linear=True))

builders = [
    build_rescale(src, BicubicSharp),
    build_rescale(src, Lanczos),
]
rescale = MixedRB(*builders).get_upscaled()

deh = fine_dehalo(rescale, ss=1, planes=0)
denoise = bm3d_luma(deh, sigma=4, tr=2, profile=bm3d.Profile.FAST)
denoise = nl_means(denoise, h=0.4, tr=2, planes=[1, 2])
deband = core.placebo.Deband(denoise, threshold=1, planes=1, grain=1)
finalize_clip(deband).set_output()

 