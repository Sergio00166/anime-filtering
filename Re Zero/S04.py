#############################
#         Re:Zero           #   
#      S04 filtering        #  
#       WEB source          #  
#     by Sergio00166        #
#############################
# Use FGS at 10
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

_, rescale = (
    RescaleBuilder(src)
    .descale(Bilinear(border_handling=1), 1500, 844)
    .double(ArtCNN.R8F64)
    .errormask(0.0275)
    .linemask() # Default is good
    .downscale(Hermite(linear=True))
    .final()
)
deband = core.placebo.Deband(rescale, threshold=1, planes=1, grain=0)
finalize_clip(deband).set_output()

 
