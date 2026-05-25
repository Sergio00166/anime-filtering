#############################
#   Cyberpunk Edgerunners   #   
#    Season 2 filtering     #  
#   Netflix 4K SDR source   #  
#     by Sergio00166        #
#############################
# Use FGS at 20
from vstools import initialize_clip, finalize_clip
from vsdehalo.mask import fine_dehalo
from vsdenoise import deblock as dbk
from vssource import BestSource
from vskernels import Hermite
import vapoursynth as vs
from os import cpu_count

core = vs.core
core.num_threads = cpu_count() // 2
src = BestSource.source("__SRC__")
src = initialize_clip(src)

# Remove the source blocking at 4k resolution
deblock = dbk.deblock_qed(src, beta=(8,8), planes=0)

# Downscale to native resolution (1080p)
sharp = core.cas.CAS(deblock, sharpness=0.25, planes=0)
downscale = Hermite(linear=True).scale(sharp, width=1920, height=1080)

# Finalize cleaning up the lineart
deh = fine_dehalo(downscale, ss=1, planes=0)
finalize_clip(deh).set_output()

 