# Setup based on VLA observations from Chia-Lin Ko

import time
Simobserve = True
Clean = True

start = time.time()

if Simobserve:
    print('\033[1m\n[vla_simulation] Observing Stokes I ...\033[0m')
    simobserve(
        project = 'bandKu_I',
        skymodel = 'polaris_I.fits',
        incenter = '18GHz',
        inwidth = '6.144GHz', 
        setpointings = True,
        integration = '2s',
        totaltime = '32.5min',
        indirection = 'J2000 16h32m22.62 -24d28m32.5',
        refdate = '2021/01/02', 
        hourangle = 'transit',
        obsmode = 'int',
        antennalist = 'vla.a.cfg',
        thermalnoise = 'tsys-atm',
        graphics = 'both',
        overwrite = True,
        verbose = True
    )

if Clean:
    print('\033[1m\n[vla_simulation] Cleaning Stokes I ...\033[0m')
    tclean(
        vis = 'bandKu_I/bandKu_I.vla.a.noisy.ms',
        imagename = 'bandKu_I/clean_I',
        imsize = 400,
        cell = '0.01125arcsec',
        reffreq = '18GHz', 
        specmode = 'mfs',
        gridder = 'standard',
        deconvolver = 'multiscale',
        scales = [1, 8, 20], 
        weighting = 'briggs',
        robust = 0.0,
        niter = 10000,
        threshold = '2e-5Jy',
        pbcor = True, 
        mask = 'centerbox[[200pix, 200pix], [50pix, 50pix]]', 
        interactive = False,
        verbose = True
    )
    imregrid('bandKu_I/clean_I.image', template='bandKu_I/bandKu_I.vla.a.skymodel', \
        output='bandKu_I/clean_I.image_modelsize', overwrite=True)
    exportfits('bandKu_I/clean_I.image_modelsize', fitsimage='vla_I.fits', \
        dropstokes=True, overwrite=True)

print('\n[vla_simulaton] Elapsed time: {}'\
		.format(time.strftime("%H:%M:%S", time.gmtime(time.time()-start)))\
	)
