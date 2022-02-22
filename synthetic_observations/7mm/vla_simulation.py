# Setup based on Liu et al. (2018)

import time
Simobserve = True
Clean = True
polarization = False

start = time.time()

if Simobserve:
    print('\033[1m\n[vla_simulation] Observing Stokes I ...\033[0m')
    simobserve(
        project = 'bandQ_I',
        skymodel = 'polaris_I.fits',
        incenter = '44GHz',
        inwidth = '8GHz', 
        setpointings = True,
        integration = '2s',
        totaltime = '46min',
        indirection = 'J2000 16h32m22.62 -24d28m32.5',
        refdate = '2015/01/18', 
        hourangle = 'transit',
        obsmode = 'int',
        antennalist = 'vla.cnb.cfg',
        thermalnoise = 'tsys-atm',
        graphics = 'both',
        overwrite = True,
        verbose = True
    )
    if polarization:
        print('\033[1m\n[vla_simulation] Observing Stokes Q ...\033[0m')
        simobserve(
            project = 'bandQ_Q',
            skymodel = 'polaris_Q.fits',
            incenter = '44GHz',
            inwidth = '8GHz', 
            setpointings = True,
            integration = '2s',
            totaltime = '46min',
            indirection = 'J2000 16h32m22.62 -24d28m32.5',
            refdate = '2015/01/18', 
            hourangle = 'transit',
            obsmode = 'int',
            antennalist = 'vla.cnb.cfg',
            thermalnoise = 'tsys-atm',
            graphics = 'both',
            overwrite = True,
            verbose = True
        )
        print('\033[1m\n[vla_simulation]\033[0m Observing Stokes U ...\033[0m')
        simobserve(
            project = 'bandQ_U',
            skymodel = 'polaris_U.fits',
            incenter = '44GHz',
            inwidth = '8GHz', 
            setpointings = True,
            integration = '2s',
            totaltime = '46min',
            indirection = 'J2000 16h32m22.62 -24d28m32.5',
            refdate = '2015/01/18', 
            hourangle = 'transit',
            obsmode = 'int',
            antennalist = 'vla.cnb.cfg',
            thermalnoise = 'tsys-atm',
            graphics = 'both',
            overwrite = True,
            verbose = True
        )

if Clean:
    print('\033[1m\n[vla_simulation] Cleaning Stokes I ...\033[0m')
    tclean(
        vis = 'bandQ_I/bandQ_I.vla.cnb.noisy.ms',
        imagename = 'bandQ_I/clean_I',
        imsize = 400,
        cell = '0.03arcsec',
        reffreq = '44GHz', 
        specmode = 'mfs',
        gridder = 'standard',
        deconvolver = 'multiscale',
        scales = [1, 8, 20], 
        weighting = 'briggs',
        robust = 0.0,
        uvtaper = '0.1arcsec',
        niter = 10000,
        threshold = '4e-5Jy',
        #mask = 'bandQ_I/bandQ_I.vla.cnb.skymodel', 
        mask = 'centerbox[[200pix, 200pix], [50pix, 50pix]]', 
        pbcor = True, 
        interactive = True,
        verbose = True
    )
    if polarization:
        print('\033[1m\n[vla_simulation] Cleaning Stokes Q ...\033[0m')
        tclean(
            vis = 'bandQ_Q/bandQ_Q.vla.cnb.noisy.ms',
            imagename = 'bandQ_Q/clean_Q',
            imsize = 400,
            cell = '0.03arcsec',
            reffreq = '44GHz', 
            specmode = 'mfs',
            gridder = 'standard',
            deconvolver = 'hogbom',
            weighting = 'briggs',
            robust = 0.0,
            uvtaper = '0.1arcsec',
            niter = 10000,
            threshold = '15uJy',
            pbcor = True, 
            interactive = True,
            verbose = True
        )
        print('\033[1m\n[vla_simulation] Cleaning Stokes U ...\033[0m')
        tclean(
            vis = 'bandQ_U/bandQ_U.vla.cnb.noisy.ms',
            imagename = 'bandQ_U/clean_U',
            imsize = 400,
            cell = '0.03arcsec',
            reffreq = '44GHz', 
            specmode = 'mfs',
            gridder = 'standard',
            deconvolver = 'hogbom',
            weighting = 'briggs',
            robust = 0.0,
            uvtaper = '0.1arcsec',
            niter = 10000,
            threshold = '15uJy',
            pbcor = True, 
            interactive = True,
            verbose = True
        )

    imregrid('bandQ_I/clean_I.image', template='bandQ_I/bandQ_I.vla.cnb.skymodel.flat', \
        output='bandQ_I/clean_I.image_modelsize', overwrite=True)
    exportfits('bandQ_I/clean_I.image_modelsize', fitsimage='vla_I.fits', dropstokes=True, overwrite=True)

    if polarization:
        imregrid('bandQ_Q/clean_Q.image', template='bandQ_Q/bandQ_Q.vla.a.skymodel.flat', \
            output='bandQ_Q/clean_Q.image_modelsize', overwrite=True)
        imregrid('bandQ_U/clean_U.image', template='bandQ_U/bandQ_U.vla.a.skymodel.flat', \
            output='bandQ_U/clean_U.image_modelsize', overwrite=True)
        exportfits('bandQ_Q/clean_Q.image_modelsize', fitsimage='vla_Q.fits', overwrite=True, dropstokes=True)
        exportfits('bandQ_U/clean_U.image_modelsize', fitsimage='vla_U.fits', overwrite=True, dropstokes=True)


print('\n[vla_simulaton] Elapsed time: {}'\
		.format(time.strftime("%H:%M:%S", time.gmtime(time.time()-start)))\
	)
