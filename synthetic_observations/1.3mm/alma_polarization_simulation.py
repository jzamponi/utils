# Setup based on Sadavoy et al. (2018b)

import time
Simobserve = True
Clean = True
polarization = True

start = time.time()

if Simobserve:
    print('\033[1m\n[alma_simulation] Observing Stokes I ...\033[0m')
    simobserve(
        project = 'band6_I',
        skymodel = 'polaris_I.fits',
        incenter = '233GHz',
        inwidth = '7.5GHz', 
        setpointings = True,
        integration = '2s',
        totaltime = '7.2min',
        indirection = 'J2000 16h32m22.63 -24d28m31.8',
        refdate = '2017/05/20', 
        hourangle = 'transit',
        obsmode = 'int',
        antennalist = 'alma.cycle3.5.cfg',
        thermalnoise = 'tsys-atm',
        graphics = 'both',
        overwrite = True,
        verbose = True
    )
    if polarization:
        print('\033[1m\n[alma_simulation] Observing Stokes Q ...\033[0m')
        simobserve(
            project = 'band6_Q',
            skymodel = 'polaris_Q.fits',
            incenter = '233GHz',
            inwidth = '7.5GHz', 
            setpointings = True,
            integration = '2s',
            refdate = '2017/05/20', 
            totaltime = '7.2min',
            indirection = 'J2000 16h32m22.63 -24d28m31.8',
            hourangle = 'transit',
            obsmode = 'int',
            antennalist = 'alma.cycle3.5.cfg',
            thermalnoise = 'tsys-atm',
            graphics = 'file',
            overwrite = True,
            verbose = True
        )
        print('\033[1m\n[alma_simulation]\033[0m Observing Stokes U ...\033[0m')
        simobserve(
            project = 'band6_U',
            skymodel = 'polaris_U.fits',
            incenter = '233GHz',
            inwidth = '7.5GHz', 
            setpointings = True,
            integration = '2s',
            refdate = '2017/05/20', 
            totaltime = '7.2min',
            indirection = 'J2000 16h32m22.63 -24d28m31.8',
            hourangle = 'transit',
            obsmode = 'int',
            antennalist = 'alma.cycle3.5.cfg',
            thermalnoise = 'tsys-atm',
            graphics = 'file',
            overwrite = True,
            verbose = True
        )

if Clean:
    print('\033[1m\n[alma_simulation] Cleaning Stokes I ...\033[0m')
    tclean(
        vis = 'band6_I/band6_I.alma.cycle3.5.noisy.ms',
        imagename = 'band6_I/clean_I',
        imsize = 400,
        cell = '0.029arcsec',
        reffreq = '233GHz', 
        specmode = 'mfs',
        gridder = 'standard',
        deconvolver = 'multiscale',
        scales = [0, 8, 20, 40], 
        weighting = 'briggs',
        robust = 0.5,
        uvtaper = '0.1arcsec',
        mask = 'band6_I/band6_I.alma.cycle3.5.skymodel',
        niter = 10000,
        threshold = '250uJy',
        pbcor = True, 
        interactive = True,
        verbose = True
    )
    if polarization:
        print('\033[1m\n[alma_simulation] Cleaning Stokes Q ...\033[0m')
        tclean(
            vis = 'band6_Q/band6_Q.alma.cycle3.5.noisy.ms',
            imagename = 'band6_Q/clean_Q',
            imsize = 400,
            cell = '0.029arcsec',
            reffreq = '233GHz', 
            specmode = 'mfs',
            gridder = 'standard',
            deconvolver = 'hogbom',
            weighting = 'briggs',
            robust = 0.5,
            uvtaper = '0.1arcsec',
            mask = 'band6_I/band6_I.alma.cycle3.5.skymodel',
            niter = 10000,
            threshold = '100uJy',
            pbcor = True, 
            interactive = False,
            verbose = True
        )
        print('\033[1m\n[alma_simulation] Cleaning Stokes U ...\033[0m')
        tclean(
            vis = 'band6_U/band6_U.alma.cycle3.5.noisy.ms',
            imagename = 'band6_U/clean_U',
            imsize = 400,
            cell = '0.029arcsec',
            reffreq = '233GHz', 
            specmode = 'mfs',
            gridder = 'standard',
            deconvolver = 'hogbom',
            weighting = 'briggs',
            robust = 0.5,
            uvtaper = '0.1arcsec',
            mask = 'band6_I/band6_I.alma.cycle3.5.skymodel',
            niter = 10000,
            threshold = '100uJy',
            pbcor = True, 
            interactive = False,
            verbose = True
        )

    imregrid('band6_I/clean_I.image', template='band6_I/band6_I.alma.cycle3.5.skymodel.flat', \
        output='band6_I/clean_I.image_modelsize', overwrite=True)
    exportfits('band6_I/clean_I.image_modelsize', fitsimage='alma_I.fits', dropstokes=True, overwrite=True)

    if polarization:
        imregrid('band6_Q/clean_Q.image', template='band6_Q/band6_Q.alma.cycle3.5.skymodel.flat', \
            output='band6_Q/clean_Q.image_modelsize', overwrite=True)
        imregrid('band6_U/clean_U.image', template='band6_U/band6_U.alma.cycle3.5.skymodel.flat', \
            output='band6_U/clean_U.image_modelsize', overwrite=True)
        exportfits('band6_Q/clean_Q.image_modelsize', fitsimage='alma_Q.fits', overwrite=True, dropstokes=True)
        exportfits('band6_U/clean_U.image_modelsize', fitsimage='alma_U.fits', overwrite=True, dropstokes=True)


print('\n[alma_simulaton] Elapsed time: {}'\
		.format(time.strftime("%H:%M:%S", time.gmtime(time.time()-start)))\
	)
