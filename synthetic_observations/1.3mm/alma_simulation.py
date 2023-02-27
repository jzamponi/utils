import time
# IRAS16293 B
# Setup based on Maureira et al. (2020) 
Simobserve = True
Clean = True

start = time.time()

if Simobserve:
	simobserve(
		project = 'band6',
		skymodel = 'polaris_I.fits',
		inbright = '',
		incell = '',
		mapsize = '',
		incenter = '223GHz',
		inwidth = '0.12GHz',
		setpointings = True,
		integration = '2s',
		totaltime = '1.25h',
		indirection = 'J2000 16h32m22.63 -24d28m31.8',
		hourangle = 'transit',
		obsmode = 'int',
		antennalist = 'alma.cycle4.7.cfg',
		thermalnoise = 'tsys-manual',
		graphics = 'both',
		overwrite = True,
		verbose = True
	)

if Clean:
	tclean(
		vis = 'band6/band6.alma.cycle4.7.noisy.ms',
		imagename = 'band6/clean',
		imsize = 400,
		cell = '0.008arcsec',
        reffreq = '223GHz', 
		specmode = 'mfs',
		gridder = 'standard',
		deconvolver = 'multiscale',
        scales = [1, 8, 24], 
		weighting = 'briggs',
		uvrange = '120~2670klambda',
		robust = 0.0,
		niter = 10000,
		threshold = '2.5e-4Jy',
        mask = 'band6/band6.alma.cycle4.7.skymodel', 
		interactive = False,
		verbose = True
	)

imregrid(
    'band6/clean.image', 
    template = 'band6/band6.alma.cycle4.7.skymodel.flat',
    output = 'band6/clean.image.modelsize', 
    overwrite = True
)

exportfits(
    'band6/clean.image.modelsize', 
    fitsimage = 'alma_I.fits', 
    dropstokes = True, 
    dropstokes = True, 
    overwrite=True
)

print('[alma_simulaton] Elapsed time: {}'\
		.format(time.strftime("%H:%M:%S", time.gmtime(time.time()-start)))\
	)
