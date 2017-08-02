from astropy.io import fits
import numpy as np
import subprocess as sp
import os

class Observation:
    def __init__(self, root, name, rms):
        self.name = name
        self.root = root
        self.uvf  = fits.open(self.root + self.name + '.uvf')

        self.rms = rms

        self.dec = self.uvf[3].data['DECEPO'][0]
        self.ra = self.uvf[3].data['RAEPO'][0]

    def clean(self, show=True):
        """
        Clean and image (if desired) a observation-specific model.
        Either model image or residuals may be chosen.
        """

        # Set observation-specific clean filepath; clear filepaths
        sp.call('rm -rf {}.{{mp,bm,cl,cm}}'.format(self.root + self.name), shell=True)

        #Dirty clean; save rms for clean cutoff
        sp.call(['invert',
            'vis={}.vis'.format(self.root + self.name),
            'map={}.mp'.format(self.root + self.name),
            'beam={}.bm'.format(self.root + self.name),
            'cell=0.03arcsec', 'imsize=512', 'options=systemp,mfs', 'robust=2'])
        imstat_out=sp.check_output(['imstat',
            'in={}.mp'.format(self.root + self.name),
            "region='boxes(256,0,512,200)'"])
        dirty_rms = float(imstat_out[-38:-29])
        print("Dirty rms is {}".format(dirty_rms))


        # Clean down to half the rms
        sp.call(['clean',
            'map={}.mp'.format(self.root + self.name),
            'beam={}.bm'.format(self.root + self.name),
            'out={}.cl'.format(self.root + self.name),
            'niters=100000', 'cutoff={}'.format(dirty_rms/2)])
        sp.call(['restor',
            'map={}.mp'.format(self.root + self.name),
            'beam={}.bm'.format(self.root + self.name),
            'model={}.cl'.format(self.root + self.name),
            'out={}.cm'.format(self.root + self.name)])

        # Display clean image with 2,4,6 sigma contours, if desired
        if show == True:

            # Display an unimportant imaage to get around the fact that the first
            # image displayed with cgdisp in a session can't be deleted
            sp.call(['cgdisp', 'in=cgdisp_start.im', 'type=p', 'device=/xs'])

            #Get rms for countours
            imstat_out = sp.check_output(['imstat',
                'in={}.cm'.format(self.root + self.name),
                "region='boxes(256,0,512,200)'"])
            clean_rms = float(imstat_out[-38:-29])
            print("Clean rms is {}".format(clean_rms))

            # Display
            sp.call(['cgdisp',
                'in={}.cm,{}.cm'.format(self.root + self.name, self.root + self.name),
                'type=p,c', 'device=/xs',
                'slev=a,{}'.format(clean_rms), 'levs1=-6,-4,-2,2,4,6',
                'region=arcsec,box(-5,-5,5,5)',
                'labtyp=arcsec', 'beamtyp=b,l,3',])

class Model:

    def obs_sample(self, obs):
        """
        Create model fits file with correct header information and sample using
        ALMA observation uv coverage to to create model .vis and .uvf files.
        """

        # define observation-specific model name and delete any preexisting conditions
        path = self.root + self.name + obs.name
        sp.call('rm -rf {}{{.im,.vis,.uvf}}'.format(path), shell=True)

        # Convert model into MIRIAD .im image file
        sp.call(['fits', 'op=xyin',
            'in={}.fits'.format(path),
            'out={}.im'.format(path)], stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))

        # Sample the model image using the observation uv coverage
        sp.call(['uvmodel', 'options=replace',
            'vis={}.vis'.format(obs.root + obs.name),
            'model={}.im'.format(path),
            'out={}.vis'.format(path)], stdout=open(os.devnull, 'wb'))

        #Convert to UVfits
        sp.call(['fits', 'op=uvout',
            'in={}.vis'.format(path),
            'out={}.uvf'.format(path)], stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))

    def get_chi(self, obs):
        """
        Return chi^2 statistics of model.
        """

        datrlimwt = obs.uvf[0].data['data']
        datrl_xx = datrlimwt[:, 0, 0, 0, 0, 0, 0]
        datrl_yy = datrlimwt[:, 0, 0, 0, 0, 1, 0]
        datim_xx = datrlimwt[:, 0, 0, 0, 0, 0, 1]
        datim_yy = datrlimwt[:, 0, 0, 0, 0, 1, 1]
        weights =  datrlimwt[:, 0, 0, 0, 0, 0, 2]
        datrl_stokes = np.array((datrl_xx + datrl_yy) / 2.)
        datim_stokes = np.array((datim_xx + datim_yy) / 2.)

        uvf  = fits.open(self.root + self.name + obs.name +'.uvf')
        modrlimwt = uvf[0].data['data']
        modrl_stokes = modrlimwt[::2, 0, 0, 0, 0, 0]
        modim_stokes = modrlimwt[::2, 0, 0, 0, 0, 1]

        # Calculate chi^2
        chi = np.sum((datrl_stokes - modrl_stokes)**2 * weights +
                     (datim_stokes - modim_stokes)**2 * weights)

        self.chis.append(chi)

    def clean(self, obs, residual=False, show=True):
        """
        Clean and image (if desired) a observation-specific model.
        Either model image or residuals may be chosen.
        """

        # Set observation-specific clean filepath; clear filepaths
        filepath = self.root + self.name + '_' + obs.name
        if residual == True:
            filepath += '.residual'
        sp.call('rm -rf {}.{{mp,bm,cl,cm}}'.format(filepath), shell=True)

        # Clean down to half the observation rms
        sp.call(['invert',
            'vis={}.vis'.format(filepath),
            'map={}.mp'.format(filepath),
            'beam={}.bm'.format(filepath),
            'cell=0.03arcsec', 'imsize=512', 'options=systemp,mfs', 'robust=2'])
        sp.call(['clean',
            'map={}.mp'.format(filepath),
            'beam={}.bm'.format(filepath),
            'out={}.cl'.format(filepath),
            'niters=100000', 'cutoff={}'.format(obs.rms/2)])
        sp.call(['restor',
            'map={}.mp'.format(filepath),
            'beam={}.bm'.format(filepath),
            'model={}.cl'.format(filepath),
            'out={}.cm'.format(filepath)])

        # Display clean image with 2,4,6 sigma contours, if desired
        if show == True:

            # Display an unimportant image to get around the fact that the first
            # image displayed with cgdisp in a session can't be deleted
            sp.call(['cgdisp', 'in=cgdisp_start.im', 'type=p', 'device=/xs'])

            #Get rms for countours
            imstat_out = sp.check_output(['imstat',
                'in={}.cm'.format(filepath),
                "region='boxes(256,0,512,200)'"])
            clean_rms = float(imstat_out[-38:-29])
            print("Clean rms is {}".format(clean_rms))

            # Display
            sp.call(['cgdisp',
                'in={}.cm,{}.cm'.format(filepath, filepath),
                'type=p,c', 'device=/xs',
                'slev=a,{}'.format(clean_rms), 'levs1=-6,-4,-2,2,4,6',
                'region=arcsec,box(-5,-5,5,5)',
                'labtyp=arcsec', 'beamtyp=b,l,3',])

    def residuals(self, obs, show=True):

        """
        Create model residuals (data - model), and clean//display if desired
        """

        #Set observation-specific filepath
        filepath = self.root + self.name + '_' + obs.name

        # Subtract model visibilities from data; outfile is residual visibilities
        sp.call(['uvmodel', 'options=subtract',
            'model={}.im'.format(filepath),
            'vis={}.vis'.format(obs.root + obs.name),
            'out={}.residual.vis'.format(filepath)])

        if show == True:
            self.clean(obs, residual=True)

    def __init__(self, observations, root, name=''):

        # assign name and set of observations
        self.name = 'model' + name
        self.root = root
        self.observations = observations
        self.chis = []
