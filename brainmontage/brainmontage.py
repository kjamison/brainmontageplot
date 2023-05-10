from nilearn import plotting
from nilearn import datasets
import numpy as np
import nibabel as nib
import nibabel.processing as nibproc
import pandas as pd

import json
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import argparse
import sys
try:
    #need this for installed version
    from brainmontage.utils import *
    from brainmontage._version import __version__
except:
    #need this for source version
    from utils import *
    from _version import __version__

def parse_argument_montageplot(argv):
    parser=argparse.ArgumentParser(description='Save surface ROI and/or volume slice montage',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--outputimage',action='store',dest='outputimage',help='Filename for output image')
    parser.add_argument('--views',action='append',dest='viewnames',nargs='*',help='list of: dorsal, ventral, medial, lateral, or none')
    parser.add_argument('--surftype',action='store',dest='surftype',default='infl',help='inflated, white, pial')
    parser.add_argument('--cmap','--colormap',action='store',dest='cmapname',default='magma')
    parser.add_argument('--cmapfile','--colormapfile',action='store',dest='cmapfile')
    parser.add_argument('--clim', action='append',dest='clim',nargs=2)
    parser.add_argument('--noshading',action='store_true',dest='noshading')
    parser.add_argument('--upscale',action='store',dest='upscale',type=float,default=1,help='Image upscaling factor')
    parser.add_argument('--backgroundcolor','--bgcolor',action='store',dest='bgcolorname',default='white',help='Background color name')
    parser.add_argument('--backgroundrgb','--bgrgb',action='store',dest='bgrgb',type=float,nargs=3,help='Background color R G B (0-1.0)')

    input_arg_group=parser.add_argument_group('Input value options')
    input_arg_group.add_argument('--input',action='store',dest='inputfile',help='.mat or .txt file with ROI input values')
    input_arg_group.add_argument('--inputfield',action='store',dest='inputfieldname',help='field name in .mat INPUTFILE')
    input_arg_group.add_argument('--inputvals','--inputvalues',action='append',dest='inputvals',nargs='*',help='Input list of ROI values directly')

    slice_arg_group=parser.add_argument_group('Slice view options')
    slice_arg_group.add_argument('--slices',action='append',dest='slices',nargs='*',help='Ex: axial 5 10 15 20 sagittal 10 15 20')
    slice_arg_group.add_argument('--axmosaic',action='store',dest='axmosaic',type=int,nargs=2,help="NUMROW NUMCOL for AXIAL slices")
    slice_arg_group.add_argument('--cormosaic',action='store',dest='cormosaic',type=int,nargs=2,help="NUMROW NUMCOL for CORONAL slices")
    slice_arg_group.add_argument('--sagmosaic',action='store',dest='sagmosaic',type=int,nargs=2,help="NUMROW NUMCOL for SAGITTAL slices")
    slice_arg_group.add_argument('--stackdir',action='store',dest='stackdirection',default='horizontal',help='Stack surf+slices horizontal or vertical')

    atlasname_arg_group=parser.add_argument_group('Atlas option 1: atlas name')
    atlasname_arg_group.add_argument('--atlasname',action='store',dest='atlasname')

    atlasopt_arg_group=parser.add_argument_group('Atlas option 2: atlas details')
    atlasopt_arg_group.add_argument('--roilut',action='store',dest='roilutfile')
    atlasopt_arg_group.add_argument('--lhannot',action='store',dest='lhannotfile')
    atlasopt_arg_group.add_argument('--rhannot',action='store',dest='rhannotfile')
    atlasopt_arg_group.add_argument('--lhannotprefix',action='store',dest='lhannotprefix')
    atlasopt_arg_group.add_argument('--rhannotprefix',action='store',dest='rhannotprefix')
    atlasopt_arg_group.add_argument('--annotsurfacename',action='store',dest='annotsurface',default='fsaverage5')
    atlasopt_arg_group.add_argument('--subcortvolume',action='store',dest='subcortvol')
    
    misc_arg_group=parser.add_argument_group('Other options')
    misc_arg_group.add_argument('--version',action='store_true',dest='version')

    args=parser.parse_args(argv)

    if args.version:
        print(__version__)
        exit(0)
    
    return args

def fill_surface_rois(roivals,atlasinfo):
    lhannotprefix=None
    rhannotprefix=None
    roilutfile=atlasinfo['roilutfile']
    lhannotfile=atlasinfo['lhannotfile']
    rhannotfile=atlasinfo['rhannotfile']
    if 'lhannotprefix' in atlasinfo:
        lhannotprefix=atlasinfo['lhannotprefix']
    if 'rhannotprefix' in atlasinfo:
        rhannotprefix=atlasinfo['rhannotprefix']
    
    if lhannotfile.endswith("annot"):
        #for .annot files, we neet the LUT that says which names to include and what order they go in
        #read in the .annot data
        Troi=pd.read_table(roilutfile,delimiter='\s+',header=None,names=['label','name','R','G','B'])
        Troi=Troi[Troi['name']!='Unknown']
    
        lhlabels,ctab,lhnames=nib.freesurfer.io.read_annot(lhannotfile)
        lhnames=[(x.decode('UTF-8')) for x in lhnames]
        if lhannotprefix is not None:
            lhnames=['%s%s' % (lhannotprefix,x) for x in lhnames]

        rhlabels,ctab,rhnames=nib.freesurfer.io.read_annot(rhannotfile)
        rhnames=[(x.decode('UTF-8')) for x in rhnames]
        if rhannotprefix is not None:
            rhnames=['%s%s' % (rhannotprefix,x) for x in rhnames]

        

        lhannotval=np.zeros(Troi.shape[0])
        rhannotval=np.zeros(Troi.shape[0])
        for i,n86 in enumerate(Troi['name']):
            lhidx=[j for j,nannot in enumerate(lhnames) if nannot==n86]
            rhidx=[j for j,nannot in enumerate(rhnames) if nannot==n86]
            if len(lhidx)==1:
                lhannotval[i]=lhidx[0]
            elif len(rhidx)==1:
                rhannotval[i]=rhidx[0]
        
    elif lhannotfile.endswith(".shape.gii") or lhannotfile.endswith(".label.gii"):
        lhgii=nib.load(lhannotfile)
        lhlabels=lhgii.agg_data()
        
        rhgii=nib.load(rhannotfile)
        rhlabels=rhgii.agg_data()
        
        maxval=np.max([lhlabels]+[rhlabels])
        lhannotval=np.arange(1,maxval+1)
        rhannotval=np.arange(1,maxval+1)
        
        Troi=pd.DataFrame()
    
    Troi['lhannot']=lhannotval
    Troi['rhannot']=rhannotval
    
    surfvalsLR={}
    surfvalsLR['left']=np.zeros(lhlabels.shape)
    surfvalsLR['right']=np.zeros(rhlabels.shape)
    for i in range(Troi.shape[0]):
        v=roivals[i]
        if Troi['lhannot'].iloc[i]>0:
            surfvalsLR['left'][lhlabels==Troi['lhannot'].iloc[i]]=v
        if Troi['rhannot'].iloc[i]>0:
            surfvalsLR['right'][rhlabels==Troi['rhannot'].iloc[i]]=v
            
    return surfvalsLR

def fill_volume_rois(roivals, atlasinfo, backgroundval=0, referencevolume=None):
    if not 'subcorticalvolume' in atlasinfo or atlasinfo['subcorticalvolume'] is None:
        raise Exception("Subcortical volume not found for atlas '%s'" % (atlasinfo["name"]))
    
    roilutfile=atlasinfo['roilutfile']
    Troi=pd.read_table(roilutfile,delimiter='\s+',header=None,names=['label','name','R','G','B'])
    Troi=Troi[Troi['name']!='Unknown']
    
    roinib=nib.load(atlasinfo["subcorticalvolume"])
    
    if referencevolume is None:
        refnib=None
    if isinstance(referencevolume,nib.Nifti1Image) or isinstance(referencevolume,nib.Nifti2Image):
        refnib=referencevolume
    elif isinstance(referencevolume,str):
        refnib=nib.load(referencevolume)
    
    if refnib is not None:
        #resample to reference volume space
        roinib=nibproc.resample_from_to(roinib,refnib,order=0)

    Vroi=roinib.get_fdata()
    Vnew=backgroundval*np.ones(Vroi.shape)
    for i in range(Troi.shape[0]):
        if(np.any(Vroi==Troi['label'].iloc[i])):
            Vnew[Vroi==Troi['label'].iloc[i]]=roivals[i]
    
    return Vnew

def retrieve_atlas_info(atlasname, atlasinfo_jsonfile=None, scriptdir=None):
    if scriptdir is None:
        scriptdir=getscriptdir()
    
    if atlasinfo_jsonfile is None:
        atlasinfo_jsonfile="%s/atlases/atlas_info.json" % (scriptdir)
    
    lhannotprefix=None
    rhannotprefix=None
    subcortfile=None
    with open(atlasinfo_jsonfile,'r') as f:
        atlas_info_list=json.load(f)
    
    if atlasname in atlas_info_list:
        roilutfile=atlas_info_list[atlasname]['roilut'].replace('%SCRIPTDIR%',scriptdir)
        lhannotfile=atlas_info_list[atlasname]['lhannot'].replace('%SCRIPTDIR%',scriptdir)
        rhannotfile=atlas_info_list[atlasname]['rhannot'].replace('%SCRIPTDIR%',scriptdir)
        
        if 'lhannotprefix' in atlas_info_list[atlasname]:
            lhannotprefix=atlas_info_list[atlasname]['lhannotprefix']
        if 'rhannotprefix' in atlas_info_list[atlasname]:
            rhannotprefix=atlas_info_list[atlasname]['rhannotprefix']

        if 'subcorticalvolume' in atlas_info_list[atlasname]:
            subcortfile=atlas_info_list[atlasname]['subcorticalvolume'].replace('%SCRIPTDIR%',scriptdir)
        
        annotsurfacename=atlas_info_list[atlasname]['annotsurface']
    else:
        raise Exception("atlas name '%s' not found. Choose from %s" % (atlasname, ",".join(atlas_info_list.keys())))
    
    atlasinfo={'atlasname':atlasname,'roilutfile':roilutfile,'lhannotfile':lhannotfile,'rhannotfile':rhannotfile,
        'annotsurfacename':annotsurfacename,'lhannotprefix':lhannotprefix,'rhannotprefix':rhannotprefix,
        'subcorticalvolume':subcortfile}
    
    return atlasinfo

def render_surface_view(surfvals,surf,azel=None,surfbgvals=None,shading=True,lightdir=None,val_smooth_iters=0,shading_smooth_iters=1,
                        colormap=None, clim=None,
                        backgroundcolor=None,figsize=None,figure=None):

    if figsize is None:
        figsize=figure.get_size_inches()
    
    if backgroundcolor is None:
        backgroundcolor=figure.get_facecolor()

    figure.clear()
    
    if shading and lightdir is not None:
        shadingvals=mesh_shading(surf[0], surf[1], lightdir)
        shadingvals,_=mesh_diffuse(vertvals=shadingvals,verts=surf[0],faces=surf[1],iters=shading_smooth_iters)
        
        #adjust computed shading values to look brighter
        shadingvals=np.minimum(shadingvals**.5,.8)/.8
        
        shadingvals=shadingvals/shadingvals.max()
        
        #need a colormap where black->white starts in the middle
        #since plot_stat_map requires symmetric range
        symmgraycolors = plt.get_cmap("gray",256)(abs(np.linspace(-1, 1, 256)))
        cmap_symmgray=ListedColormap(symmgraycolors)

        v=plotting.plot_surf_stat_map(surf, stat_map=shadingvals,
                            bg_map=surfbgvals, bg_on_data=False,
                            darkness=.5,cmap=cmap_symmgray, colorbar=False, vmax=1,
                            figure=figure)
        v.set_size_inches(figsize)
        
        if azel is not None:
            v.get_axes()[-1].view_init(azim=azel[0],elev=azel[1])
        
        pixshading=fig2pixels(v)
        figure.clear()

        pixshading=pixshading[:,:,:3].astype(np.float32)/255.0
        pixshading=np.mean(pixshading[:,:,:3],axis=2,keepdims=True)
        #pixshading=pixshading/pixshading.max()
        pixshading=np.clip(pixshading/pixshading[pixshading<1.0].max(),0,1)
    
    ##############################################
    #generate main ROI surface image (unshaded)
    
    #plot_surf_stat_map(stat_map=, vmax=)
    #plot_surf_roi(roi_map=, vmin=, vmax=)
    
    surfvals,_=mesh_diffuse(vertvals=surfvals,verts=surf[0],faces=surf[1],iters=val_smooth_iters)

    if clim is None:
        clim=[surfvals.min(), surfvals.max()]
    v=plotting.plot_surf_roi(surf, roi_map=surfvals,
                        bg_map=surfbgvals, bg_on_data=False,
                        darkness=.5,cmap=colormap, colorbar=False, vmin=clim[0], vmax=clim[1],
                        figure=figure)
    v.set_size_inches(figsize)
    
    if azel is not None:
        v.get_axes()[-1].view_init(azim=azel[0],elev=azel[1])

    if backgroundcolor is not None:
        v.get_axes()[-1].set_facecolor(backgroundcolor)

    pix=fig2pixels(v)

    ##############################################
    #compute background mask for cropping purposes
    
    #find an rgb combination that doesnt exist anywhere in the image
    rgblist=pix[:,:,:3].astype(np.float32)
    rgblist=rgblist[:,:,0]+rgblist[:,:,1]*256+rgblist[:,:,2]*256*256
    rgblist_unique=np.unique(rgblist)
    rgbnew=0
    for irgb in range(256**3):
        if not irgb in rgblist_unique:
            rgbnew=irgb
            break
    rgbnew=[rgbnew % 256, rgbnew // 256 % 256, rgbnew // (256*256)]
    
    #use [0-1] based rgb vals for background
    rgbnew=[x/255.0 for x in rgbnew]
    v.get_axes()[-1].set_facecolor(rgbnew)
    
    pixbg=fig2pixels(v)
    figure.clear()

    _,cropcoord=cropbg(pixbg)
    ##############################################
    
    #pix,cropcoord=cropbg(pix)
    pix=pix[cropcoord[0]:cropcoord[1],cropcoord[2]:cropcoord[3],:]
    
    if shading:
        pixshading=pixshading[cropcoord[0]:cropcoord[1],cropcoord[2]:cropcoord[3],:]
        if pix.shape[2]==4:
            #dont apply shading to alpha channel
            pixalpha=pix[:,:,3]
        pix=np.uint8(np.clip(np.round(pix.astype(np.float32)*pixshading),0,255))
        if pix.shape[2]==4:
            pix[:,:,3]=pixalpha
    return pix

def slice_volume_to_rgb(volvals,bgvolvals,bgmaskvals,sliceaxis,slice_indices,mosaic,cmap,clim,bg_cmap,blank_cmap):
    imgslice,mosiacinfo=vol2mosaic(volvals, sliceaxis=sliceaxis, slice_indices=slice_indices, mosaic=mosaic)
    imgslice_brainbg,_=vol2mosaic(bgvolvals,sliceaxis=sliceaxis,slice_indices=mosiacinfo['slice_indices'],mosaic=mosiacinfo['mosaic'])
    imgslice_brainmask,_=vol2mosaic(bgmaskvals,sliceaxis=sliceaxis,slice_indices=mosiacinfo['slice_indices'],mosaic=mosiacinfo['mosaic'])
    rgbslice=val2rgb(imgslice,cmap,clim)
    rgbslice_brainbg=val2rgb(imgslice_brainbg,bg_cmap)
    rgbblank=val2rgb(np.zeros(imgslice.shape),blank_cmap,[0,1])
    imgslice_alpha=np.atleast_3d(np.logical_not(np.isnan(imgslice)))
    imgslice_brainmask=np.atleast_3d(imgslice_brainmask)
    rgbslice=rgbslice_brainbg*(1-imgslice_alpha)+rgbslice*(imgslice_alpha)
    rgbslice=rgbblank*(1-imgslice_brainmask)+rgbslice*(imgslice_brainmask)

    return rgbslice

def create_montage_figure(roivals,atlasinfo=None, atlasname=None,
    roilutfile=None,lhannotfile=None,rhannotfile=None,annotsurfacename='fsaverage5',lhannotprefix=None, rhannotprefix=None,
    viewnames=None,surftype='infl',clim=None,colormap=None, noshading=False, upscale_factor=1, backgroundcolor="white",
    slice_dict={}, mosaic_dict={},slicestack_order=['axial','coronal','sagittal'],slicestack_direction='horizontal',
    outputimagefile=None):
    
    try:
        colormap=stringfromlist(colormap,list(plt.colormaps.keys()),allow_startswith=False)
    except:
        pass

    if clim is None or len(clim)==0:
        clim=[np.nanmin(roivals), np.nanmax(roivals)]
    
    if viewnames is not None and isinstance(viewnames,str):
        #make sure viewnames is iterable
        viewnames=[viewnames]
            
    if viewnames is None or len(viewnames)==0:
        viewnames=['dorsal','lateral','medial','ventral']
    elif "all" in [v.lower() for v in viewnames]:
        viewnames=['dorsal','lateral','medial','ventral']
    
    try:
        viewnames=stringfromlist(viewnames,['none','dorsal','lateral','medial','ventral'])
    except:
        raise Exception("Viewname must be one of: none, dorsal, lateral, medial, ventral")

    if atlasname is not None and atlasinfo is None:
        atlasinfo=retrieve_atlas_info(atlasname)
    
    if atlasinfo is None:
        atlasinfo['roilutfile']=roilutfile
        atlasinfo['lhannotfile']=lhannotfile
        atlasinfo['rhannotfile']=rhannotfile
        atlasinfo['annotsurfacename']=annotsurfacename
        atlasinfo['lhannotprefix']=lhannotprefix
        atlasinfo['rhannotprefix']=rhannotprefix
    
    #just to make things easier now that we are inside the function
    shading=not noshading
    
    #slicestack_direction
    try:
        slicestack_direction=stringfromlist(slicestack_direction,['horizontal','vertical'])
    except:
        raise Exception("Slicestack direction must be one of: horizontal, vertical")
    
    try:
        slicestack_order=stringfromlist(slicestack_order,['axial','coronal','sagittal'])
    except:
        raise Exception("Slicestack order options must be on of: axial, coronal, sagittal")
    
    fsaverage = datasets.fetch_surf_fsaverage(mesh=atlasinfo['annotsurfacename'])
    
    view_azel_lh={'dorsal':[180,90],'lateral':[180,0], 'medial':[0,0], 'ventral':[0,-90]}
    view_azel_rh={'dorsal':[0,90],'lateral':[0,0], 'medial':[180,0], 'ventral':[180,-90]}
    view_lightdir_lh={'dorsal':[0,0,1],'lateral':[-1,0,0],'medial':[1,0,0],'ventral':[0,0,-1]}
    view_lightdir_rh={'dorsal':[0,0,1],'lateral':[1,0,0],'medial':[-1,0,0],'ventral':[0,0,-1]}
    
    pixlist=[]
    pixlist_hemi=[]

    surfLR={}

    shading_smooth_iters=1
    if surftype == 'infl':
        shading_smooth_iters=10
    
    surfLR['left']=nib.load(fsaverage[surftype+'_'+'left']).agg_data()
    surfLR['right']=nib.load(fsaverage[surftype+'_'+'right']).agg_data()

    surfbgvalsLR={'left':fsaverage['sulc_left'],'right':fsaverage['sulc_right']}

    #rescale values (and clim) to [0,1000] so that plot_roi doesn't have issues with near-zero values
    zeroval=1e-8
    roivals=roivals.flatten()
    roivals[roivals==0]=zeroval
    #roivals_rescaled=roivals
    #clim_rescaled=clim
    
    clim_denom=clim[1]-clim[0]
    if clim_denom==0:
        clim_denom=1
        
    roivals_rescaled=np.clip(1000*((roivals-clim[0])/clim_denom),zeroval,1000)
    clim_rescaled=[0,1000]
    
    surfvalsLR=fill_surface_rois(roivals_rescaled,atlasinfo)

    fig=plt.figure(figsize=(6.4,6.4),facecolor=backgroundcolor)
    figsize=fig.get_size_inches()
    figsize=[x*upscale_factor for x in figsize]
    fig.set_size_inches(figsize)

    for ih,h in enumerate(['left','right']):
        for iv, viewname in enumerate(viewnames):
            if viewname == 'none':
                continue
            azel=None
            if h=='left':
                azel=view_azel_lh[viewname]
                lightdir=view_lightdir_lh[viewname]
            else:
                azel=view_azel_rh[viewname]
                lightdir=view_lightdir_rh[viewname]
            
            pix=render_surface_view(surfvals=surfvalsLR[h],surf=surfLR[h],surfbgvals=surfbgvalsLR[h],
                                    azel=azel,lightdir=lightdir,shading=shading,shading_smooth_iters=shading_smooth_iters,
                                    colormap=colormap, clim=clim_rescaled,
                                    figure=fig)

            pix=padimage(pix,bgcolor=None,padamount=1)

            pixlist+=[pix]
            pixlist_hemi+=[h]
    
    #pad all surface views to the same width
    if len(pixlist)>0:
        wmax=max([x.shape[1] for x in pixlist])
        pixlist=[padimage(x,bgcolor=None,padfinalsize=[-1,wmax]) for x in pixlist]
        
        pixlist_left=[x for i,x in enumerate(pixlist) if pixlist_hemi[i]=='left']
        pixlist_right=[x for i,x in enumerate(pixlist) if pixlist_hemi[i]=='right']
        
        #pad matching L/R pairs to the same height
        lhshape=[x.shape for x in pixlist_left]
        rhshape=[x.shape for x in pixlist_right]
        
        for i in range(len(lhshape)):
                hmax=max(lhshape[i][0],rhshape[i][0])
                pixlist_left[i]=padimage(pixlist_left[i],bgcolor=None,padfinalsize=[hmax,-1])
                pixlist_right[i]=padimage(pixlist_right[i],bgcolor=None,padfinalsize=[hmax,-1])
        
        pixlist_stack=np.hstack((np.vstack(pixlist_left),np.vstack(pixlist_right)))
    else:
        pixlist_stack=[]
    

    #######################
    #volume slice (if any)

    bgvolfile="%s/atlases/MNI152_T1_1mm_headmasked.nii.gz" % (getscriptdir())
    bgmaskfile="%s/atlases/MNI152_T1_1mm_headmask.nii.gz" % (getscriptdir())
    volvals=None
    bgvolvals=None
    slicevol_cmap=None
    bgvol_cmap=None
    blank_cmap=ListedColormap(backgroundcolor)

    if slice_dict:
        refnib=nib.load(bgvolfile)
        masknib=nib.load(bgmaskfile)
        bgvolvals=refnib.get_fdata()
        bgmaskvals=np.clip(masknib.get_fdata(),0,1)
        volvals=fill_volume_rois(roivals,atlasinfo,backgroundval=np.nan,referencevolume=refnib)

        #quick (and dirty) test for ax flipping (ex: MNI reference volume is [-1,1,1])
        ax_to_flip=np.where(np.diag(refnib.affine)[:3]<0)[0]
        bgvolvals=np.flip(bgvolvals,ax_to_flip)
        bgmaskvals=np.flip(bgmaskvals,ax_to_flip)
        volvals=np.flip(volvals,ax_to_flip)
        
        bgvol_cmap=plt.get_cmap("gray")
        if colormap is None or isinstance(colormap,str):
            slicevol_cmap=plt.get_cmap(colormap)
        else:
            slicevol_cmap=colormap
    
    imgslice_dict={}
    sliceax={'axial':2,'coronal':1,'sagittal':0}
    for a in ['axial','coronal','sagittal']:
        if not a in slice_dict or slice_dict[a] is None or len(slice_dict[a])==0:
            continue
        if not a in mosaic_dict:
            mosaic_dict[a]=None
        imgslice_dict[a]=slice_volume_to_rgb(volvals,bgvolvals,bgmaskvals,sliceaxis=sliceax[a],slice_indices=slice_dict[a],mosaic=mosaic_dict[a],
                                                   cmap=slicevol_cmap,clim=clim,bg_cmap=bgvol_cmap,blank_cmap=blank_cmap)

    #order slice axes were given
    imgslice_list=[imgslice_dict[k] for k in slicestack_order if k in imgslice_dict]

    #now resize volume slice images to size of surf renderings
    current_image_size=None
    if len(pixlist_stack)>0:
        current_image_size=pixlist_stack.shape
    
    imgslice_list_resized=[]
    for imgslice in imgslice_list:
        imgslice=np.uint8(np.clip(np.round(imgslice.astype(np.float32)*255),0,255))
        imgslice=Image.fromarray(imgslice)

        if current_image_size is None:
            current_image_size=imgslice.size
            #if no surface views, apply upscale factor to volume slices directly
            current_image_size=[int(round(x*upscale_factor)) for x in current_image_size]

        if slicestack_direction.lower()=='horizontal':
            imgscale=current_image_size[0]/imgslice.size[0]
            newsize=[current_image_size[0], int(round(imgslice.size[1]*imgscale))]
        else:
            imgscale=current_image_size[1]/imgslice.size[1]
            newsize=[int(round(imgslice.size[0]*imgscale)), current_image_size[1]]

        imgslice=np.asarray(imgslice.resize(newsize,resample=Image.Resampling.LANCZOS))
        #reorient for display
        imgslice=np.flip(np.transpose(imgslice,[1,0,2]),0)
        imgslice_list_resized+=[imgslice]
    
    #now stack the surf views with the volume slices
    if len(pixlist_stack)>0 and len(imgslice_list_resized) > 0:
        #if surfviews AND voliume slices
        if slicestack_direction=='horizontal':
            pixlist_stack=np.hstack((pixlist_stack,*imgslice_list_resized))
        else:
            pixlist_stack=np.vstack((pixlist_stack,*imgslice_list_resized))
    elif len(imgslice_list_resized) > 0:
        #if only volume slices
        if slicestack_direction=='horizontal':
            pixlist_stack=np.hstack(imgslice_list_resized)
        else:
            pixlist_stack=np.vstack(imgslice_list_resized)
    
    if outputimagefile is not None:
        save_image(pixlist_stack,outputimagefile)
        print("Saved %s" % (outputimagefile))
    
    plt.close(fig)

    return pixlist_stack

def run_montageplot(argv=None):
    if argv is None:
        argv=sys.argv[1:]
    args=parse_argument_montageplot(argv)
    
    inputfile=args.inputfile
    inputfieldname=args.inputfieldname
    viewnames=flatarglist(args.viewnames)
    outputimage=args.outputimage
    surftype=args.surftype
    clim=flatarglist(args.clim)
    cmapname=args.cmapname
    cmapfile=args.cmapfile
    atlasname=args.atlasname
    roilutfile=args.roilutfile
    lhannotfile=args.lhannotfile
    rhannotfile=args.rhannotfile
    no_shading=args.noshading
    annotsurfacename=args.annotsurface
    lhannotprefix=args.lhannotprefix
    rhannotprefix=args.rhannotprefix
    subcortvolfile=args.subcortvol
    
    upscale_factor=args.upscale

    slicearg=args.slices
    stackdirection=args.stackdirection
    slicedict_order=None

    slicemosaic_dict={'axial':args.axmosaic,'coronal':args.cormosaic,'sagittal':args.sagmosaic}
    slicedict={}
    if slicearg is not None:
        slicearg=flatarglist(slicearg)
        slicedict={'axial':[],'coronal':[],'sagittal':[]}
        slicedict_order=[]
        curax=None
        curlist=[]
        for i,s in enumerate(slicearg):
            try:
                s=stringfromlist(s,["axial","coronal","sagittal"])
                if curax is not None:
                    slicedict[curax]+=curlist
                    curlist=[]
                    if not curax in slicedict_order:
                        slicedict_order+=[curax]
                curax=s
            except:
                pass

            v=None
            try:
                #v=float(s)
                v=int(s)
            except:
                pass
            if v is not None:
                curlist+=[v]

        if curax is not None and len(curlist)>0:
            slicedict[curax]+=curlist
            if not curax in slicedict_order:
                slicedict_order+=[curax]

    bgcolor=args.bgcolorname
    if args.bgrgb is not None:
        bgcolor=args.bgrgb
        try:
            tmpcmap=ListedColormap(bgcolor)
        except:
            raise Exception("Invalid --bgrgb entry:",args.bgrgb,". Must be R G B triplet 0-1.0")

    inputvals_arg=flatarglist(args.inputvals)
    
    if len(clim)==2:
        clim=[np.float32(x) for x in clim]
    else:
        clim=None
    
    #nilearn uses 'Spectral' instead of matplotlib 'spectral'
    try:
        cmap=stringfromlist(cmapname,list(plt.colormaps.keys()),allow_startswith=False)
    except:
        cmap=cmapname
    
    if cmapfile is not None:
        cmapdata=np.loadtxt(cmapfile)
        if cmapdata.shape[1]!=3:
            raise Exception("colormap file must have 3 columns")
        if cmapdata.max()>1:
            cmapdata/=255
        cmap=ListedColormap(cmapdata)
        
    #roivals=np.arange(86)+1
    
    if inputfile is None:
        inputfile=""
    
    if len(inputvals_arg)>0:
        roivals=np.array(inputvals_arg).astype(float)
    elif inputfile.lower().endswith(".txt"):
        roivals=np.loadtxt(inputfile)
    elif inputfile.lower().endswith(".mat"):
        
        Mroivals=loadmat(inputfile)
        mkeys=[k for k in Mroivals.keys() if not k.startswith("__")]
        if len(mkeys)==1:
            print("Only one field found in %s. Loading '%s'." % (inputfile,mkeys[0]))
            roivals=Mroivals[mkeys[0]]
        elif inputfieldname in mkeys:
            roivals=Mroivals[inputfieldname]
        else:
            raise Exception("Multiple data fields found in %s. Specify one with --inputfield:",mkeys)
    else:
        raise Exception("Invalid inputfile: %s" % (inputfile))
    
    if atlasname is not None:
        atlasname=args.atlasname.lower()
        atlas_info=retrieve_atlas_info(atlasname)
    else:
        atlas_info={'atlasname':None,'roilutfile':roilutfile,'lhannotfile':lhannotfile,'rhannotfile':rhannotfile,
            'annotsurfacename':annotsurfacename,'lhannotprefix':lhannotprefix,'rhannotprefix':rhannotprefix,'subcorticalvolume':subcortvolfile}
    
    surftype_allowed=['white','infl','pial']
    if not surftype in surftype_allowed:
        surftype_found=[s for s in surftype_allowed if surftype.lower().startswith(s)]
        if len(surftype_found)>0:
            surftype=surftype_found[0]
    
    img=create_montage_figure(roivals,atlasinfo=atlas_info,
        viewnames=viewnames,surftype=surftype,clim=clim,colormap=cmap,noshading=no_shading,
        outputimagefile=outputimage,upscale_factor=upscale_factor,slicestack_direction=stackdirection,
        slice_dict=slicedict,mosaic_dict=slicemosaic_dict,slicestack_order=slicedict_order,
        backgroundcolor=bgcolor)

if __name__ == "__main__":
    run_montageplot(sys.argv[1:])
