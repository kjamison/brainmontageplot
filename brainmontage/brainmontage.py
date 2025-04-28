from nilearn import plotting
from nilearn import datasets
import numpy as np
import nibabel as nib
import nibabel.processing as nibproc
import pandas as pd
from tqdm import tqdm
import warnings

import json
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from datetime import datetime
from PIL import Image, ImageFilter

from matplotlib import use as matplotlib_set_backend
from matplotlib import get_backend as matplotlib_get_backend

import argparse
import sys
import shutil

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
    parser.add_argument('--views',action='append',dest='viewnames',nargs='*',help='list of: dorsal, ventral, medial, lateral, anterior, posterior, or none')
    parser.add_argument('--surftype',action='store',dest='surftype',default='infl',help='inflated, white, pial')
    parser.add_argument('--cmap','--colormap',action='store',dest='cmapname',default='magma')
    parser.add_argument('--cmapfile','--colormapfile',action='store',dest='cmapfile')
    parser.add_argument('--clim', action='append',dest='clim',nargs=2)
    parser.add_argument('--noshading',action='store_true',dest='noshading')
    parser.add_argument('--upscale',action='store',dest='upscale',type=float,default=1,help='Image upscaling factor')
    parser.add_argument('--backgroundcolor','--bgcolor',action='store',dest='bgcolorname',default='white',help='Background color name')
    parser.add_argument('--backgroundrgb','--bgrgb',action='store',dest='bgrgb',type=float,nargs=3,help='Background color R G B (0-1.0)')
    parser.add_argument('--facemode',action='store',dest='facemode',default='mode',help='How to map vertices to faces: mode (default), mean, or best (slow)')
    parser.add_argument('--bestmodeiters',action='store',dest='bestmodeiter',default=5,type=int,help='For "best" facemode, how many selection smoothing iterations?')

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
    slice_arg_group.add_argument('--slicebgalpha',action='store',dest='slice_background_alpha',type=float,default=1,help='Opacity of slice background volume')
    slice_arg_group.add_argument('--slicezoom',action='store',dest='slice_zoom',type=float,default=1,help='Zoom in on slice images (and crop. 1=no zoom. Valid > 1)')

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
   
    cbar_arg_group=parser.add_argument_group('Colorbar options')
    cbar_arg_group.add_argument('--colorbar',action='store_true',dest='colorbar',help='Add colorbar to output image')
    cbar_arg_group.add_argument('--colorbarcolor',action='store',dest='colorbar_color',help='Color to use for colorbar ticks, box, labels')
    cbar_arg_group.add_argument('--colorbarfontsize',action='store',dest='colorbar_fontsize',help='Font size for colorbar')
    cbar_arg_group.add_argument('--colorbarlocation',action='store',dest='colorbar_location',default='right',help='Location for color bar (default=right)')
    cbar_arg_group.add_argument('--colorbartext',action='store',dest='colorbar_label',help='Text to to add to colorbar')
    cbar_arg_group.add_argument('--colorbartextrotation',action='store_true',dest='colorbar_label_rotation',help='Rotate colorbar text 180deg')

    misc_arg_group=parser.add_argument_group('Other options')
    misc_arg_group.add_argument('--version',action='store_true',dest='version')
    misc_arg_group.add_argument('--nolookup',action='store_true',dest='no_lookup',help='Do not use saved lookups (mainly for testing)')
    misc_arg_group.add_argument('--createlookup',action='store_true',dest='create_lookup',help='Create lookup if not found')
    misc_arg_group.add_argument('--clearall',action='store_true',dest='clear_all_cache',help='Clear all stored lookups and facemapping cache')
    misc_arg_group.add_argument('--clearlookups',action='store_true',dest='clear_lookups',help='Clear all stored lookups')
    misc_arg_group.add_argument('--clearfacemaps',action='store_true',dest='clear_facemaps',help='Clear all stored face-mapping cache')
    misc_arg_group.add_argument('--showcache',action='store_true',dest='show_cache',help='Show cache locations')

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
    
    if lhannotfile.endswith(".annot") or lhannotfile.endswith(".label.gii"):
        #for .annot files, we neet the LUT that says which names to include and what order they go in
        #read in the .annot data
        Troi=read_lut(lutfile=roilutfile)

        if lhannotfile.endswith(".annot"):
            lhlabels,ctab,lhnames=nib.freesurfer.io.read_annot(lhannotfile)
            lhnames=[(x.decode('UTF-8')) for x in lhnames]
            lhvals=np.arange(len(lhnames))

            rhlabels,ctab,rhnames=nib.freesurfer.io.read_annot(rhannotfile)
            rhnames=[(x.decode('UTF-8')) for x in rhnames]
            rhvals=np.arange(len(rhnames))

        elif lhannotfile.endswith(".label.gii"):
            lhgii=nib.load(lhannotfile)
            lhlabels=lhgii.agg_data()
            lhdict=lhgii.labeltable.get_labels_as_dict()
            lhnames=[v for k,v in lhdict.items()]
            lhvals=[k for k,v in lhdict.items()]
            
            rhgii=nib.load(rhannotfile)
            rhlabels=rhgii.agg_data()
            rhdict=rhgii.labeltable.get_labels_as_dict()
            rhnames=[v for k,v in rhdict.items()]
            rhvals=[k for k,v in rhdict.items()]

        if lhannotprefix is not None:
            if not any([x.startswith(lhannotprefix) for x in lhnames]):
                lhnames=['%s%s' % (lhannotprefix,x) for x in lhnames]        
        if rhannotprefix is not None:
            if not any([x.startswith(rhannotprefix) for x in rhnames]):
                rhnames=['%s%s' % (rhannotprefix,x) for x in rhnames]

        #now re-fill label values from hemi-specific annot with the values from the full LUT
        lhannotval=np.zeros(Troi.shape[0])
        rhannotval=np.zeros(Troi.shape[0])
        lhlabels_new=np.zeros(lhlabels.shape)
        rhlabels_new=np.zeros(rhlabels.shape)
        for i,n86 in enumerate(Troi['name']):
            lhidx=[j for j,nannot in enumerate(lhnames) if nannot==n86]
            rhidx=[j for j,nannot in enumerate(rhnames) if nannot==n86]
            if len(lhidx)==1:
                lhannotval[i]=lhvals[lhidx[0]]
                lhlabels_new[lhlabels==lhvals[lhidx[0]]]=Troi['label'].iloc[i]
            if len(rhidx)==1:
                rhannotval[i]=rhvals[rhidx[0]]
                rhlabels_new[rhlabels==rhvals[rhidx[0]]]=Troi['label'].iloc[i]
        lhlabels=lhlabels_new
        rhlabels=rhlabels_new
        
    elif lhannotfile.endswith(".shape.gii"):
        lhgii=nib.load(lhannotfile)
        lhlabels=lhgii.agg_data()
        
        rhgii=nib.load(rhannotfile)
        rhlabels=rhgii.agg_data()
        
    surfvalsLR={}
    for hemi in ['left','right']:
        if hemi == 'left':
            labels=lhlabels
        else:
            labels=rhlabels

        parcmask=labels!=0
        uparc,uparc_idx=np.unique(labels[parcmask],return_inverse=True)
        vnew=np.nan*np.ones(labels.shape)
        uparc=uparc.astype(int)-1
        if roivals.size<=uparc.max():
            #expand roivals with nan to maximum needed in parc
            roivals_new=np.nan*np.ones(uparc.max()+1)
            roivals_new[:roivals.size]=roivals
            roivals=roivals_new
        roivals_uparc=roivals[uparc]
        vnew[parcmask]=roivals_uparc[uparc_idx]
        surfvalsLR[hemi]=vnew

    return surfvalsLR

def fill_volume_rois(roivals, atlasinfo, backgroundval=0, referencevolume=None):
    if not 'subcorticalvolume' in atlasinfo or \
        atlasinfo['subcorticalvolume'] is None or \
        atlasinfo['subcorticalvolume'] == "":
        raise Exception("Subcortical volume not found for atlas '%s'" % (atlasinfo["atlasname"]))
    
    Troi=read_lut(lutfile=atlasinfo['roilutfile'])
    
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

    parcmask=Vroi!=0
    uparc,uparc_idx=np.unique(Vroi[parcmask],return_inverse=True)
    Vnew=backgroundval*np.ones(Vroi.shape)
    uparc=uparc.astype(int)-1
    if roivals.size<=uparc.max():
        #expand roivals with nan to maximum needed in parc
        roivals_new=np.nan*np.ones(uparc.max()+1)
        roivals_new[:roivals.size]=roivals
        roivals=roivals_new
    roivals_uparc=roivals[uparc]
    Vnew[parcmask]=roivals_uparc[uparc_idx]

    return Vnew

def map_vertices_to_faces(surfLR,surfvalsLR,face_mode='mean',face_best_mode_iters=1,atlasinfo=None):
    facefile=None
    if atlasinfo is not None and face_mode=='best':
        #look for cached file
        cachedir=get_data_dir('facemap')
        if not os.path.exists(cachedir):
            os.mkdir(cachedir)
        facefile="%s/facemapping_%s_%s_%diter.mat" % (cachedir,atlasinfo['atlasname'],atlasinfo['annotsurfacename'],face_best_mode_iters)
        if os.path.exists(facefile):
            Mface=loadmat(facefile,simplify_cells=True)
            cache_matches=True
            #check if cached file has identical vertex ROI info
            #if not, regenerate mapping
            for h in ['left','right']:
                v=surfvalsLR[h]
                vcache=Mface['surfvalsLR'][h]
                m_notnan=np.logical_not(np.isnan(vcache))
                if v.shape != vcache.shape:
                    cache_matches=False
                    break

                if not np.all(v[m_notnan]==vcache[m_notnan]):
                    cache_matches=False
                    break
            if cache_matches:
                return Mface['facevalsLR']

    if face_mode == 'mean' and atlasinfo is not None and atlasinfo['atlasname']=='cifti91k':
        warnings.warn("face_mode='mean' cannot be used with cifti91k. Using 'mode' instead.")
        face_mode='mode'
    
    facevalsLR={}

    for h in ['left','right']:      
        if face_mode == 'mean':
            #interp each face as average of vertices
            facevalsLR[h]=vert2face(surf=surfLR[h],vertvals=surfvalsLR[h])
        elif face_mode == 'mode':
            #face = mode (if 2 vs 1) or vert 3 (if all 3 are different)
            facevertvals=surfvalsLR[h][surfLR[h][1]] #face x 3
            facevalsLR[h]=facevertvals[:,2]
            m12=facevertvals[:,0]==facevertvals[:,1]
            facevalsLR[h][m12]=facevertvals[m12,0]
        elif face_mode == 'best':
            #1. make vert x roi sparse matrix 
            #2. smooth those vertex masks
            #3. map smooth vertex masks to faces (interp)
            #4. argmax to find the ROI for each face
            T_vert2face=vert2face(surf=surfLR[h])
            T_diffuse=mesh_diffuse(surf=surfLR[h])
            uvals,uidx=np.unique(surfvalsLR[h],return_inverse=True)
            vert_by_roi=sparse.csr_matrix((np.ones(uidx.shape),(range(len(uidx)),uidx)),shape=(len(uidx),len(uvals)))

            for hifi_iter in range(face_best_mode_iters):
                vert_by_roi = T_diffuse @ vert_by_roi
            face_by_roi=T_vert2face @ vert_by_roi
            facevalsLR[h]=uvals[np.argmax(face_by_roi,axis=1)][:,0]
        else:
            raise Exception("Unknown face_mode: %s" % (face_mode))
    
    if facefile is not None:
        savemat(facefile,{'surfvalsLR':surfvalsLR,'facevalsLR':facevalsLR},format='5',do_compression=True)
        print("Saved cached face-mapping file %s" % (facefile))
    
    return facevalsLR

def read_lut(lutfile=None,atlasname=None,atlasinfo=None):
    expected_lut_columns=['label','name','R','G','B','A']
    if atlasinfo is None and atlasname is not None:
        atlasinfo=retrieve_atlas_info(atlasname)
    if lutfile is None and atlasinfo is not None:
        lutfile=atlasinfo['roilutfile']
    if lutfile.lower().endswith(".mat"):
        Xlut=loadmat(lutfile,simplify_cells=True)
        if 'lut' in Xlut:
            Xlut=Xlut['lut']
        elif 'LUT' in Xlut:
            Xlut=Xlut['LUT']
        else:
            raise Exception("LUT.mat file must have 'lut' field with ROI x [label,name,R,G,B]")

        lut_columns=expected_lut_columns[:Xlut.shape[1]]
        Troi=pd.DataFrame(Xlut,columns=lut_columns)
        for c in expected_lut_columns[Xlut.shape[1]:]:
            Troi[c]=np.nan*np.ones(Troi.shape[0])
    else:
        Troi=pd.read_table(lutfile,delimiter='\s+',header=None,names=expected_lut_columns)

    #remove any entries for 'Unknown' or '???'
    Troi=Troi[Troi['name']!='Unknown']
    Troi=Troi[Troi['name']!='???']

    return Troi

def retrieve_atlas_info(atlasname, lookup=False, atlasinfo_jsonfile=None, scriptdir=None):
    if scriptdir is None:
        scriptdir=getscriptdir()
    
    if atlasinfo_jsonfile is None:
        atlasinfo_jsonfile="%s/atlas_info.json" % (get_data_dir("atlas"))
    
    roicount=None
    lhannotprefix=""
    rhannotprefix=""
    subcortfile=""
    lookupsurface=None
    lhlookupannotfile=None
    rhlookupannotfile=None
    
    with open(atlasinfo_jsonfile,'r') as f:
        atlas_info_list=json.load(f)
    
    if atlasname == 'list':
        return list(atlas_info_list.keys())
    
    for a in atlas_info_list.keys():
        if 'aliases' in atlas_info_list[a] and atlasname in atlas_info_list[a]['aliases']:
            atlasname=a
            break
    
    if atlasname in atlas_info_list:
        roicount=atlas_info_list[atlasname]['roicount']
        roilutfile=atlas_info_list[atlasname]['roilut'].replace('%SCRIPTDIR%',scriptdir)
        lhannotfile=atlas_info_list[atlasname]['lhannot'].replace('%SCRIPTDIR%',scriptdir)
        rhannotfile=atlas_info_list[atlasname]['rhannot'].replace('%SCRIPTDIR%',scriptdir)
        annotsurfacename=atlas_info_list[atlasname]['annotsurface']

        if 'lhannotprefix' in atlas_info_list[atlasname]:
            lhannotprefix=atlas_info_list[atlasname]['lhannotprefix']
        if 'rhannotprefix' in atlas_info_list[atlasname]:
            rhannotprefix=atlas_info_list[atlasname]['rhannotprefix']

        if 'subcorticalvolume' in atlas_info_list[atlasname]:
            subcortfile=atlas_info_list[atlasname]['subcorticalvolume'].replace('%SCRIPTDIR%',scriptdir)
        
        if "annotsurface_lookup" in atlas_info_list[atlasname]:
            lookupsurface=atlas_info_list[atlasname]['annotsurface_lookup']
        if "lhannot_lookup" in atlas_info_list[atlasname]:
            lhlookupannotfile=atlas_info_list[atlasname]['lhannot_lookup'].replace('%SCRIPTDIR%',scriptdir)
        if "rhannot_lookup" in atlas_info_list[atlasname]:
            rhlookupannotfile=atlas_info_list[atlasname]['rhannot_lookup'].replace('%SCRIPTDIR%',scriptdir)
    else:
        raise Exception("atlas name '%s' not found. Choose from %s" % (atlasname, ",".join(atlas_info_list.keys())))
    
    if lookup and lookupsurface is not None and lhlookupannotfile is not None and rhlookupannotfile is not None:
        lhannotfile=lhlookupannotfile
        rhannotfile=rhlookupannotfile
        annotsurfacename=lookupsurface
        
    atlasinfo={'atlasname':atlasname,'roicount':roicount,'roilutfile':roilutfile,'lhannotfile':lhannotfile,'rhannotfile':rhannotfile,
        'annotsurfacename':annotsurfacename,'lhannotprefix':lhannotprefix,'rhannotprefix':rhannotprefix,
        'subcorticalvolume':subcortfile}
    
    return atlasinfo

def generate_surface_view_lookup(surf=None,hemi=None,azel=None,figsize=None,figdpi=200,lightdir=None,shading_smooth_iters=1,surfbgvals=None):
    
    if figsize is None:
        figsize=(6.4,6.4)

    faces=surf[1]
    T_vert2face=vert2face(faces=faces)

    facecount=faces.shape[0]

    #convert face index to RGB triplet
    face_idx=np.arange(facecount)+1 #use 1-based index here so we know 0=no face
    rgbface_idx=np.stack((face_idx % 256, face_idx // 256 % 256, face_idx // (256*256)),axis=-1)
    f_rgb=rgbface_idx/255.0

    surfvals=np.zeros(surf[0].shape[0]) #per vertex

    colormap='gray'
    clim=[0,1]

    #explicitly set backend to ensure dpi/rendering consistent across installations
    current_backend=matplotlib_get_backend()
    matplotlib_set_backend('Agg')

    backgroundcolor=[0,0,0]
    figure=plt.figure(figsize=figsize,facecolor=backgroundcolor,dpi=figdpi)
    v=plotting.plot_surf_roi(surf, roi_map=surfvals,
                            bg_map=surfbgvals, bg_on_data=False,
                            darkness=.5,cmap=colormap, colorbar=False, vmin=clim[0], vmax=clim[1],
                            figure=figure)

    #also explicitly set xyz aspect to ensure consistent render scale across installations
    v.get_axes()[-1].set_box_aspect([1,1,.66])

    hmesh=v.findobj(lambda obj: isinstance(obj, Poly3DCollection))[0]

    v.get_axes()[-1].set_facecolor(backgroundcolor)
    if azel is not None:
        v.get_axes()[-1].view_init(azim=azel[0],elev=azel[1])

    imgbgvals=None
    if surfbgvals is not None:
        imgbgvals=fig2pixels(figure,dpi=figdpi)
        imgbgvals=imgbgvals[:,:,:3].astype(np.float32)/255.0


    hmesh.set_facecolor(f_rgb)
    img=fig2pixels(figure,dpi=figdpi).astype(int)

    #convert rgb triplets back to index
    img= img[:,:,0] + img[:,:,1]*256 + img[:,:,2]*256*256
    
    imgnonzero=(img>0).astype(float)
    imgmaxidx=img.astype(int)-1 #back to zero-based ROI now
    imgmaxidx[imgmaxidx<0]=0
    
    #render a mask for the whole surface (white) on black bg
    v.get_axes()[-1].set_facecolor([0,0,0])
    hmesh.set_facecolor('white')
    imgmask=fig2pixels(figure,dpi=figdpi)
    imgmask=imgmask[:,:,:3].astype(np.float32)/255.0
    imgmask=np.mean(imgmask,axis=2)

    ############
    #shading
    imgshading=None
    if lightdir is not None:
        shadingvals=mesh_shading(surf[0], surf[1], lightdir)
        shadingvals,_=mesh_diffuse(vertvals=shadingvals,verts=surf[0],faces=surf[1],iters=shading_smooth_iters)

        #adjust computed shading values to look brighter
        shadingvals=np.minimum(shadingvals**.5,.8)/.8

        shadingvals=shadingvals/shadingvals.max()

        f=T_vert2face @ shadingvals #convert back to faces for this operation

        shading_cmap = plt.get_cmap("gray")
        f_rgb=shading_cmap(f)

        v.get_axes()[-1].set_facecolor('white')
        hmesh.set_facecolor(f_rgb)
        imgshading=fig2pixels(v,dpi=figdpi)

        imgshading=imgshading[:,:,:3].astype(np.float32)/255.0
        imgshading=np.mean(imgshading[:,:,:3],axis=2,keepdims=True)
        #pixshading=pixshading/pixshading.max()
        imgshading=np.clip(imgshading/imgshading[imgshading<1.0].max(),0,1)


    ############
    plt.close(figure)

    #restore previous backend
    matplotlib_set_backend(current_backend)

    imgmask,cropbox=cropbg(imgmask)
    imgmaxidx=imgmaxidx[cropbox[0]:cropbox[1],cropbox[2]:cropbox[3]]
    imgnonzero=imgnonzero[cropbox[0]:cropbox[1],cropbox[2]:cropbox[3]]
    if imgshading is None:
        imgshading=[]
    else:
        imgshading=imgshading[cropbox[0]:cropbox[1],cropbox[2]:cropbox[3],:]
    if imgbgvals is None:
        imgbgvals=[]
    else:
        imgbgvals=imgbgvals[cropbox[0]:cropbox[1],cropbox[2]:cropbox[3],:]

    lookup={'roimap':imgmaxidx,'roinonzero':imgnonzero,'shading':imgshading,'mask':imgmask,'brainbackground':imgbgvals}
    return lookup

def save_mesh_lookup_file(lookup_surface_name, surftype='infl',viewnames='all',figsize=(6.4,6.4),figdpi=200, shading=True, only_shading=False, overwrite_existing=True):

    if isinstance(viewnames,str):
        viewnames=[viewnames]

    if 'all' in viewnames:
        viewnames=['dorsal','lateral','medial','ventral','anterior','posterior']

    lookup_dir=get_data_dir('lookup')
    lookup_file="%s/meshlookup_%s_%s.mat" % (lookup_dir,surftype,lookup_surface_name)

    if not os.path.exists(lookup_dir):
        os.mkdir(lookup_dir)

    if os.path.exists(lookup_file):
        if overwrite_existing:
            print("Lookup already exists. Overwriting: %s" % (lookup_file))
        else:
            print("Lookup already exists. Not overwriting: %s" % (lookup_file))
            return lookup_file
    

    shading = shading or only_shading
    #use larger values here for high-res fsaverage 
    shading_smooth_iters=2
    if surftype == 'infl':
        shading_smooth_iters=20

    fsaverage = fetch_surface_dataset(mesh=lookup_surface_name)
    surfbgvalsLR={'left':fsaverage['sulc_left'],'right':fsaverage['sulc_right']}

    surfLR={}
    surfLR['left']=nib.load(fsaverage[surftype+'_'+'left']).agg_data()
    surfLR['right']=nib.load(fsaverage[surftype+'_'+'right']).agg_data()

    
    lookup={}
    lookup['info']={'mesh':lookup_surface_name,'dpi':figdpi,'figsize':figsize,
                    'surftype':surftype,'timestamp':datetime.now().strftime("%Y%m%d_%H%M%S")}

    print("Generating lookup for %s %s" % (lookup_surface_name,surftype))

    progress= tqdm([(h,v) for h in ['left','right'] for v in viewnames])
    viewkeys=[]

    for h,viewname in progress:
        if viewname == 'none':
            continue
        azel=get_view_azel(h,viewname)
        lightdir=get_light_dir(h,viewname)

        k="%s_%s" % (h,viewname)
        viewkeys+=[k]
        
        surfbgvals=surfbgvalsLR[h]

        if not shading:
            surfbgvals=None
            lightdir=None

        progress.set_description("%s" % (k))

        lookup[k] = generate_surface_view_lookup(surfLR[h],hemi=h,surfbgvals=surfbgvals,azel=azel,lightdir=lightdir,
                                    shading_smooth_iters=shading_smooth_iters,figsize=figsize,figdpi=figdpi)
        
    lookup['views']=np.array(viewkeys,dtype=object)

    if shading and only_shading:
        #deep copy shading info to new struct before deleting from roi lookup
        for k in viewkeys:
            del lookup[k]['roimap']
            del lookup[k]['roinonzero']

    if not shading:
        for k in viewkeys:
            del lookup[k]['shading']
            del lookup[k]['brainbackground']

    savemat(lookup_file,lookup,format='5',do_compression=True)
    print("Saved %s" % (lookup_file))

    return lookup_file

def render_surface_lookup(roivals,lookup,cmap='magma',clim=None,backgroundcolor=None,shading=True,braincolor=None, borderimage=None, bordercolor='black',borderwidth=1):
    if backgroundcolor is None:
        backgroundcolor=[0,0,0]

    if clim is None:
        clim=[np.nanmin(roivals),np.nanmax(roivals)]
    
    blank_cmap=ListedColormap(backgroundcolor)

    roiimg=roivals[lookup['roimap']]
    if isinstance(cmap,str):
        cmap=plt.get_cmap(cmap)
    
    roiimg_rgb=cmap((roiimg-clim[0])/(clim[1]-clim[0]))
    
    if braincolor is None and 'brainbackground' in lookup:
        brainbg_rgb=lookup['brainbackground']
        if roiimg_rgb.shape[2]==4:
            brainbg_rgb=np.concatenate((brainbg_rgb,np.ones(roiimg_rgb.shape[:2])[:,:,np.newaxis]),axis=2)
    else:
        if braincolor is None:
            braincolor='gray'
        brainbg_cmap=ListedColormap(braincolor)
        brainbg_rgb=brainbg_cmap(np.ones(roiimg.shape))
    
    imgnonzero=np.atleast_3d(np.logical_and(lookup['roinonzero'],np.logical_not(np.isnan(roiimg))).astype(float))
    
    ###############
    if borderimage is not None:
        #perform gaussian blur on border image
        if borderwidth>0:
            border_edgefilter_gauss=ImageFilter.GaussianBlur(radius=borderwidth)
            borderimage=Image.fromarray(borderimage).convert("L").filter(border_edgefilter_gauss)
            borderimage=np.asarray(borderimage)
            if np.any(borderimage>1):
                borderimage=np.clip(borderimage/255,0,.01)/.01
            else:
                borderimage=np.clip(borderimage,0,.01)/.01
        borderimage=np.atleast_3d(borderimage)
        #make a blank image that is just the desired color at every pixel
        border_cmap=ListedColormap(bordercolor)
        rgbborder=val2rgb(np.ones(borderimage.shape[:2]),border_cmap,[0,1])
        rgbborder[:,:,3]=1
        #rgbborder=np.uint8(rgbborder)
        
        roiimg_rgb=np.clip(roiimg_rgb*(1-borderimage)+rgbborder*borderimage,0,1)
        
    ###############
    roiimg_rgb=roiimg_rgb*imgnonzero + brainbg_rgb*(1-imgnonzero)
    
    imgalpha=np.atleast_3d(lookup['mask'])

    blank_rgb=blank_cmap(np.ones(roiimg.shape))
    
    newimg_rgb=roiimg_rgb*imgalpha + blank_rgb*(1-imgalpha)

    
    if shading and 'shading' in lookup:
        imgshading=np.atleast_3d(lookup['shading'])
        if newimg_rgb.shape[2]==4:
            #dont apply shading to alpha channel
            alphachan=newimg_rgb[:,:,3].copy()
        newimg_rgb*=imgshading
        if newimg_rgb.shape[2]==4:
            newimg_rgb[:,:,3]=alphachan
    
    newimg_rgb=np.uint8(np.clip(np.round(newimg_rgb*255),0,255))

    return newimg_rgb

def render_surface_view(surfvals,surf,azel=None,surfbgvals=None,shading=True,lightdir=None,val_smooth_iters=0,shading_smooth_iters=1,
                        colormap=None, clim=None,
                        backgroundcolor=None,figsize=None,figure=None,figdpi=None):

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
        
        pixshading=fig2pixels(v,dpi=figdpi)
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

    pix=fig2pixels(v,dpi=figdpi)

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
    
    pixbg=fig2pixels(v,dpi=figdpi)
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

def slice_volume_to_rgb(volvals,bgvolvals,bgmaskvals,sliceaxis,slice_indices,mosaic,cmap,clim,bg_cmap,blank_cmap,background_alpha=1,
                        borderimage=None,bordercolor='black',borderwidth=1, slice_zoom=None):
    imgslice,mosaicinfo=vol2mosaic(volvals, sliceaxis=sliceaxis, slice_indices=slice_indices, mosaic=mosaic,extra_slice_val=np.nan, slice_zoom_factor=slice_zoom)
    imgslice_brainbg,_=vol2mosaic(bgvolvals,sliceaxis=sliceaxis,slice_indices=mosaicinfo['slice_indices'],mosaic=mosaicinfo['mosaic'], slice_zoom_box=mosaicinfo['slice_zoom_box'])
    imgslice_brainmask,_=vol2mosaic(bgmaskvals,sliceaxis=sliceaxis,slice_indices=mosaicinfo['slice_indices'],mosaic=mosaicinfo['mosaic'], slice_zoom_box=mosaicinfo['slice_zoom_box'])
        
    rgbslice=val2rgb(imgslice,cmap,clim)
    rgbslice_brainbg=val2rgb(imgslice_brainbg,bg_cmap)
    rgbblank=val2rgb(np.zeros(imgslice.shape),blank_cmap,[0,1])
    imgslice_alpha=np.atleast_3d(np.logical_not(np.isnan(imgslice)))
    imgslice_brainmask=np.atleast_3d(imgslice_brainmask)
    #combine blank and bgvol using headmask and global alpha
    rgbslice_background=rgbblank*(1-imgslice_brainmask*background_alpha)+rgbslice_brainbg*(imgslice_brainmask*background_alpha)
    #now mix data volume
    rgbslice=rgbslice_background*(1-imgslice_alpha)+rgbslice*(imgslice_alpha)


    if borderimage is not None:
        #perform gaussian blur on border image
        if borderwidth>0:
            border_edgefilter_gauss=ImageFilter.GaussianBlur(radius=borderwidth)
            borderimage=Image.fromarray(borderimage).convert("L").filter(border_edgefilter_gauss)
            borderimage=np.asarray(borderimage)
            if np.any(borderimage>1):
                borderimage=np.clip(borderimage/255,0,.01)/.01
            else:
                borderimage=np.clip(borderimage,0,.01)/.01
        borderimage=np.atleast_3d(borderimage)
        #make a blank image that is just the desired color at every pixel
        border_cmap=ListedColormap(bordercolor)
        rgbborder=val2rgb(np.ones(borderimage.shape[:2]),border_cmap,[0,1])
        rgbborder[:,:,3]=1
        #rgbborder=np.uint8(rgbborder)
        
        rgbslice=np.clip(rgbslice*(1-borderimage)+rgbborder*borderimage,0,1)
        
    return rgbslice

def add_colorbar_to_image(img,colorbar_color=None,colorbar_fontsize=None,colorbar_location=None,colorbar_label=None,colorbar_label_rotation=False,padding=None,figdpi=None,colormap=None,clim=None,backgroundcolor=None):
    
    #new figure with extra size for colorbar (will crop extra later)
    cbar_extra_scale=1.5
    
    ticksetter=None
    labelsetter=None

    colorbar_label_rotation_default=0
    colorbar_label_verticalalignment='top'

    if colorbar_location is None or colorbar_location == 'right':
        newfigsize=[cbar_extra_scale*img.shape[1]/figdpi,img.shape[0]/figdpi]
        newaxsize=[0,0,1/cbar_extra_scale,1]
        cbar_w=.05
        cbar_hpad=.03
        cbar_vpad=.03
        cbar_inset_size=[1+cbar_hpad, cbar_vpad, cbar_w, 1-2*cbar_vpad]
        cbar_orientation="vertical"
        colorbar_label_rotation_default=90
        if colorbar_label_rotation:
            colorbar_label_verticalalignment='bottom'

    elif colorbar_location == 'left':
        newfigsize=[cbar_extra_scale*img.shape[1]/figdpi,img.shape[0]/figdpi]
        newaxsize=[1-1/cbar_extra_scale,0,1/cbar_extra_scale,1]
        cbar_w=.05
        cbar_hpad=.03
        cbar_vpad=.03
        cbar_inset_size=[-cbar_w-cbar_hpad, cbar_vpad, cbar_w, 1-2*cbar_vpad]
        cbar_orientation="vertical"
        colorbar_label_rotation_default=90
        if not colorbar_label_rotation:
            colorbar_label_verticalalignment='bottom'
        
        ticksetter=lambda hcbar:hcbar.ax.tick_params(left=True,right=False,labelleft=True,labelright=False)
        labelsetter=lambda hcbar:hcbar.ax.yaxis.set_label_position('left')
        
    elif colorbar_location == 'top':
        newfigsize=[img.shape[1]/figdpi,cbar_extra_scale*img.shape[0]/figdpi]
        newaxsize=[0,0,1,1/cbar_extra_scale]
        cbar_h=.05
        cbar_vpad=0.03       
        cbar_hpad=0.03     
        cbar_inset_size=[cbar_hpad, 1+cbar_vpad, 1-2*cbar_hpad, cbar_h ]
        cbar_orientation="horizontal"
        colorbar_label_verticalalignment='bottom'
        ticksetter=lambda hcbar:hcbar.ax.tick_params(top=True,bottom=False,labeltop=True,labelbottom=False)
        labelsetter=lambda hcbar:hcbar.ax.xaxis.set_label_position('top')
        
    elif colorbar_location == 'bottom':
        newfigsize=[img.shape[1]/figdpi,cbar_extra_scale*img.shape[0]/figdpi]
        newaxsize=[0,1-1/cbar_extra_scale,1,1/cbar_extra_scale]
        cbar_h=.05
        cbar_vpad=0.03
        cbar_hpad=.03
        cbar_inset_size=[cbar_hpad, -cbar_vpad-cbar_h, 1-2*cbar_hpad, cbar_h ]
        cbar_orientation="horizontal"
        
    if colorbar_label_rotation:
        colorbar_label_rotation=colorbar_label_rotation_default + 180
    else:
        colorbar_label_rotation=colorbar_label_rotation_default

    fig=plt.figure(figsize=newfigsize,dpi=figdpi,facecolor=backgroundcolor)
    plt.axis('off')
    ax=fig.gca()
    ax.set_position(newaxsize)
    himg=ax.imshow(img,cmap=colormap,vmin=clim[0],vmax=clim[1])
    inset_axesaxins = ax.inset_axes(cbar_inset_size,transform=ax.transAxes)
    hcbar=plt.colorbar(himg,cax=inset_axesaxins,orientation=cbar_orientation)
    hcbar.ax.tick_params(axis='y', direction='in',labelsize=colorbar_fontsize)
    hcbar.ax.tick_params(axis='x', direction='in',labelsize=colorbar_fontsize)

    if ticksetter is not None:
        ticksetter(hcbar)
    if colorbar_label is not None and labelsetter is not None:
        labelsetter(hcbar)

    if colorbar_label is not None:
        #right: rotation=90, verticalalignment='top' is the default, which reads bottom to top
        #right: rotation=270, verticalalignment='bottom' reads top to bottom
        hcbar.set_label(colorbar_label,fontsize=colorbar_fontsize,rotation=colorbar_label_rotation,verticalalignment=colorbar_label_verticalalignment)

    if colorbar_color is not None:
        hcbar.ax.tick_params(color=colorbar_color,labelcolor=colorbar_color)
        hcbar.outline.set_color(colorbar_color)

    newimg=fig2pixels(fig,dpi=figdpi)
    plt.close(fig)
    
    _,cropcoord=cropbg(newimg)
    
    #cropcoord[0]=0
    #cropcoord[2]=0
    newimg=cropbg(newimg,cropcoord=cropcoord)
    newimg=padimage(newimg,padamount=padding)
    
    return newimg
    
def create_montage_figure(roivals,atlasinfo=None, atlasname=None,
    roilutfile=None,lhannotfile=None,rhannotfile=None,annotsurfacename='fsaverage5',lhannotprefix=None, rhannotprefix=None, subcorticalvolume=None,
    viewnames=None,surftype='infl',clim=None,colormap=None, noshading=False, upscale_factor=1, backgroundcolor="white",
    slice_dict={}, mosaic_dict={},slicestack_order=['axial','coronal','sagittal'],slicestack_direction='horizontal', slice_background_alpha=1, slice_zoom=None,
    outputimagefile=None, figdpi=200, no_lookup=False, create_lookup=False,face_mode='mode',face_best_mode_iters=5,
    add_colorbar=False, colorbar_color=None, colorbar_fontsize=None,colorbar_location='right',colorbar_label=None, colorbar_label_rotation=False,
    border_roimask=None, border_color='black',border_width=1):

    #default factor=1 is way too big in general, so for surface views scale this down by 25%
    surface_scale_factor=upscale_factor*.25
    #for slice views with surface, we just scale to match surface view
    #but for slice-only, use the upscale_factor argument (as-is, scale=1 is fine for slice-only)
    slice_only_scale_factor=upscale_factor
    
    border_scale_factor=.5 #border=1 is a good number, but we should scale that to draw at 0.5
    border_width=border_width*border_scale_factor
    
    if slice_zoom is None:
        slice_zoom=1
    
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
        viewnames=['dorsal','lateral','medial','ventral','anterior','posterior']
    elif "all" in [v.lower() for v in viewnames]:
        viewnames=['dorsal','lateral','medial','ventral','anterior','posterior']
    
    try:
        viewnames=stringfromlist(viewnames,['none','dorsal','lateral','medial','ventral','anterior','posterior','flat'])
    except:
        raise Exception("Viewname must be one of: none, dorsal, lateral, medial, ventral, anterior, posterior, flat")
    
    if atlasname is not None and atlasinfo is None:
        atlasinfo=retrieve_atlas_info(atlasname)
    
    if atlasinfo is None:
        atlasinfo={}
        atlasinfo['atlasname']=None
        atlasinfo['roilutfile']=roilutfile
        atlasinfo['lhannotfile']=lhannotfile
        atlasinfo['rhannotfile']=rhannotfile
        atlasinfo['annotsurfacename']=annotsurfacename
        atlasinfo['lhannotprefix']=lhannotprefix
        atlasinfo['rhannotprefix']=rhannotprefix
        atlasinfo['subcorticalvolume']=subcorticalvolume

    atlasname=atlasinfo['atlasname']

    if isinstance(colormap,str) and colormap.lower()=='lut' and atlasinfo['roilutfile'] is not None:
        #generate new colormap and roivals from LUT
        if not os.path.exists(atlasinfo['roilutfile']):
            raise Exception("ROI LUT file must exist for lut cmap option")
        
        Troi=read_lut(lutfile=atlasinfo['roilutfile'])
        cmapdata=np.stack((Troi['R'],Troi['G'],Troi['B']),axis=-1).astype(float)

        if np.all(np.isnan(cmapdata)):
            cmapdata=np.random.random(cmapdata.shape)

        if cmapdata.max()>1:
            cmapdata/=255
        
        cmapdata=np.clip(cmapdata,0,1)

        print("For cmapfile=%s, override input values and clim to display LUT colormap: %s." % (colormap,atlasinfo['roilutfile']))
        colormap=ListedColormap(cmapdata)
        roivals=np.arange(cmapdata.shape[0])+1
        clim=[0.5,cmapdata.shape[0]+.5]
    
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
        if slicestack_order is None:
            slicestack_order=['axial','coronal','sagittal']
        else:
            raise Exception("Slicestack order options must be one of: axial, coronal, sagittal")
    

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
    
    if no_lookup:
        lookup_dict=None
    else:
        if atlasname is None:
            atlasinfo_lookup=atlasinfo
        else:
            atlasinfo_lookup=retrieve_atlas_info(atlasname,lookup=True)
        lookup_surface_name=atlasinfo_lookup['annotsurfacename']
        lookup_dir=get_data_dir('lookup')
        lookup_file="%s/meshlookup_%s_%s.mat" % (lookup_dir,surftype,lookup_surface_name)

        if os.path.exists(lookup_file):
            lookup_dict=loadmat(lookup_file,simplify_cells=True)
        else:
            #raise Exception("Lookup not found: %s" % (lookup_file))
            print("Lookup not found: %s" % (lookup_file))
            lookup_dict=None

        if lookup_dict is None and create_lookup:
            print("Creating lookup for %s %s (this may take up to 10 minutes the first time...)" % (lookup_surface_name,surftype))
            
            lookup_file=save_mesh_lookup_file(lookup_surface_name=lookup_surface_name,surftype=surftype,viewnames='all',shading=True,figdpi=figdpi)
            lookup_dict=loadmat(lookup_file,simplify_cells=True)

        if lookup_dict is not None:
            atlasinfo=atlasinfo_lookup

    fsaverage = fetch_surface_dataset(mesh=atlasinfo['annotsurfacename'])
    surfLR={}
    surfLR['left']=nib.load(fsaverage[surftype+'_'+'left']).agg_data()
    surfLR['right']=nib.load(fsaverage[surftype+'_'+'right']).agg_data()

    fig=None
    pixlist=[]
    pixlist_hemi=[]
    pixlist_view=[]


    if border_roimask is not None and len(border_roimask)!=len(roivals):
        raise Exception("Border roimask must be same length as roivals")
    
    if border_roimask is not None:
        print("Rendering ROI borders for %d/%d ROIs. This might slow down rendering." % (np.sum(border_roimask>0),len(roivals)))
    
    #for rendering ROI borders
    border_bgcolor=[0,0,0]
    border_colormap=ListedColormap([[0,0,0],[1,1,1]])
    border_clim=[0,1]
    border_edgefilter=ImageFilter.Kernel((3,3),(-1, -1, -1, -1, 8,-1, -1, -1, -1), 1, 0)
    
    if lookup_dict is not None:
        #first map roi indices to surface vertices
        #then map those roi indices to to faces
        # (if using 'best', this will check for cached version to speed things even more)
        surfroisLR=fill_surface_rois(np.arange(len(roivals)),atlasinfo)
        faceroisLR=map_vertices_to_faces(surfLR=surfLR,surfvalsLR=surfroisLR,
                                         face_mode=face_mode,face_best_mode_iters=face_best_mode_iters,
                                         atlasinfo=atlasinfo)

        facevalsLR={}
        for h in ['left','right']:
            m_notnan=np.logical_not(np.isnan(faceroisLR[h]))
            facevalsLR[h]=np.ones(faceroisLR[h].shape)*np.nan
            facevalsLR[h][m_notnan]=roivals_rescaled[faceroisLR[h][m_notnan].astype(int)]

        for ih,h in enumerate(['left','right']):
            for iv,viewname in enumerate(viewnames):
                if viewname == 'none':
                    continue

                viewkey="%s_%s" % (h,viewname)
                
                #########
                #render ROI borders
                roi_edge_mask=None
                if border_roimask is not None:
                    for r,rmask in enumerate(border_roimask):
                        if rmask==0 or not np.any(faceroisLR[h]==r):
                            #ROI is not contained in this cortical hemisphere
                            continue
                        
                        pix_roi=render_surface_lookup(faceroisLR[h]==r,lookup_dict[viewkey],cmap=border_colormap,clim=border_clim,
                                                backgroundcolor=border_bgcolor,shading=False,braincolor=None)
                        
                        if np.min(pix_roi[:,:,0])==np.max(pix_roi[:,:,0]):
                            #ROI is not shown in this view
                            continue
                        
                        pix_edge=Image.fromarray(pix_roi).convert("L").filter(border_edgefilter)
                        pix_edge=np.asarray(pix_edge)
                        if roi_edge_mask is None:
                            roi_edge_mask=pix_edge
                        else:
                            roi_edge_mask=np.maximum(roi_edge_mask,pix_edge)
                ##########
                pix=render_surface_lookup(facevalsLR[h],lookup_dict[viewkey],cmap=colormap,clim=clim_rescaled,
                            backgroundcolor=backgroundcolor,shading=shading,braincolor=None,
                            borderimage=roi_edge_mask,bordercolor=border_color, borderwidth=border_width/surface_scale_factor)

                pix=padimage(pix,bgcolor=None,padamount=1)

                if surface_scale_factor != 1:
                    newsize=[np.round(x*surface_scale_factor).astype(int) for x in pix.shape[:2]]
                    newsize=[newsize[1],newsize[0]]
                    pix=np.asarray(Image.fromarray(pix).resize(newsize,resample=Image.Resampling.LANCZOS))
                    
                pixlist+=[pix]
                pixlist_hemi+=[h]
                pixlist_view+=[viewname]
    else:
        if not no_lookup:
            print("Using real-time offscreen render. This is slower and lower resolution than lookup mode. Consider adding --createlookup")
        surfvalsLR=fill_surface_rois(roivals_rescaled,atlasinfo)
        surfbgvalsLR={'left':fsaverage['sulc_left'],'right':fsaverage['sulc_right']}

        shading_smooth_iters=1
        if surftype == 'infl':
            shading_smooth_iters=10

        fig=plt.figure(figsize=(6.4,6.4),facecolor=backgroundcolor)
        figsize=fig.get_size_inches()
        figsize=[x*surface_scale_factor for x in figsize]
        fig.set_size_inches(figsize)

        for ih,h in enumerate(['left','right']):
            for iv, viewname in enumerate(viewnames):
                if viewname == 'none':
                    continue
                azel=get_view_azel(h,viewname)
                lightdir=get_light_dir(h,viewname)
                
                pix=render_surface_view(surfvals=surfvalsLR[h],surf=surfLR[h],surfbgvals=surfbgvalsLR[h],
                                        azel=azel,lightdir=lightdir,shading=shading,shading_smooth_iters=shading_smooth_iters,
                                        colormap=colormap, clim=clim_rescaled,
                                        figure=fig,figdpi=figdpi)

                pix=padimage(pix,bgcolor=None,padamount=1)

                pixlist+=[pix]
                pixlist_hemi+=[h]
                pixlist_view+=[viewname]
    
    #pad all surface views to the same width
    if len(pixlist)>0:
        
        if "anterior" in pixlist_view and "posterior" in pixlist_view and "left" in pixlist_hemi and "right" in pixlist_hemi:
            #special case for left+right anterior+posterior
            idx_left_ant=[i for i,x in enumerate(pixlist) if pixlist_hemi[i]+pixlist_view[i]=='leftanterior']
            idx_right_ant=[i for i,x in enumerate(pixlist) if pixlist_hemi[i]+pixlist_view[i]=='rightanterior']
            idx_left_post=[i for i,x in enumerate(pixlist) if pixlist_hemi[i]+pixlist_view[i]=='leftposterior']
            idx_right_post=[i for i,x in enumerate(pixlist) if pixlist_hemi[i]+pixlist_view[i]=='rightposterior']
            if all([len(x)==1 for x in [idx_left_ant,idx_right_ant,idx_left_post,idx_right_post]]):
                [pix_left_ant,pix_right_ant,pix_left_post,pix_right_post]=pad_to_max_height([
                    pixlist[idx_left_ant[0]],
                    pixlist[idx_right_ant[0]],
                    pixlist[idx_left_post[0]],
                    pixlist[idx_right_post[0]]
                    ])
                pix_ant=np.hstack((pix_right_ant,pix_left_ant))
                pix_post=np.hstack((pix_left_post,pix_right_post))
                #make the "left" = both L+R anterior, and the "right" = both L+R posterior
                pixlist[idx_left_ant[0]]=pix_ant
                pixlist[idx_right_ant[0]]=pix_post
                #set hemi to blank for the remaining entries so they are left out of pixlist_left/right below
                pixlist_hemi[idx_left_post[0]]=[]
                pixlist_hemi[idx_right_post[0]]=[]
        
        elif ( (all([x=='dorsal' for x in pixlist_view]) or all([x=='ventral' for x in pixlist_view])) 
            and "left" in pixlist_hemi and "right" in pixlist_hemi):
            #special case when it is ONLY dorsal or ONLY ventral
            dors_vent=pixlist_view[0]
            idx_left_dv=[i for i,x in enumerate(pixlist) if pixlist_hemi[i]+pixlist_view[i]=='left'+dors_vent]
            idx_right_dv=[i for i,x in enumerate(pixlist) if pixlist_hemi[i]+pixlist_view[i]=='right'+dors_vent]
            if all([len(x)==1 for x in [idx_left_dv,idx_right_dv]]):
                if dors_vent == 'dorsal':
                    pix_left_dv=np.flip(np.transpose(pixlist[idx_left_dv[0]],[1,0,2]),1)
                    pix_right_dv=np.flip(np.transpose(pixlist[idx_right_dv[0]],[1,0,2]),0)
                    [pix_left_dv,pix_right_dv]=pad_to_max_height([pix_left_dv,pix_right_dv])
                    pixlist[idx_left_dv[0]]=pix_left_dv
                    pixlist[idx_right_dv[0]]=pix_right_dv
                elif dors_vent == 'ventral':
                    pix_left_dv=np.flip(np.transpose(pixlist[idx_left_dv[0]],[1,0,2]),0)
                    pix_right_dv=np.flip(np.transpose(pixlist[idx_right_dv[0]],[1,0,2]),1)
                    [pix_left_dv,pix_right_dv]=pad_to_max_height([pix_left_dv,pix_right_dv])
                    #swap hemis so they display correctly from underneath
                    pixlist[idx_left_dv[0]]=pix_right_dv
                    pixlist[idx_right_dv[0]]=pix_left_dv

        #wmax=max([x.shape[1] for x in pixlist])
        #pixlist=[padimage(x,bgcolor=None,padfinalsize=[-1,wmax]) for x in pixlist]
        pixlist=pad_to_max_width(pixlist)
        
        pixlist_left=[x for i,x in enumerate(pixlist) if pixlist_hemi[i]=='left']
        pixlist_right=[x for i,x in enumerate(pixlist) if pixlist_hemi[i]=='right']

        whichview=[x for i,x in enumerate(pixlist_view) if pixlist_hemi[i]=='left']
        
        #pad matching L/R pairs to the same height
        
        for i in range(len(pixlist_left)):
            pixlist_left[i],pixlist_right[i]=pad_to_max_height([pixlist_left[i],pixlist_right[i]])
        
        pixlist_stack=np.hstack((np.vstack(pixlist_left),np.vstack(pixlist_right)))
    else:
        pixlist_stack=[]
    

    #######################
    #volume slice (if any)

    bgvolfile="%s/MNI152_T1_1mm_headmasked.nii.gz" % (get_data_dir('atlas'))
    bgmaskfile="%s/MNI152_T1_1mm_headmask.nii.gz" % (get_data_dir('atlas'))
    volvals=None
    volvals_roiindex=None
    bgvolvals=None
    slicevol_cmap=None
    bgvol_cmap=None
    blank_cmap=ListedColormap(backgroundcolor)
    slice_background_alpha=np.clip(slice_background_alpha,0,1)
    
    if slice_dict:
        refnib=nib.load(bgvolfile)
        masknib=nib.load(bgmaskfile)
        bgvolvals=refnib.get_fdata()
        bgmaskvals=np.clip(masknib.get_fdata(),0,1)
        volvals=fill_volume_rois(roivals,atlasinfo,backgroundval=np.nan,referencevolume=refnib)
        volvals_roiindex=fill_volume_rois(np.arange(len(roivals)),atlasinfo,backgroundval=np.nan,referencevolume=refnib)
        
        #quick (and dirty) test for ax flipping (ex: MNI reference volume is [-1,1,1])
        ax_to_flip=np.where(np.diag(refnib.affine)[:3]<0)[0]
        bgvolvals=np.flip(bgvolvals,ax_to_flip)
        bgmaskvals=np.flip(bgmaskvals,ax_to_flip)
        volvals=np.flip(volvals,ax_to_flip)
        volvals_roiindex=np.flip(volvals_roiindex,ax_to_flip)
        
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
        
        #########
        #render ROI borders
        roi_edge_mask=None
        if border_roimask is not None:
            for r,rmask in enumerate(border_roimask):
                if rmask==0 or not np.any(volvals_roiindex==r):
                    #ROI is not contained in this volume
                    continue
                
                pix_roi=slice_volume_to_rgb(volvals_roiindex==r,bgvolvals=bgvolvals,bgmaskvals=bgmaskvals,sliceaxis=sliceax[a],slice_indices=slice_dict[a],mosaic=mosaic_dict[a],
                                                   cmap=border_colormap,clim=border_clim,bg_cmap=bgvol_cmap,blank_cmap=blank_cmap, background_alpha=0, slice_zoom=slice_zoom)
                
                if np.min(pix_roi[:,:,0])==np.max(pix_roi[:,:,0]):
                    #ROI is not shown in this view
                    continue
                
                edgepad=1
                if edgepad > 0:
                    pix_roi=np.pad(pix_roi,([edgepad,edgepad],[edgepad,edgepad],[0,0]),constant_values=1) #pad array to avoid marking edges as border
                pix_edge=Image.fromarray(np.round(pix_roi*255).astype(np.uint8)).convert("L").filter(border_edgefilter)
                pix_edge=np.asarray(pix_edge)>1e-6
                if edgepad > 0:
                    pix_edge=pix_edge[edgepad:-edgepad,edgepad:-edgepad] #remove padding
                if roi_edge_mask is None:
                    roi_edge_mask=pix_edge
                else:
                    roi_edge_mask=np.maximum(roi_edge_mask,pix_edge)
        ##########
        imgslice_dict[a]=slice_volume_to_rgb(volvals,bgvolvals,bgmaskvals,sliceaxis=sliceax[a],slice_indices=slice_dict[a],mosaic=mosaic_dict[a], slice_zoom=slice_zoom,
                                                   cmap=slicevol_cmap,clim=clim,bg_cmap=bgvol_cmap,blank_cmap=blank_cmap, background_alpha=slice_background_alpha,
                                                   borderimage=roi_edge_mask,bordercolor=border_color,borderwidth=border_width/slice_zoom)

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
            current_image_size=[int(round(x*slice_only_scale_factor)) for x in current_image_size]

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
    
    if fig is not None:
        plt.close(fig)

    if add_colorbar:
        pixlist_stack=add_colorbar_to_image(pixlist_stack,colorbar_color=colorbar_color,colorbar_fontsize=colorbar_fontsize,colorbar_location=colorbar_location,
                                            padding=25,figdpi=figdpi,colormap=colormap,clim=clim,backgroundcolor=backgroundcolor, colorbar_label=colorbar_label,
                                            colorbar_label_rotation=colorbar_label_rotation)
    
    if outputimagefile is not None:
        save_image(pixlist_stack,outputimagefile)
        print("Saved %s" % (outputimagefile))
    

    return pixlist_stack

def fetch_surface_dataset(mesh='fsaverage5',data_dir=None):
    if data_dir is None:
        data_dir=get_data_dir('nilearn')
    
    mesh_info= datasets.fetch_surf_fsaverage(mesh=mesh,data_dir=data_dir)
    
    extra_mesh_info={'semi_left':'semi_left.gii.gz','semi_right':'semi_right.gii.gz',
                     'mid_left':'mid_left.gii.gz','mid_right':'mid_right.gii.gz',
                     'flat_left':'flat_left.gii.gz','flat_right':'flat_right.gii.gz'}
    
    extra_mesh_info={k:os.path.join(get_data_dir('custom_surfaces'),mesh,f) for k,f in extra_mesh_info.items()}
    for k,f in extra_mesh_info.items():
        if os.path.exists(f):
            mesh_info[k]=f
    return mesh_info
    
def get_data_dir(data_type):
    data_paths={'atlas': os.path.join(getscriptdir(),"atlases"),
                'lookup': os.path.join(getscriptdir(),"lookups"),
                'facemap': os.path.join(getscriptdir(),"atlas_cache"),
                'nilearn': os.path.join(getscriptdir(),"nilearn_data"),
                'custom_surfaces': os.path.join(getscriptdir(),"custom_surface_data")}
    
    if data_type in data_paths:
        return data_paths[data_type]
    else:
        raise Exception("Unknwn data path type: %s. Choose from %s" % (data_type, ",".join(data_paths.keys())))
    
def clear_cache(which_cache='facemap'):
    if isinstance(which_cache,str):
        which_cache=[which_cache]

    cache_dirs=[]
    if 'facemap' in which_cache:
        cache_dirs+=[get_data_dir('facemap')]
    if 'lookup' in which_cache:
        cache_dirs+=[get_data_dir('lookup')]
    if 'nilearn' in which_cache:
        cache_dirs+=[get_data_dir('nilearn')]

    if 'all' in which_cache:
        cache_dirs+=[get_data_dir(s) for s in ['facemap','lookup','nilearn']]

    cache_dirs=list(set(cache_dirs))

    for d in cache_dirs:
        try:
            if os.path.exists(d):
                shutil.rmtree(d)
                print("Removed cache dir: %s" % (d))
        except Exception as e:
            print(e)
            print("Could not clear cache folder %s" % (d))

def print_cache(which_cache='facemap'):
    if isinstance(which_cache,str):
        which_cache=[which_cache]

    cache_dirs=[]
    if 'facemap' in which_cache:
        cache_dirs+=[get_data_dir('facemap')]
    if 'lookup' in which_cache:
        cache_dirs+=[get_data_dir('lookup')]
    if 'nilearn' in which_cache:
        cache_dirs+=[get_data_dir('nilearn')]

    if 'all' in which_cache:
        which_cache=['facemap','lookup','nilearn']

    which_cache=list(set(which_cache))

    for c in which_cache:
        d=get_data_dir(c)
        try:
            if os.path.exists(d):
                print("")
                print("%s: %s" % (c,d))
                print("\n".join(os.listdir(d)))
        except Exception as e:
            print(e)
            print("Could not display cache folder %s" % (d))

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
    no_lookup=args.no_lookup
    create_lookup=args.create_lookup
    facemode=args.facemode
    bestmodeiters=args.bestmodeiter

    slicearg=args.slices
    stackdirection=args.stackdirection
    slicedict_order=None
    slice_background_alpha=args.slice_background_alpha
    slice_zoom=args.slice_zoom
    
    add_colorbar=args.colorbar
    colorbar_color=args.colorbar_color
    colorbar_fontsize=args.colorbar_fontsize
    colorbar_location=args.colorbar_location
    colorbar_label=args.colorbar_label
    colorbar_label_rotation=args.colorbar_label_rotation

    if args.show_cache:
        print_cache('all')
        exit(0)
    
    if args.clear_facemaps:
        clear_cache('facemap')
    elif args.clear_lookups:
        clear_cache('lookup')
    elif args.clear_all_cache:
        clear_cache('all')
        

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

    if inputfile is None:
        inputfile=""
    
    roivals=None
    if len(inputvals_arg)>0:
        roivals=np.array(inputvals_arg).astype(float)
    elif inputfile.lower().endswith(".txt"):
        roivals=np.loadtxt(inputfile)
    elif inputfile.lower().endswith(".csv"):
        roivals=np.loadtxt(inputfile,delimiter=",")
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
    
    if atlasname is not None:
        atlasname=args.atlasname.lower()
        atlas_info=retrieve_atlas_info(atlasname)
    else:
        atlas_info={'atlasname':None,'roilutfile':roilutfile,'lhannotfile':lhannotfile,'rhannotfile':rhannotfile,
            'annotsurfacename':annotsurfacename,'lhannotprefix':lhannotprefix,'rhannotprefix':rhannotprefix,'subcorticalvolume':subcortvolfile}
    
    surftype_allowed=['white','inflated','pial','semi','mid','flat']
    try:
        surftype=stringfromlist(surftype,surftype_allowed)
        if surftype == 'inflated':
            surftype='infl'
    except:
        print("surftype must be one of: %s" % (",".join(surftype_allowed)))
        exit(1)
    
    facemode_allowed=["mode","mean","best"]
    try:
        facemode=stringfromlist(facemode,facemode_allowed)
    except:
        print("facemode must be one of: %s" % (",".join(facemode_allowed)))
        exit(1)
    
        
    #nilearn uses 'Spectral' instead of matplotlib 'spectral'
    if isinstance(cmapname,str) and cmapname.lower()=='lut':
        cmap='lut'
        roivals=1 #placeholder
    else:
        try:
            cmap=stringfromlist(cmapname,list(plt.colormaps.keys()),allow_startswith=False)
        except:
            cmap=cmapname
    
    #do cmapfile AFTER checking atlasinfo
    if cmapfile is not None:
        cmapdata=np.loadtxt(cmapfile)
        if cmapdata.shape[1]!=3:
            raise Exception("colormap file must have 3 columns")
        if cmapdata.max()>1:
            cmapdata/=255
        cmap=ListedColormap(cmapdata)
        
    if roivals is None:
        raise Exception("Invalid inputfile: %s" % (inputfile))
    
    img=create_montage_figure(roivals,atlasinfo=atlas_info,
        viewnames=viewnames,surftype=surftype,clim=clim,colormap=cmap,noshading=no_shading,
        outputimagefile=outputimage,upscale_factor=upscale_factor,slicestack_direction=stackdirection,
        slice_dict=slicedict,mosaic_dict=slicemosaic_dict,slicestack_order=slicedict_order,slice_background_alpha=slice_background_alpha,
        slice_zoom=slice_zoom,
        backgroundcolor=bgcolor,no_lookup=no_lookup,create_lookup=create_lookup,
        face_mode=facemode,face_best_mode_iters=bestmodeiters,
        add_colorbar=add_colorbar, colorbar_color=colorbar_color,colorbar_fontsize=colorbar_fontsize,colorbar_location=colorbar_location,
        colorbar_label=colorbar_label,colorbar_label_rotation=colorbar_label_rotation)

if __name__ == "__main__":
    run_montageplot(sys.argv[1:])
