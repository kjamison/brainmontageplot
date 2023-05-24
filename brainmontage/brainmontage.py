from nilearn import plotting
from nilearn import datasets
import numpy as np
import nibabel as nib
import nibabel.processing as nibproc
import pandas as pd
from tqdm import tqdm

import json
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from datetime import datetime

from matplotlib import use as matplotlib_set_backend
from matplotlib import get_backend as matplotlib_get_backend

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
    misc_arg_group.add_argument('--nolookup',action='store_true',dest='no_lookup',help='Do not use saved lookups (mainly for testing)')
    misc_arg_group.add_argument('--createlookup',action='store_true',dest='create_lookup',help='Create lookup if not found')

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
        Troi=pd.read_table(roilutfile,delimiter='\s+',header=None,names=['label','name','R','G','B'])
        Troi=Troi[Troi['name']!='Unknown']
    
        if lhannotfile.endswith(".annot"):
            lhlabels,ctab,lhnames=nib.freesurfer.io.read_annot(lhannotfile)
            lhnames=[(x.decode('UTF-8')) for x in lhnames]

            rhlabels,ctab,rhnames=nib.freesurfer.io.read_annot(rhannotfile)
            rhnames=[(x.decode('UTF-8')) for x in rhnames]
        elif lhannotfile.endswith(".label.gii"):
            lhgii=nib.load(lhannotfile)
            lhlabels=lhgii.agg_data()
            lhnames=[v for k,v in lhgii.labeltable.get_labels_as_dict().items()]

            rhgii=nib.load(rhannotfile)
            rhlabels=rhgii.agg_data()
            rhnames=[v for k,v in rhgii.labeltable.get_labels_as_dict().items()]
            
        if lhannotprefix is not None:
            lhnames=['%s%s' % (lhannotprefix,x) for x in lhnames]        
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
        
    elif lhannotfile.endswith(".shape.gii"):
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
    if not 'subcorticalvolume' in atlasinfo or \
        atlasinfo['subcorticalvolume'] is None or \
        atlasinfo['subcorticalvolume'] == "":
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
    
    roicount=None
    lhannotprefix=""
    rhannotprefix=""
    subcortfile=""
    lookupsurface=""
    lhlookupannotfile=""
    rhlookupannotfile=""
    
    with open(atlasinfo_jsonfile,'r') as f:
        atlas_info_list=json.load(f)
    
    if atlasname == 'list':
        return list(atlas_info_list.keys())
    
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
        
        #use the original annot info unless lookup-specific info is provided
        lookupsurface=annotsurfacename
        lhlookupannotfile=lhannotfile
        rhlookupannotfile=rhannotfile

        if "lookupsurface" in atlas_info_list[atlasname]:
            lookupsurface=atlas_info_list[atlasname]['lookupsurface']
        if "lhlookupannot" in atlas_info_list[atlasname]:
            lhlookupannotfile=atlas_info_list[atlasname]['lhlookupannot'].replace('%SCRIPTDIR%',scriptdir)
        if "rhlookupannot" in atlas_info_list[atlasname]:
            rhlookupannotfile=atlas_info_list[atlasname]['rhlookupannot'].replace('%SCRIPTDIR%',scriptdir)
    else:
        raise Exception("atlas name '%s' not found. Choose from %s" % (atlasname, ",".join(atlas_info_list.keys())))
    
    atlasinfo={'atlasname':atlasname,'roicount':roicount,'roilutfile':roilutfile,'lhannotfile':lhannotfile,'rhannotfile':rhannotfile,
        'annotsurfacename':annotsurfacename,'lhannotprefix':lhannotprefix,'rhannotprefix':rhannotprefix,
        'subcorticalvolume':subcortfile,'lhlookupannotfile':lhlookupannotfile,'rhlookupannotfile':rhlookupannotfile,'lookupsurface':lookupsurface}
    
    return atlasinfo

def generate_surface_view_lookup(atlasinfo,surf,hemi=None,azel=None,figsize=None,figdpi=200,lightdir=None,shading_smooth_iters=1,surfbgvals=None):
    
    if figsize is None:
        figsize=(6.4,6.4)

    roicount=atlasinfo['roicount']
    roivals=np.zeros(roicount)

    ##############
    # for each ROI, map to vertices, and then to faces, and then collapse into a FACES x 1 roi index vector
    # which we then render and capture the result for the lookup

    faces=surf[1]
    #now map vertex values to faces (face=average)
    #make a sparse [faces x verts] matrix so that each face ends up with a mean of all of its verts
    spi=np.concatenate((faces[:,0],faces[:,1],faces[:,2]),axis=0)
    spj=np.concatenate((np.arange(faces.shape[0]),np.arange(faces.shape[0]),np.arange(faces.shape[0])),axis=0)

    T_vert2face=sparse.csr_matrix((np.ones(spj.shape),(spj,spi)))
    T_vert2face[T_vert2face>1]=1 #avoid double links
    sT=np.array(T_vert2face.sum(axis=1))[:,0]
    sT[sT==0]=1
    sT=sparse.csr_matrix((1/sT,(np.arange(sT.shape[0]),np.arange(sT.shape[0]))))
    T_vert2face=sT@T_vert2face
    
    faceval_list=[]
    for i in range(roicount):
        roivals[:]=0
        roivals[i]=1
        surfvalsLR=fill_surface_rois(roivals,atlasinfo)
        surfvals=surfvalsLR[hemi]

        if not any(surfvals>0):
            f=None
            faceval_list+=[f]
            continue

        surfvals=mesh_diffuse(verts=surf[0],faces=surf[1],vertvals=surfvals,iters=1)[0]

        f=T_vert2face @ surfvals

        faceval_list+=[f]

    faceval_shape=[f.shape for f in faceval_list if f is not None][0]
    for i,f in enumerate(faceval_list):
        if f is None:
            faceval_list[i]=np.zeros(faceval_shape)

    #add an extra blank for the NON-ROI brain
    faceval_list=[np.zeros(faceval_shape)]+faceval_list

    faceval_list=np.stack(faceval_list)
    faceval_maxidx=np.argmax(faceval_list,axis=0)

    #turn ROI index into an unique R,G,B triplet for rendering
    roi_idx=np.arange(roicount+1) #use 1-based ROI here so we know 0=no ROI
    rgbval_idx=np.stack((roi_idx % 256, roi_idx // 256 % 256, roi_idx // (256*256)),axis=-1)
    cmap_faces=ListedColormap(rgbval_idx/255.0)

    f_rgb=cmap_faces(faceval_maxidx/roi_idx.max())
    ##############

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
    v.get_axes()[-1].set_box_aspect([1,1,.75])

    hmesh=v.findobj(lambda obj: isinstance(obj, Poly3DCollection))[0]

    v.get_axes()[-1].set_facecolor(backgroundcolor)
    if azel is not None:
        v.get_axes()[-1].view_init(azim=azel[0],elev=azel[1])

    imgbgvals=None
    if surfbgvals is not None:
        imgbgvals=fig2pixels(figure,dpi=figdpi)
        imgbgvals=imgbgvals[:,:,:3].astype(np.float32)/255.0


    hmesh.set_facecolor(f_rgb)
    img=fig2pixels(figure,dpi=figdpi)

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


def load_atlas_lookup(atlasname,surftype,shading=True):
    atlasinfo=retrieve_atlas_info(atlasname=atlasname)

    lookup_surface_name=atlasinfo['lookupsurface']

    lookup_dir="%s/lookups" % (getscriptdir())

    shading_lookup_file="%s/shadinglookup_%s_%s.mat" % (lookup_dir,surftype,lookup_surface_name)
    lookup_file="%s/roilookup_%s_%s_%s.mat" % (lookup_dir,atlasname,surftype,lookup_surface_name)

    if not os.path.exists(lookup_file):
        #raise Exception("Lookup not found: %s" % (lookup_file))
        print("Lookup not found: %s" % (lookup_file))
        return None
    
    lookup=loadmat(lookup_file,simplify_cells=True)
    lookup['info']['atlasname']=atlasname

    lookup_has_shading=all(['shading' in lookup[k] for k in lookup['views']])
    if shading and not lookup_has_shading:
        if not os.path.exists(shading_lookup_file):
            #raise Exception("Shading lookup not found: %s" % (shading_lookup_file))
            print("Shading lookup not found: %s" % (shading_lookup_file))
            lookup=None
        else:
            shading_lookup=loadmat(shading_lookup_file,simplify_cells=True)
            for v in lookup['views']:
                if not lookup[v]['roimap'].shape[:2]==shading_lookup[v]['shading'].shape[:2]:
                    print("Shading data in %s does not match ROI map image for %s:" % (shading_lookup_file,lookup_file))
                    #use basic (slow) rendering instead
                    lookup=None
                    break
                #copy shading and brainbackground (eg: sulc) images into lookup
                lookup[v]['shading']=shading_lookup[v]['shading'].copy()
                lookup[v]['brainbackground']=shading_lookup[v]['brainbackground'].copy()
    
    return lookup

def save_atlas_lookup_file(atlasname=None, atlasinfo=None, surftype='infl',viewnames='all',figsize=(6.4,6.4),figdpi=200, shading=True, only_shading=False, overwrite_existing=True):

    if isinstance(viewnames,str):
        viewnames=[viewnames]

    if 'all' in viewnames:
        viewnames=['dorsal','lateral','medial','ventral','anterior','posterior']

    if atlasname is not None and atlasinfo is None:
        atlasinfo=retrieve_atlas_info(atlasname=atlasname)
    
    atlasname=atlasinfo['atlasname']

    roicount=atlasinfo['roicount']

    lookup_surface_name=atlasinfo['lookupsurface']
    atlasinfo['lhannotfile']=atlasinfo['lhlookupannotfile']
    atlasinfo['rhannotfile']=atlasinfo['rhlookupannotfile']
    atlasinfo['annotsurfacename']=lookup_surface_name

    lookup_dir="%s/lookups" % (getscriptdir())
    if only_shading:
        lookup_file="%s/shadinglookup_%s_%s.mat" % (lookup_dir,surftype,lookup_surface_name)
    else:
        lookup_file="%s/roilookup_%s_%s_%s.mat" % (lookup_dir,atlasname,surftype,lookup_surface_name)

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

    fsaverage = datasets.fetch_surf_fsaverage(mesh=lookup_surface_name,data_dir=os.path.join(getscriptdir(),"nilearn_data"))
    surfbgvalsLR={'left':fsaverage['sulc_left'],'right':fsaverage['sulc_right']}

    surfLR={}
    surfLR['left']=nib.load(fsaverage[surftype+'_'+'left']).agg_data()
    surfLR['right']=nib.load(fsaverage[surftype+'_'+'right']).agg_data()

    
    lookup={}
    lookup['info']={'atlasname':atlasname,'atlasinfo':atlasinfo,'dpi':figdpi,'figsize':figsize,
                    'surftype':surftype,'timestamp':datetime.now().strftime("%Y%m%d_%H%M%S")}

    if only_shading:
        print("Generating shading-only lookup for %s %s" % (atlasname,surftype))
    else:
        print("Generating lookup for %s %s" % (atlasname,surftype))

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

        lookup[k] = generate_surface_view_lookup(atlasinfo,surfLR[h],hemi=h,surfbgvals=surfbgvals,azel=azel,lightdir=lightdir,
                                    shading_smooth_iters=shading_smooth_iters,figsize=figsize,figdpi=figdpi)
        
    lookup['views']=np.array(viewkeys,dtype=object)

    if shading and only_shading:
        #deep copy shading info to new struct before deleting from roi lookup
        lookup['info']['annotsurfacename']=lookup['info']['atlasinfo']['annotsurfacename']
        del lookup['info']['atlasinfo']
        del lookup['info']['atlasname']
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

def render_surface_lookup(roivals,lookup,cmap='magma',clim=None,backgroundcolor=None,shading=True,braincolor=None):
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
            brainbg_rgb=np.concatenate((brainbg_rgb,roiimg_rgb[:,:,3][:,:,np.newaxis]),axis=2)
    else:
        if braincolor is None:
            braincolor='gray'
        brainbg_cmap=ListedColormap(braincolor)
        brainbg_rgb=brainbg_cmap(np.ones(roiimg.shape))
    
    
    imgnonzero=np.atleast_3d(lookup['roinonzero'])
    
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
    roilutfile=None,lhannotfile=None,rhannotfile=None,annotsurfacename='fsaverage5',lhannotprefix=None, rhannotprefix=None, subcorticalvolume=None,
    viewnames=None,surftype='infl',clim=None,colormap=None, noshading=False, upscale_factor=1, backgroundcolor="white",
    slice_dict={}, mosaic_dict={},slicestack_order=['axial','coronal','sagittal'],slicestack_direction='horizontal',
    outputimagefile=None, no_lookup=False, create_lookup=False):
    
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
        viewnames=stringfromlist(viewnames,['none','dorsal','lateral','medial','ventral','anterior','posterior'])
    except:
        raise Exception("Viewname must be one of: none, dorsal, lateral, medial, ventral, anterior, posterior")

    if atlasname is not None and atlasinfo is None:
        atlasinfo=retrieve_atlas_info(atlasname)
    
    if atlasinfo is None:
        atlasinfo={}
        atlasname['atlasname']=None
        atlasinfo['roilutfile']=roilutfile
        atlasinfo['lhannotfile']=lhannotfile
        atlasinfo['rhannotfile']=rhannotfile
        atlasinfo['annotsurfacename']=annotsurfacename
        atlasinfo['lhannotprefix']=lhannotprefix
        atlasinfo['rhannotprefix']=rhannotprefix
        atlasinfo['subcorticalvolume']=subcorticalvolume

    atlasname=atlasinfo['atlasname']

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
        lookup_dict=load_atlas_lookup(atlasname=atlasname,surftype=surftype)
        if lookup_dict is None and create_lookup:
            print("Creating lookup for %s %s (this may take up to 10 minutes the first time...)" % (atlasname,surftype))
            lookup_file=save_atlas_lookup_file(atlasname=atlasname,surftype=surftype,viewnames='all',shading=False)
            shading_lookup_file=save_atlas_lookup_file(atlasname=atlasname,surftype=surftype,viewnames='all',shading=True,only_shading=True,overwrite_existing=False)
            lookup_dict=load_atlas_lookup(atlasname=atlasname,surftype=surftype)

    fig=None
    pixlist=[]
    pixlist_hemi=[]
    pixlist_view=[]

    if lookup_dict is not None:

        for ih,h in enumerate(['left','right']):
            for iv,viewname in enumerate(viewnames):
                if viewname == 'none':
                    continue

                viewkey="%s_%s" % (h,viewname)
                pix=render_surface_lookup(roivals_rescaled,lookup_dict[viewkey],cmap=colormap,clim=clim_rescaled,
                                          backgroundcolor=backgroundcolor,shading=shading,braincolor=None)

                pix=padimage(pix,bgcolor=None,padamount=1)

                if upscale_factor != 1:
                    newsize=[np.round(x*upscale_factor).astype(int) for x in pix.shape[:2]]
                    newsize=[newsize[1],newsize[0]]
                    pix=np.asarray(Image.fromarray(pix).resize(newsize,resample=Image.Resampling.LANCZOS))
                    
                pixlist+=[pix]
                pixlist_hemi+=[h]
                pixlist_view+=[viewname]
    else:
        fsaverage = datasets.fetch_surf_fsaverage(mesh=atlasinfo['annotsurfacename'],data_dir=os.path.join(getscriptdir(),"nilearn_data"))
        
        surfLR={}

        shading_smooth_iters=1
        if surftype == 'infl':
            shading_smooth_iters=10
        
        surfLR['left']=nib.load(fsaverage[surftype+'_'+'left']).agg_data()
        surfLR['right']=nib.load(fsaverage[surftype+'_'+'right']).agg_data()

        surfbgvalsLR={'left':fsaverage['sulc_left'],'right':fsaverage['sulc_right']}
        
        surfvalsLR=fill_surface_rois(roivals_rescaled,atlasinfo)

        fig=plt.figure(figsize=(6.4,6.4),facecolor=backgroundcolor)
        figsize=fig.get_size_inches()
        figsize=[x*upscale_factor for x in figsize]
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
                                        figure=fig)

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
    
    if fig is not None:
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
    no_lookup=args.no_lookup
    create_lookup=args.create_lookup

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
        backgroundcolor=bgcolor,no_lookup=no_lookup,create_lookup=create_lookup)

if __name__ == "__main__":
    run_montageplot(sys.argv[1:])
