from nilearn import plotting
from nilearn import datasets
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from PIL import Image
import io
import sys
import argparse
import json
import os
from scipy.io import loadmat
from matplotlib import cm
from matplotlib.colors import ListedColormap

def parse_argument_surfplot(argv):
    parser=argparse.ArgumentParser(description='save surface ROI montage')
    
    parser.add_argument('--input',action='store',dest='inputfile')
    parser.add_argument('--inputfield',action='store',dest='inputfieldname')
    parser.add_argument('--views',action='append',dest='viewnames',nargs='*')
    parser.add_argument('--outputimage',action='store',dest='outputimage')
    parser.add_argument('--surftype',action='store',dest='surftype',default='infl')
    parser.add_argument('--cmap','--colormap',action='store',dest='cmapname',default='magma')
    parser.add_argument('--clim', action='append',dest='clim',nargs=2)
    parser.add_argument('--roilut',action='store',dest='roilutfile')
    parser.add_argument('--lhannot',action='store',dest='lhannotfile')
    parser.add_argument('--rhannot',action='store',dest='rhannotfile')
    parser.add_argument('--annotsurfacename',action='store',dest='annotsurface',default='fsaverage5')
    parser.add_argument('--noshading',action='store_true',dest='noshading')
    parser.add_argument('--atlasname',action='store',dest='atlasname')
    
    return parser.parse_args(argv)

def flatlist(l):
    if l is None:
        return []
    return [x for y in l for x in y]

def flatarglist(l):
    if l is None:
        return []
    return flatlist([x.split(",") for x in flatlist(l)])

def cropbg(img,bgcolor=None):
    if bgcolor is None:
        bgcolor=img[0,0,:]
    bgcolor=np.reshape(bgcolor,[1,1,img.shape[2]])
    bgimg=np.repeat(np.repeat(bgcolor,img.shape[0],axis=0),img.shape[1],axis=1)
    bgmask=np.logical_not(np.all(img==bgimg,axis=2))
    idx0=np.argwhere(np.any(bgmask,axis=1)).flatten()
    idx1=np.argwhere(np.any(bgmask,axis=0)).flatten()
    cropcoord=[idx0[0],idx0[-1]+1,idx1[0],idx1[-1]+1]
    newimg=img[cropcoord[0]:cropcoord[1],cropcoord[2]:cropcoord[3],:]
    return newimg, cropcoord

def padimage(img,bgcolor=None,padamount=None,padfinalsize=None):
    if bgcolor is None:
        bgcolor=img[0,0,:]
    bgcolor=np.reshape(bgcolor,[1,1,img.shape[2]])
    
    if padfinalsize is not None:
        finalsize=padfinalsize
        for i,v in enumerate(finalsize):
            if v < 0:
                finalsize[i]=img.shape[i]
        
        xstart=int(np.round((finalsize[0]-img.shape[0])/2))
        ystart=int(np.round((finalsize[1]-img.shape[1])/2))
        
    elif padamount is not None:
        try:
            #test if iterable
            for i in padamount:
                pass
        except:
            padamount=[padamount]
        if(len(padamount)==1):
            padamount=4*padamount
        elif(len(padamount)==2):
            padamount=2*padamount
        finalsize=[img.shape[0]+padamount[1]+padamount[3], img.shape[1]+padamount[0]+padamount[2]]
        xstart=int(padamount[1])
        ystart=int(padamount[0])

    newimg=np.repeat(np.repeat(bgcolor,finalsize[0],axis=0),finalsize[1],axis=1)
    newimg[xstart:xstart+img.shape[0],ystart:ystart+img.shape[1],:]=img
    
    return newimg

def mesh_diffuse(verts=None,faces=None,adjacency=None,vertvals=None,iters=1):

    if adjacency is not None:
        A=adjacency
    elif faces is not None:
        spval=np.ones(faces.shape[0]*6)
        spi=np.concatenate((faces[:,0],faces[:,0],faces[:,1],faces[:,1],faces[:,2],faces[:,2]),axis=0)
        spj=np.concatenate((faces[:,1],faces[:,2],faces[:,0],faces[:,2],faces[:,0],faces[:,1]),axis=0)
        
        A=sparse.csr_matrix((spval,(spi,spj)),shape=[verts.shape[0],verts.shape[0]])
        A[A>1]=1 #avoid double links
        A+=sparse.eye(A.shape[0])
        sA=np.array(A.sum(axis=1))[:,0]
        sA[sA==0]=1
        sA=sparse.csr_matrix((1/sA,(np.arange(sA.shape[0]),np.arange(sA.shape[0]))))
        A=sA@A
    else:
        raise Exception("faces or adjacency must be provided")
        
    
    if vertvals is None:
        return A
    
    vertvals_new=vertvals
    for i in range(iters):
        vertvals_new=A@vertvals_new
    return vertvals_new, A

def mesh_shading(verts, faces, lightdirection):
    #compute face norms by cross product of v1->2 and v1->3
    v1=verts[faces[:,0],:]
    v2=verts[faces[:,1],:]
    v3=verts[faces[:,2],:]
    v12=v2-v1
    v13=v3-v1
    v12/=np.sqrt(np.sum(v12**2,axis=1,keepdims=True))
    v13/=np.sqrt(np.sum(v13**2,axis=1,keepdims=True))
    facenorm=np.cross(v12,v13)
    
    #now dot the light direction with the norm to get incidence angle
    lightdir=np.array(lightdirection)
    lightdir=lightdir/np.sqrt(np.sum(lightdir**2))
    facenormdot=np.dot(facenorm,lightdir)
    facenormdot=np.clip(facenormdot,0,1)
    
    #now map face norm to vertex
    #make a sparse [verts x faces] matrix so that each vertex ends up with a mean of all the faces that contain it 
    spi=np.concatenate((faces[:,0],faces[:,1],faces[:,2]),axis=0)
    spj=np.concatenate((np.arange(faces.shape[0]),np.arange(faces.shape[0]),np.arange(faces.shape[0])),axis=0)
    
    T_face2vert=sparse.csr_matrix((np.ones(spi.shape),(spi,spj)))
    T_face2vert[T_face2vert>1]=1 #avoid double links
    sT=np.array(T_face2vert.sum(axis=1))[:,0]
    sT[sT==0]=1
    sT=sparse.csr_matrix((1/sT,(np.arange(sT.shape[0]),np.arange(sT.shape[0]))))
    T_face2vert=sT@T_face2vert
    
    vertshading=T_face2vert@facenormdot
    
    return vertshading

def fig2pixels(fig,imgbuffer=None):
    if imgbuffer is None:
        imgbuffer=io.BytesIO()
    imgbuffer.seek(0)
    fig.savefig(imgbuffer,format='png',bbox_inches=0)
    imgbuffer.seek(0)
    img=Image.open(imgbuffer,'r')
    return np.asarray(img)

def retrieve_atlas_info(atlasname, atlasinfo_jsonfile=None, scriptdir=None):
    if scriptdir is None:
        scriptdir=os.path.realpath(os.path.dirname(__file__))
    if atlasinfo_jsonfile is None:
        atlasinfo_jsonfile="%s/atlas_info.json" % (scriptdir)
    
    lhannotprefix=None
    rhannotprefix=None

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
        annotsurfacename=atlas_info_list[atlasname]['annotsurface']
    else:
        raise Exception("atlas name '%s' not found. Choose from %s" % (atlasname, ",".join(atlas_info_list.keys())))
    
    atlasinfo={'atlasname':atlasname,'roilutfile':roilutfile,'lhannotfile':lhannotfile,'rhannotfile':rhannotfile,
        'annotsurfacename':annotsurfacename,'lhannotprefix':lhannotprefix,'rhannotprefix':rhannotprefix}
    
    return atlasinfo

def save_montage_figure(roivals,atlasinfo=None,
    roilutfile=None,lhannotfile=None,rhannotfile=None,annotsurfacename='fsaverage5',lhannotprefix=None, rhannotprefix=None,
    viewnames=None,surftype='infl',clim=None,colormap=None, noshading=False,
    outputimagefile=None):
    
    if clim is None or len(clim)==0:
        clim=[np.nanmin(roivals), np.nanmax(roivals)]
    
    if viewnames is not None and isinstance(viewnames,str):
        #make sure viewnames is iterable
        viewnames=[viewnames]
            
    if viewnames is None or len(viewnames)==0:
        viewnames=['dorsal','lateral','medial','ventral']
    elif "all" in [v.lower() for v in viewnames]:
        viewnames=['dorsal','lateral','medial','ventral']
    
    if atlasinfo is not None:
        roilutfile=atlasinfo['roilutfile']
        lhannotfile=atlasinfo['lhannotfile']
        rhannotfile=atlasinfo['rhannotfile']
        annotsurfacename=atlasinfo['annotsurfacename']
        lhannotprefix=atlasinfo['lhannotprefix']
        rhannotprefix=atlasinfo['rhannotprefix']
    
    #just to make things easier now that we are inside the function
    shading=not noshading
    
    fsaverage = datasets.fetch_surf_fsaverage(mesh=annotsurfacename)
    
    lhlabels,ctab,lhnames=nib.freesurfer.io.read_annot(lhannotfile)
    lhnames=[(x.decode('UTF-8')) for x in lhnames]
    if lhannotprefix is not None:
        lhnames=['%s%s' % (lhannotprefix,x) for x in lhnames]

    rhlabels,ctab,rhnames=nib.freesurfer.io.read_annot(rhannotfile)
    rhnames=[(x.decode('UTF-8')) for x in rhnames]
    if rhannotprefix is not None:
        rhnames=['%s%s' % (rhannotprefix,x) for x in rhnames]

    Troi=pd.read_table(roilutfile,delimiter='\s+',header=None,names=['label','name','R','G','B'])
    Troi=Troi[Troi['name']!='Unknown']

    lhannotval=np.zeros(Troi.shape[0])
    rhannotval=np.zeros(Troi.shape[0])
    for i,n86 in enumerate(Troi['name']):
        lhidx=[j for j,nannot in enumerate(lhnames) if nannot==n86]
        rhidx=[j for j,nannot in enumerate(rhnames) if nannot==n86]
        if len(lhidx)==1:
            lhannotval[i]=lhidx[0]
        elif len(rhidx)==1:
            rhannotval[i]=rhidx[0]
    
    Troi['lhannot']=lhannotval
    Troi['rhannot']=rhannotval
    
    view_azel_lh={'dorsal':[180,90],'lateral':[180,0], 'medial':[0,0], 'ventral':[0,-90]}
    view_azel_rh={'dorsal':[0,90],'lateral':[0,0], 'medial':[180,0], 'ventral':[180,-90]}
    view_lightdir_lh={'dorsal':[0,0,1],'lateral':[-1,0,0],'medial':[1,0,0],'ventral':[0,0,-1]}
    view_lightdir_rh={'dorsal':[0,0,1],'lateral':[1,0,0],'medial':[-1,0,0],'ventral':[0,0,-1]}
    
    pixlist=[]
    pixlist_hemi=[]
    pixlist_view=[]
    
    surfLR={}
    diffuserLR_sparse={}
    if shading:
        surfLR['left']=nib.load(fsaverage[surftype+'_'+'left']).agg_data()
        surfLR['right']=nib.load(fsaverage[surftype+'_'+'right']).agg_data()
        diffuserLR_sparse['left']=mesh_diffuse(verts=surfLR['left'][0], faces=surfLR['left'][1])
        diffuserLR_sparse['right']=mesh_diffuse(verts=surfLR['right'][0], faces=surfLR['right'][1])
    
    #need a colormap where black->white starts in the middle
    #since plot_stat_map requires symmetric range
    symmgraycolors = cm.get_cmap("gray",256)(abs(np.linspace(-1, 1, 256)))
    cmap_symmgray=ListedColormap(symmgraycolors)
    
    #rescale values (and clim) to [0,1000] so that plot_roi doesn't have issues with near-zero values
    zeroval=1e-8
    roivals=roivals.flatten()[:Troi.shape[0]]
    roivals[roivals==0]=zeroval
    #roivals_rescaled=roivals
    #clim_rescaled=clim
    
    roivals_rescaled=np.clip(1000*((roivals-clim[0])/(clim[1]-clim[0])),zeroval,1000)
    clim_rescaled=[0,1000]
    
    surfvalsLR={}
    surfvalsLR['left']=np.zeros(lhlabels.shape)
    surfvalsLR['right']=np.zeros(rhlabels.shape)
    for i in range(Troi.shape[0]):
        v=roivals_rescaled[i]
        if Troi['lhannot'].iloc[i]>0:
            surfvalsLR['left'][lhlabels==Troi['lhannot'].iloc[i]]=v
        elif Troi['rhannot'].iloc[i]>0:
            surfvalsLR['right'][rhlabels==Troi['rhannot'].iloc[i]]=v
    
    for ih,h in enumerate(['left','right']):
        for iv, viewname in enumerate(viewnames):
            azel=None
            if h=='left':
                azel=view_azel_lh[viewname]
                lightdir=view_lightdir_lh[viewname]
            else:
                azel=view_azel_rh[viewname]
                lightdir=view_lightdir_rh[viewname]
            
            ##############################################
            # generate shading image for each view
            
            if shading:
                shading_smooth_iters=1
                if surftype == 'infl':
                    #for inflated surfaces, some weird patterns in shading, but disappear with smoothing
                    shading_smooth_iters=10
                shadingvals=mesh_shading(surfLR[h][0], surfLR[h][1], lightdir)
                shadingvals,_=mesh_diffuse(vertvals=shadingvals,adjacency=diffuserLR_sparse[h],iters=shading_smooth_iters)
                shadingvals=shadingvals/shadingvals.max()
                
                v=plotting.plot_surf_stat_map(fsaverage[surftype+'_'+h], stat_map=shadingvals,
                                       hemi=h, view=viewname,
                                       bg_map=fsaverage['sulc_'+h], bg_on_data=False,
                                       darkness=.5,cmap=cmap_symmgray, colorbar=False, vmax=1)
                if azel is not None:
                    v.get_axes()[0].view_init(azim=azel[0],elev=azel[1])
                
                pixshading=fig2pixels(v)
                
                pixshading=pixshading[:,:,:3].astype(np.float32)/255.0
                pixshading=np.mean(pixshading[:,:,:3],axis=2,keepdims=True)
                pixshading=pixshading/pixshading.max()
            
            ##############################################
            #generate main ROI surface image (unshaded)
            
            #plot_surf_stat_map(stat_map=, vmax=)
            #plot_surf_roi(roi_map=, vmin=, vmax=)
            v=plotting.plot_surf_roi(fsaverage[surftype+'_'+h], roi_map=surfvalsLR[h],
                                   hemi=h, view=viewname,
                                   bg_map=fsaverage['sulc_'+h], bg_on_data=False,
                                   darkness=.5,cmap=colormap, colorbar=False, vmin=clim_rescaled[0], vmax=clim_rescaled[1])
            if azel is not None:
                v.get_axes()[0].view_init(azim=azel[0],elev=azel[1])
            
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
            v.get_axes()[0].set_facecolor(rgbnew)
            
            pixbg=fig2pixels(v)
            
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
            
            pix=padimage(pix,bgcolor=None,padamount=1)
            
            pixlist+=[pix]
            pixlist_hemi+=[h]
    
    #pad all to the same width
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
    
    if outputimagefile is not None:
        newimg=Image.fromarray(pixlist_stack)
        newimg.save(outputimagefile)
        print("Saved %s" % (outputimagefile))
    
    return pixlist_stack

def run_surfplot(argv):
    args=parse_argument_surfplot(argv)
    
    inputfile=args.inputfile
    inputfieldname=args.inputfieldname
    viewnames=flatarglist(args.viewnames)
    outputimage=args.outputimage
    surftype=args.surftype
    clim=flatarglist(args.clim)
    cmapname=args.cmapname
    roilutfile=args.roilutfile
    lhannotfile=args.lhannotfile
    rhannotfile=args.rhannotfile
    no_shading=args.noshading
    annotsurfacename=args.annotsurface
    
    if len(clim)==2:
        clim=[np.float32(x) for x in clim]
    else:
        clim=None
    
    #nilearn uses 'Spectral' instead of matplotlib 'spectral'
    if cmapname.lower()=='spectral':
        cmapname='Spectral'
    elif cmapname.lower()=='spectral_r':
        cmapname='Spectral_r'
    
    #roivals=np.arange(86)+1
    
    if inputfile is None:
        inputfile=""
    
    if inputfile.lower().endswith(".txt"):
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
            raise Exception("Multiple data fields found in %s. Specify one with --inputfield")
    else:
        raise Exception("Invalid inputfile: %s" % (inputfile))
    
    atlasname=args.atlasname.lower()
    atlas_info=retrieve_atlas_info(atlasname)
    
    img=save_montage_figure(roivals,atlasinfo=atlas_info,
        viewnames=viewnames,surftype=surftype,clim=clim,colormap=cmapname,noshading=no_shading,
        outputimagefile=outputimage)
    
if __name__ == "__main__":
    run_surfplot(sys.argv[1:])
