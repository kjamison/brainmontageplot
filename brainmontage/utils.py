
import numpy as np
from PIL import Image
from scipy import sparse
import io
import os
from matplotlib import pyplot as plt

def flatlist(l):
    if l is None:
        return []
    return [x for y in l for x in y]

def flatarglist(l):
    if l is None:
        return []
    return flatlist([x.split(",") for x in flatlist(l)])

def stringfromlist(teststring,validstrings,allow_startswith=True):
    is_str=False
    
    if isinstance(teststring,str):
        is_str=True
        teststring=[teststring]
    
    for i,s in enumerate(teststring):
        if allow_startswith:
            s_found=[x for x in validstrings if x.lower().startswith(s.lower())]
        else:
            s_found=[x for x in validstrings if x.lower()==s.lower()]
        
        if len(s_found)==0:
            raise Exception("Not found")
        teststring[i]=s_found[0]
    
    if is_str:
        teststring=teststring[0]
    return teststring

def getscriptdir():
    return os.path.realpath(os.path.dirname(__file__))

def get_view_azel(hemi,viewname):
    view_azel={}
    view_azel['left']={'dorsal':[180,90],'lateral':[180,0], 'medial':[0,0], 'ventral':[0,-90],'anterior':[90,0],'posterior':[-90,0]}
    view_azel['right']={'dorsal':[0,90],'lateral':[0,0], 'medial':[180,0], 'ventral':[180,-90],'anterior':[90,0],'posterior':[-90,0]}
    return view_azel[hemi][viewname]

def get_light_dir(hemi,viewname):
    view_lightdir={}
    view_lightdir['left']={'dorsal':[0,0,1],'lateral':[-1,0,0],'medial':[1,0,0],'ventral':[0,0,-1],'anterior':[0,1,0],'posterior':[0,-1,0]}
    view_lightdir['right']={'dorsal':[0,0,1],'lateral':[1,0,0],'medial':[-1,0,0],'ventral':[0,0,-1],'anterior':[0,1,0],'posterior':[0,-1,0]}
    return view_lightdir[hemi][viewname]

def cropbg(img,bgcolor=None,return_bbox=True,cropcoord=None):
    was_2d=False
    if img.ndim == 2:
        was_2d=True
        img=np.stack([img]*3,axis=2)
    
    if cropcoord is not None:
        return_bbox=False
    
    if cropcoord is None:
        if bgcolor is None:
            bgcolor=img[0,0,:]
        bgcolor=np.reshape(bgcolor,[1,1,img.shape[2]])
        bgimg=np.repeat(np.repeat(bgcolor,img.shape[0],axis=0),img.shape[1],axis=1)
        bgmask=np.logical_not(np.all(img==bgimg,axis=2))
        idx0=np.argwhere(np.any(bgmask,axis=1)).flatten()
        idx1=np.argwhere(np.any(bgmask,axis=0)).flatten()
        cropcoord=[idx0[0],idx0[-1]+1,idx1[0],idx1[-1]+1]
    newimg=img[cropcoord[0]:cropcoord[1],cropcoord[2]:cropcoord[3],:]
    if was_2d:
        newimg=newimg[:,:,0]
    
    if return_bbox:
        return newimg, cropcoord
    else:
        return newimg

def pad_to_max_height(imglist):
    hmax=max([x.shape[0] for x in imglist])
    return [padimage(x,bgcolor=None,padfinalsize=[hmax,-1]) for x in imglist]
    
def pad_to_max_width(imglist):
    wmax=max([x.shape[1] for x in imglist])
    return [padimage(x,bgcolor=None,padfinalsize=[-1,wmax]) for x in imglist]

def padimage(img,bgcolor=None,padamount=None,padfinalsize=None):
    was_2d=False
    if img.ndim == 2:
        was_2d=True
        img=np.stack([img]*3,axis=2)
    
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
    
    if was_2d:
        newimg=newimg[:,:,0]
    
    return newimg

def val2rgb(v, cmap, clim=None):
    if clim is None:
        clim=[v.min(), v.max()]
    if not np.isfinite(clim[0]):
        clim[0]=v.min()
    if not np.isfinite(clim[1]):
        clim[1]=v.max()
    
    v=plt.Normalize(clim[0],clim[1])(v)
    return cmap(v)

    
def vol2mosaic(V, sliceaxis=2, slice_indices=None, mosaic=None,extra_slice_val=0):
    if len(slice_indices) == 0:
        return None, None
    
    if sliceaxis == 0:
        V=np.transpose(V,[1,2,0])
    elif sliceaxis == 1:
        V=np.transpose(V,[0,2,1])
    elif sliceaxis == 2:
        #slice already at end
        pass
    else:
        raise Exception("Slice axis must be 0-2")

    if slice_indices is None:
        slice_indices=np.arange(V.shape[2])

    V=V[:,:,slice_indices]
    numslices=V.shape[2]

    if mosaic is None or all([np.isnan(x) or x<0 for x in mosaic]):
        mosaic=[np.floor(np.sqrt(numslices)), np.nan]
    if np.isnan(mosaic[0]) or mosaic[0]<0:
        mosaic[0]=np.ceil(numslices/mosaic[1])
    elif np.isnan(mosaic[1]) or mosaic[1]<0:
        mosaic[1]=np.ceil(numslices/mosaic[0])
    mosaic=[int(x) for x in mosaic]
    numslices_mosaic=mosaic[0]*mosaic[1]
    if numslices_mosaic>numslices:
        V_extra=extra_slice_val*np.ones((V.shape[0],V.shape[1],numslices_mosaic-numslices))
        V=np.concatenate((V,V_extra),axis=2)

    Vmosaic=np.hstack([np.vstack([V[:,:,mosaic[1]*i+x] for x in range(mosaic[1])]) for i in range(mosaic[0])])

    mosaic_info={"sliceaxis":sliceaxis,"slice_indices":slice_indices,"mosaic":mosaic}
    return Vmosaic,mosaic_info

def mesh_diffuse(verts=None,faces=None,surf=None,adjacency=None,vertvals=None,iters=1):
    if verts is None and faces is None and surf is not None:
        verts=surf[0]
        faces=surf[1]
    
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

def vert2face(surf=None,faces=None, vertvals=None):
    if faces is None and surf is not None:
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

    if vertvals is None:
        return T_vert2face
    else:
        return T_vert2face @ vertvals
    
def face2vert(surf=None,faces=None, facevals=None):
    if faces is None and surf is not None:
        faces=surf[1]
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

    if facevals is None:
        return T_face2vert
    else:
        return T_face2vert @ facevals

def mesh_shading(verts, faces, lightdirection, return_face_values=False):
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
    
    if return_face_values:
        return facenormdot
    
    #now map face norm to vertex
    vertshading=face2vert(faces=faces,facevals=facenormdot)
    
    return vertshading

def fig2pixels(fig,imgbuffer=None,dpi=100):
    if imgbuffer is None:
        imgbuffer=io.BytesIO()
    imgbuffer.seek(0)
    fig.savefig(imgbuffer,format='png',bbox_inches=0,dpi=dpi)
    imgbuffer.seek(0)
    img=Image.open(imgbuffer,'r')
    return np.asarray(img)

def save_image(imgdata,filename,dpi=None):
    if dpi is not None:
        dpi=[dpi,dpi]
    if isinstance(imgdata,Image.Image):
        imgdata.save(filename,dpi=dpi)
    else:
        Image.fromarray(imgdata).save(filename,dpi=dpi)
