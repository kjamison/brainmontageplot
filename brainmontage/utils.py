
import numpy as np
from PIL import Image
from scipy import sparse
import io

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

def save_image(imgdata,filename):
    Image.fromarray(imgdata).save(filename)
