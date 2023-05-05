import sys
import argparse

from brainmontage.brainmontage import *

def parse_argument_montageplot(argv):
    parser=argparse.ArgumentParser(description='save surface ROI montage')
    
    parser.add_argument('--input',action='store',dest='inputfile')
    parser.add_argument('--inputfield',action='store',dest='inputfieldname')
    parser.add_argument('--views',action='append',dest='viewnames',nargs='*')
    parser.add_argument('--outputimage',action='store',dest='outputimage')
    parser.add_argument('--surftype',action='store',dest='surftype',default='infl')
    parser.add_argument('--cmap','--colormap',action='store',dest='cmapname',default='magma')
    parser.add_argument('--cmapfile','--colormapfile',action='store',dest='cmapfile')
    parser.add_argument('--clim', action='append',dest='clim',nargs=2)
    parser.add_argument('--roilut',action='store',dest='roilutfile')
    parser.add_argument('--atlasname',action='store',dest='atlasname')
    parser.add_argument('--lhannot',action='store',dest='lhannotfile')
    parser.add_argument('--rhannot',action='store',dest='rhannotfile')
    parser.add_argument('--lhannotprefix',action='store',dest='lhannotprefix')
    parser.add_argument('--rhannotprefix',action='store',dest='rhannotprefix')
    parser.add_argument('--annotsurfacename',action='store',dest='annotsurface',default='fsaverage5')
    parser.add_argument('--noshading',action='store_true',dest='noshading')
    parser.add_argument('--inputvals',action='append',dest='inputvals',nargs='*')
    
    args=parser.parse_args(argv)
    return args

def run_montageplot(argv):
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
    
    inputvals_arg=flatarglist(args.inputvals)
    
    if len(clim)==2:
        clim=[np.float32(x) for x in clim]
    else:
        clim=None
    
    #nilearn uses 'Spectral' instead of matplotlib 'spectral'
    if cmapname.lower()=='spectral':
        cmap='Spectral'
    elif cmapname.lower()=='spectral_r':
        cmap='Spectral_r'
    else:
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
        atlasinfo={'atlasname':None,'roilutfile':roilutfile,'lhannotfile':lhannotfile,'rhannotfile':rhannotfile,
            'annotsurfacename':annotsurfacename,'lhannotprefix':lhannotprefix,'rhannotprefix':rhannotprefix}
    
    surftype_allowed=['white','infl','pial']
    if not surftype in surftype_allowed:
        surftype_found=[s for s in surftype_allowed if surftype.lower().startswith(s)]
        if len(surftype_found)>0:
            surftype=surftype_found[0]
    
    img=create_montage_figure(roivals,atlasinfo=atlas_info,
        viewnames=viewnames,surftype=surftype,clim=clim,colormap=cmap,noshading=no_shading,
        outputimagefile=outputimage)
    
if __name__ == "__main__":
    run_montageplot(sys.argv[1:])
