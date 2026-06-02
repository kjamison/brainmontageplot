function [imgpath, img] = brainmontage(varargin)
% run brainmontage.py from MATLAB, using CLI-like arguments
%
% Will return the path to the generated image and the image array itself
% If no output image path is specified, a temporary file will be created
% and deleted after reading the image.
%
% Examples:
% [imgpath,~] = brainmontage(vals, '--atlasname','fs86','--clim',[-1 1],...)
% [~,img] = brainmontage(vals, '--atlasname fs86 --clim -1 1 ...')
% brainmontage('--inputvals 1 2 3 4 ... --atlasname fs86 --clim -1 1 ...')
% brainmontage(vals,'--command','/path/to/python /path/to/brainmontage.py','--atlasname','fs86','--clim',[-1 1],...)


args=varargin;
imgpath='';
imgpath_istemp=true;

%% Configuration options for executing brainmontage.py
%%%%%%%%%%%%%%%%%%%

%By default, the script will try to find brainmontage.py in the parent
%directory of this matlab script. This can be overridden by hardcoding the 
%path to brainmontage.py below. The entire BRAINMONTAGE_RUNCMD 
%"python /path/to/brainmontage.py" prefix can be overridden by providing 
%'--command' argument to this function.

%%%%%%%%%%%%%%%%%%%
%1. Identify the path to brainmontage.py:

%if the matlab script is in the original python package directory, find 
% the path automatically
[scriptdir,~,~]=fileparts(mfilename("fullpath"));
[scriptdir,~,~]=fileparts(scriptdir);
BRAINMONTAGE_PYSCRIPT=fullfile(scriptdir,'brainmontage','brainmontage.py');

%if the matlab script is NOT in the original python package directory, 
% point to the python script here:
%BRAINMONTAGE_PYSCRIPT='~/Source/brainmontageplot/brainmontage/brainmontage.py';

%%%%%%%%%%%%%%%%%%%
%2. How to execute the python script:

%if brainmontage package and dependencies are installed globally
%BRAINMONTAGE_PYCMD='python';
%BRAINMONTAGE_PYCMD='/usr/bin/python';

%if have conda installed and the package is installed inside the
%base or default environment:
%BRAINMONTAGE_PYCMD='source ~/.bashrc >/dev/null; $(which conda) run python';
BRAINMONTAGE_PYCMD='source ~/.bash_profile >/dev/null; $(which conda) run python';

%if it is in a custom conda environment:
%BRAINMONTAGE_PYCMD='source $HOME/.bash_profile >/dev/null; $(which conda) run -n MYENV python';

%%%%%%%%%%%%%%%%%%%
%3. (Optional) If brainmontage CLI is installed, you can specify its path here

BRAINMONTAGE_CLI=''; %default = empty = use python script, not installed CLI
%[~,BRAINMONTAGE_CLI]=system('which brainmontage'); BRAINMONTAGE_CLI=strtrim(BRAINMONTAGE_CLI);
%BRAINMONTAGE_CLI='~/.local/bin/brainmontage'; %path to brainmontage CLI if installed globally

if(~isempty(BRAINMONTAGE_CLI))
    BRAINMONTAGE_RUNCMD=BRAINMONTAGE_CLI;
else
    BRAINMONTAGE_RUNCMD=strjoin({BRAINMONTAGE_PYCMD,BRAINMONTAGE_PYSCRIPT});
end
%%%%%%%%%%%%%%%%%%%

%%
valarg={};
input_tempfile='';
if(isnumeric(args{1}))
    tmpd=tempdir();
    if(tmpd(end)=='/')
        tmpd=tmpd(1:end-1);
    end
    input_tempfile=sprintf('%s/tmpinput.txt',tmpd);
    vals=args{1};
    save(input_tempfile,'vals','-ascii');
    valarg={'--input',input_tempfile};

    %valarg={'--inputvals',num2str(reshape(args{1},1,[]))};
    args=args(2:end);
end

if(numel(args)==1 && ischar(args{1}))
    args=strsplit(args{1},'--');
    newargs={};
    for i = 1:numel(args)
        if(isempty(args{i}))
            continue;
        end
        tmparg=strsplit(args{i});
        tmparg{1}=strtrim(['--' tmparg{1}]);
        if(numel(tmparg)>1)
            tmparg={tmparg{1} strtrim(strjoin(tmparg(2:end)))};
        end
        newargs=[newargs tmparg];
    end
    args=newargs;
end
if(numel(args)>1 && any(strcmpi(args,'--command')))
    cmdidx=find(strcmpi(args,'--command'));
    BRAINMONTAGE_RUNCMD=args{cmdidx+1};
    args=args(setdiff(1:numel(args),[cmdidx cmdidx+1]));
end


for i = 1:numel(args)
    if(ischar(args{i}) && strcmpi(args{i},'--outputimage'))
        imgpath=args{i+1};
        imgpath_istemp=false;
    end
    if(isnumeric(args{i}) || islogical(args{i}))
        args{i}=num2str(reshape(args{i},1,[]));
    end
end
if(isempty(imgpath))
    tmpd=tempdir();
    if(tmpd(end)=='/')
        tmpd=tmpd(1:end-1);
    end
    imgpath=sprintf('%s/tempimg.png',tmpd);
    args{end+1}='--outputimage';
    args{end+1}=imgpath;
end
if(~isempty(valarg))
    args=[args valarg];
end

argstring=strjoin(args,' ');
errcode=system(strjoin({BRAINMONTAGE_RUNCMD,argstring}));

if(~isempty(input_tempfile))
    delete(input_tempfile);
end
if(errcode>0)
    error('Error executing brainmontage.py. Check the input values and try again.');
end
if(~exist(imgpath,'file'))
    img=[];
    return
end
img=imread(imgpath);
if(imgpath_istemp)
    delete(imgpath); % Delete the temporary image file
end