"""Build public pinned source and consume an already-public G7 artifact."""
import argparse,hashlib,json,pathlib,shutil,subprocess,sys
ROOT=pathlib.Path(__file__).parent
SIM_SHA='7476a81954020087da31d41d16fddf475746ec2d'
PYBIND_SHA='3e9dfa2866941655c56877882565e7577de6fc7b'
def run(args,**kw):subprocess.run(args,check=True,**kw)
def build(destination):
    destination.mkdir(parents=True,exist_ok=True);source=destination/'source';builddir=destination/'build'
    if not source.exists():
        run(['git','clone','https://github.com/gamerpuppy/sts_lightspeed.git',str(source)])
        run(['git','-C',str(source),'checkout',SIM_SHA])
        run(['git','-C',str(source),'submodule','update','--init','--recursive'])
        run(['git','-C',str(source/'pybind11'),'fetch','origin',PYBIND_SHA])
        run(['git','-C',str(source/'pybind11'),'checkout',PYBIND_SHA])
        run(['git','-C',str(source),'apply',str(ROOT/'upstream-hooks.patch')])
        shutil.copy2(ROOT/'sampler-binding.cpp',source/'bindings/slaythespire.cpp')
    cmake=shutil.which('cmake') or str(pathlib.Path(sys.executable).with_name('cmake'))
    run([cmake,'-S',str(source),'-B',str(builddir),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -include algorithm','-DPYTHON_EXECUTABLE='+sys.executable])
    run([cmake,'--build',str(builddir),'--target','slaythespire','-j2'])
    binary=next(builddir.glob('slaythespire*.so'));(ROOT/'native').mkdir(exist_ok=True);shutil.copy2(binary,ROOT/'native'/binary.name)
    result={'simulator_sha':SIM_SHA,'pybind_sha':PYBIND_SHA,'native_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'native_origin':'compiled from public pinned source during this workflow'}
    (ROOT/'native-build-provenance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
def g7(downloads):
    expected=json.loads((ROOT/'package-manifest.json').read_text())['g7_sha256'];matches=[]
    for p in downloads.rglob('*.pt'):
        if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==expected:matches.append(p)
    if not matches:raise RuntimeError('Expected G7 checkpoint not found in the existing public artifact')
    import torch
    weights=torch.load(matches[0],weights_only=True,map_location='cpu')
    if not isinstance(weights,dict) or not all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in weights.items()):raise RuntimeError('G7 must contain only a named tensor state dictionary')
    shutil.copy2(matches[0],ROOT/'g7.pt')
    print(json.dumps({'g7_sha256':expected,'tensor_parameters':len(weights),'source':'already-public sts-public-ci Actions artifact 11079234863','new_checkpoint_published_to_git':False}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','g7']);p.add_argument('--destination',type=pathlib.Path);p.add_argument('--downloads',type=pathlib.Path);a=p.parse_args()
    if a.stage=='build':build(a.destination.resolve())
    else:g7(a.downloads.resolve())
