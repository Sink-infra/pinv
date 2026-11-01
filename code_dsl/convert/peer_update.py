"""Export validated local and peer TileLang bodies against a matched main baseline."""
import argparse,json,re
from pathlib import Path
from hashlib import sha256
from code_dsl.source import dsl_digest
from code_dsl.convert.window_export import definition,callable_source

def checksum(path):
 return sha256(Path(path).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('baseline-module','baseline-peer','module-build','peer-build','mock-result','output','receipt'):
  p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--extra-helper',type=Path,action='append',default=[])
 p.add_argument('--local-linkage',choices=['static','tiling-template'],default='static')
 a=p.parse_args();current=dsl_digest();report=json.loads(a.mock_result.read_text())
 if not (report.get('backend')=='jit' and report.get('suite')=='mock' and report.get('complete') and report.get('passed') and report['build']['dsl_sha256']==current):
  raise ValueError('complete current JIT Mock evidence required')
 cases=report['cases']
 if {x['name'] for x in cases}!={'mock_%02d'%i for i in range(1,13)}:
  raise ValueError('all Mock fixtures required')
 for row in cases:
  if not (row['status']=='Pass' and row['returncode']==0 and row['first']['passed'] and row['final']['passed'] and row.get('steady') and row.get('profile_samples',0)>=9 and row.get('profiler_backend')=='task'):
   raise ValueError('invalid complete Task fixture evidence')
 records={}
 builds={}
 for key,root,artifact,fixture in [('local',a.module_build,'local_module',('mock_02','mock_05','mock_08','mock_11')),('peer',a.peer_build,'peer_pipeline_module',('mock_06','mock_12'))]:
  receipt=json.loads((root/'build-receipt.json').read_text());entry=receipt['artifacts'][artifact]
  cpp=root/('local/kernel.cpp' if key=='local' else 'kernel.cpp')
  so=cpp.with_suffix('.so')
  if not (receipt['producer']=='TileLang JIT' and receipt['dsl_sha256']==current and checksum(cpp)==entry['source_sha256'] and checksum(so)==entry['library_sha256']):
   raise ValueError('current immutable JIT build required: '+key)
  for row in cases:
   if row['name'] in fixture and row['provenance']['library_sha256']!=entry['library_sha256']:
    raise ValueError('measured library differs: '+row['name'])
  builds[key]=cpp.read_text()
  records[key]={'source_sha256':entry['source_sha256'],'library_sha256':entry['library_sha256']}
 kernel=a.output/'op_kernel'
 files={name:kernel/name for name in ('pinv_dsl_local.h','pinv_dsl_peer_tiles.h','pinv_dsl_peer_rows.h','pinv_dsl_common.h')}
 texts={name:path.read_text() for name,path in files.items()}
 cmake={path:path.read_bytes() for path in a.output.rglob('*CMakeList*') if path.is_file()}
 old=callable_source((a.baseline_module/'local/kernel.cpp').read_text(),'PvStageLocal',noinline=True).strip()
 existing=definition(texts['pinv_dsl_local.h'],'PvStageLocal')
 signature='void PvStageLocal('
 if existing[existing.index(signature):]!=old[old.index(signature):]:
  raise ValueError('current local native baseline differs')
 new=callable_source(builds['local'],'PvStageLocal',noinline=True).strip()
 if a.local_linkage=='tiling-template':
  new=new.replace('__aicore__ static __attribute__((noinline))','__aicore__ __attribute__((noinline))',1)
  marker='#if defined(TILING_KEY_VAR)\ntemplate<uint32_t PvLocalKey=TILING_KEY_VAR>\n#else\ntemplate<uint32_t PvLocalKey=0>\n#endif\n'
  if marker not in texts['pinv_dsl_local.h']:new=marker+new
 texts['pinv_dsl_local.h']=texts['pinv_dsl_local.h'].replace(existing,new,1)
 records['PvStageLocal']={'before':sha256(old.encode()).hexdigest(),'after':sha256(new.encode()).hexdigest()}
 baseline=a.baseline_peer/'kernel.cpp';before=baseline.read_text()
 for original,symbol,name in [('PvDslPipelinePeer','PvDslPeerTileProducer','pinv_dsl_peer_tiles.h'),('PvDslPipelineRows','PvDslPeerTileRows','pinv_dsl_peer_rows.h')]:
  old=definition(before,original).replace(original+'(',symbol+'(',1)
  if definition(texts[name],symbol)!=old:
   raise ValueError('current peer native baseline differs: '+symbol)
  new=definition(builds['peer'],original).replace(original+'(',symbol+'(',1)
  texts[name]=texts[name].replace(old,new,1)
  records[symbol]={'before':sha256(old.encode()).hexdigest(),'after':sha256(new.encode()).hexdigest()}
 helpers=[Path(__file__).parents[1]/'op_kernel/peer_finish_instructions.h',*a.extra_helper]
 for helper in helpers:
  extra=helper.read_text();additions=[]
  symbols=re.findall(r'__aicore__ inline void (\w+)\(',extra)
  if not symbols:raise ValueError('elementary helper definitions required')
  for symbol in symbols:
   body=definition(extra,symbol)
   try:existing=definition(texts['pinv_dsl_common.h'],symbol)
   except ValueError:existing=None
   if existing is None:additions.append(body)
   elif existing!=body:raise ValueError('existing elementary helper differs: '+symbol)
  if additions:texts['pinv_dsl_common.h']+='\nnamespace tl::ascend {\n'+'\n'.join(additions)+'\n}\n'
 for path,expected in cmake.items():
  if path.read_bytes()!=expected:raise ValueError('CMake changed')
 native_before={name:checksum(path) for name,path in files.items()}
 for name,path in files.items():
  if checksum(path)!=native_before[name]:raise ValueError('native baseline changed during export')
 for name,path in files.items():path.write_text(texts[name])
 a.receipt.parent.mkdir(parents=True,exist_ok=True)
 a.receipt.write_text(json.dumps({'dsl_sha256':current,'mock_result_sha256':checksum(a.mock_result),'strategies':records,'local_linkage':a.local_linkage,'elementary_helpers_sha256':{str(helper):checksum(helper) for helper in helpers},'native_before':native_before,'native_after':{name:checksum(path) for name,path in files.items()},'cmake_unchanged':True},indent=2))
 print(json.dumps({'updated':list(files),'dsl_sha256':current}))
if __name__=='__main__':main()
