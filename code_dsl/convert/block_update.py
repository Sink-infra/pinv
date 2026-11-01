"""Export a validated block update against the current main window baseline."""
import argparse,json,re
from pathlib import Path
from hashlib import sha256
from code_dsl.source import dsl_digest
from code_dsl.convert.window_export import definition,callable_source

def checksum(path):
 return sha256(Path(path).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('baseline-window','window-build','mock-result','output','receipt'):
  p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args()
 report=json.loads(a.mock_result.read_text())
 current=dsl_digest()
 if not (report.get('backend')=='jit' and report.get('suite')=='mock' and report.get('complete') and report.get('passed') and report['build']['dsl_sha256']==current):
  raise ValueError('complete current JIT Mock evidence required')
 rows=report['cases']
 if {x['name'] for x in rows}!={'mock_%02d'%i for i in range(1,13)}:
  raise ValueError('all Mock fixtures required')
 for row in rows:
  if not (row['status']=='Pass' and row['returncode']==0 and row['first']['passed'] and row['final']['passed'] and row.get('steady') and row.get('profile_samples',0)>=9 and row.get('profiler_backend')=='task'):
   raise ValueError('invalid complete Task fixture evidence')
 receipt=json.loads((a.window_build/'build-receipt.json').read_text())
 artifact=receipt['artifacts']['window_cube_module']
 if not (receipt['producer']=='TileLang JIT' and receipt['dsl_sha256']==current and receipt['complete_continuation']):
  raise ValueError('current complete JIT build required')
 if checksum(a.window_build/'kernel.cpp')!=artifact['source_sha256'] or checksum(a.window_build/'kernel.so')!=artifact['library_sha256']:
  raise ValueError('window artifact changed')
 for row in rows:
  if row['name'] in ('mock_03','mock_09'):
   if row['provenance']['library_sha256']!=artifact['library_sha256']:
    raise ValueError('measured window library differs')
 old=(a.baseline_window/'kernel.cpp').read_text()
 new=(a.window_build/'kernel.cpp').read_text()
 kernel=a.output/'op_kernel'
 window=kernel/'pinv_dsl_window.h'
 local=kernel/'pinv_dsl_block_local.h'
 atomic=kernel/'pinv_dsl_block_atomic.h'
 texts={f:f.read_text() for f in (window,local,atomic)}
 cmake={f:f.read_bytes() for f in a.output.rglob('*CMakeLists.txt')}
 records={}
 for symbol,file in [('BlockQuick',local),('BlockRaw',window),('BlockDecode',window),('BlockPack',window),('BlockFinish',window)]:
  before=definition(old,symbol)
  if definition(texts[file],symbol)!=before:
   raise ValueError('current native baseline differs: '+symbol)
  after=definition(new,symbol)
  texts[file]=texts[file].replace(before,after,1)
  records[symbol]={'before':sha256(before.encode()).hexdigest(),'after':sha256(after.encode()).hexdigest()}
 before=callable_source(old,'PvStageWindow',result_scalar='all_admitted').strip()
 if definition(texts[window],'PvStageWindow')!=before:
  raise ValueError('window baseline differs')
 after=callable_source(new,'PvStageWindow',result_scalar='all_admitted').strip()
 texts[window]=texts[window].replace(before,after,1)
 records['PvStageWindow']={'before':sha256(before.encode()).hexdigest(),'after':sha256(after.encode()).hexdigest()}
 bindings=lambda s,name: [line for line in s.splitlines() if line.startswith('#define '+name+'(')]
 for name in ('bf_block',):
  before=bindings(old,name);after=bindings(new,name)
  if len(before)!=1 or len(after)!=1 or texts[window].count(before[0])!=1:
   raise ValueError('block callback ABI changed')
  texts[window]=texts[window].replace(before[0],after[0],1)
 # Put both schedules beside the existing chained Cube stage.
 # Their callback dependencies precede this include in the window composition.
 raw=definition(new,'BlockRaw')
 texts[window]=texts[window].replace(raw,'',1)
 helpers=Path(__file__).parents[1]/'op_kernel/block_lookahead_instructions.h'
 extra=helpers.read_text()+'\nnamespace tl::ascend {\n'+definition(new,'BlockLookaheadRows')+'\n'+raw+'\n}\n'
 binding=bindings(new,'br_lookahead_chain')
 if len(binding)!=1:raise ValueError('lookahead callback ABI missing')
 extra+='\n'+binding[0]+'\n'
 # Macro expansion of the inline Raw body needs the binding first.
 extra=extra.replace(binding[0]+'\n','')
 split='namespace tl::ascend {\n'+definition(new,'BlockLookaheadRows')+'\n'
 extra=extra.replace(split,split+'}\n'+binding[0]+'\nnamespace tl::ascend {\n',1)
 texts[atomic]+='\n'+extra
 if any(re.search(r'\b'+term+r'\b',text) for text in texts.values() for term in ['printf','cout','cerr','puts','write','syscall','dlopen','dlsym','asm','freopen']):
  raise ValueError('forbidden upload token')
 for f,text in texts.items():f.write_text(text.rstrip()+'\n')
 if any(f.read_bytes()!=content for f,content in cmake.items()):
  raise ValueError('CMake changed')
 a.receipt.parent.mkdir(parents=True,exist_ok=True)
 a.receipt.write_text(json.dumps({'dsl_sha256':current,'mock_result':str(a.mock_result),'mock_result_sha256':checksum(a.mock_result),'jit_library_sha256':artifact['library_sha256'],'functions':records,'outputs':{str(f):checksum(f) for f in texts}},indent=2)+'\n')
if __name__=='__main__':main()
