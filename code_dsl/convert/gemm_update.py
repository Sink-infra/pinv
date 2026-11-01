"""Export a validated explicit Cube schedule into the current native window."""
import argparse,json,re
from pathlib import Path
from hashlib import sha256
from code_dsl.source import dsl_digest
from code_dsl.convert.window_export import definition

def checksum(path):
 return sha256(Path(path).read_bytes()).hexdigest()

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ('baseline-window','baseline-header','window-build','mock-result','output','receipt'):
  parser.add_argument('--'+name,type=Path,required=True)
 a=parser.parse_args();current=dsl_digest();report=json.loads(a.mock_result.read_text())
 if not (report.get('backend')=='jit' and report.get('suite')=='mock' and report.get('complete') and report.get('passed') and report['build']['dsl_sha256']==current):
  raise ValueError('complete current JIT Mock evidence required')
 rows=report['cases']
 if {row['name'] for row in rows}!={'mock_%02d'%i for i in range(1,13)}:raise ValueError('all Mock fixtures required')
 for row in rows:
  if not (row['status']=='Pass' and row['returncode']==0 and row['first']['passed'] and row['final']['passed'] and row.get('steady') and row.get('profile_samples',0)>=9 and row.get('profiler_backend')=='task'):
   raise ValueError('invalid complete Task fixture evidence')
 receipt=json.loads((a.window_build/'build-receipt.json').read_text());artifact=receipt['artifacts']['window_cube_module']
 if not (receipt['producer']=='TileLang JIT' and receipt['dsl_sha256']==current and receipt['complete_continuation']):
  raise ValueError('current complete JIT build required')
 for suffix,key in [('cpp','source_sha256'),('so','library_sha256')]:
  if checksum(a.window_build/('kernel.'+suffix))!=artifact[key]:raise ValueError('window artifact changed')
 for row in rows:
  if row['name'] in ('mock_03','mock_09') and row['provenance']['library_sha256']!=artifact['library_sha256']:
   raise ValueError('measured window library differs')
 old=(a.baseline_window/'kernel.cpp').read_text();new=(a.window_build/'kernel.cpp').read_text()
 window=a.output/'op_kernel/pinv_dsl_window.h';text=window.read_text();before=checksum(window)
 cmake={p:checksum(p) for p in a.output.rglob('*CMakeList*') if p.is_file()}
 old_gemm=definition(old,'BlockGemm')
 if definition(text,'BlockGemm')!=old_gemm:raise ValueError('native GEMM baseline changed')
 helper=Path(__file__).parents[1]/'op_kernel/block_gemm_pipeline_instructions.h'
 helper_text=helper.read_text();names=re.findall(r'__aicore__ inline void (\w+)\(',helper_text)
 if len(names)!=4:raise ValueError('unexpected elementary instruction set')
 functions=[definition(helper_text,name) for name in names]
 legacy=definition(new,'BlockGemmLegacy');pipeline=definition(new,'BlockGemm')
 if legacy.replace('BlockGemmLegacy(','BlockGemm(',1)!=old_gemm:raise ValueError('legacy numerical body changed')
 macro=re.findall(r'^#define bg_legacy[^\n]+',new,re.M)
 if len(macro)!=1:raise ValueError('legacy ABI bridge changed')
 replacement='\n'.join(functions+[legacy,macro[0],pipeline])
 text=text.replace(old_gemm,replacement,1)
 old_copy=definition(a.baseline_header.read_text(),'copy_gm_to_l1')
 if definition(text,'copy_gm_to_l1')!=old_copy:raise ValueError('native copy baseline changed')
 header=Path(__file__).parents[1]/'op_kernel/recovery_instructions.h'
 new_copy=definition(header.read_text(),'copy_gm_to_l1')
 text=text.replace(old_copy,new_copy,1)
 for path,expected in cmake.items():
  if checksum(path)!=expected:raise ValueError('CMake changed')
 if checksum(window)!=before:raise ValueError('native source changed during export')
 window.write_text(text)
 a.receipt.parent.mkdir(parents=True,exist_ok=True)
 record={'dsl_sha256':current,'mock_result_sha256':checksum(a.mock_result),'jit_artifact':artifact,'native_before_sha256':before,'native_after_sha256':checksum(window),'legacy_body_unchanged':True,'pipeline_body_sha256':sha256(pipeline.encode()).hexdigest(),'primitive_header_sha256':checksum(helper),'copy_header_sha256':checksum(header),'cmake_unchanged':True}
 a.receipt.write_text(json.dumps(record,indent=2))
 print(json.dumps({'updated':str(window),'dsl_sha256':current}))
if __name__=='__main__':main()
