"""Static tensor views for explicitly synchronized Cube-only components."""
import re
MARKER='#define PINV_STATIC_CUBE_STORAGE 1'
ENTRY='extern "C" __global__ __aicore__ void linalg_pinv_tilelang_kernel('
BOUNDARY='\nvoid linalg_pinv_tilelang_kernel_tiling('

def lower_source(source):
 if source.count(MARKER)!=1:raise ValueError('one Cube storage policy marker required')
 start=source.index(ENTRY);end=source.index(BOUNDARY,start);original=source[start:end];body=original
 if body.count('  AscendC::TPipe pipe;')!=1 or body.count('  pipe.Destroy();')!=1:raise ValueError('unexpected Cube pipe lifetime')
 if any(x in body for x in ['GetTPipePtr','FetchEventID','CreateVecIndex','pv_reduce_','pv_sync_all']):raise ValueError('dynamic event consumer in static Cube component')
 changes=[]
 def sub(old,new):
  nonlocal body
  if body.count(old)!=1:raise ValueError('ambiguous Cube allocation')
  body=body.replace(old,new,1);changes.append((old,new))
 allocation=re.compile(r'  AscendC::TBuf<AscendC::TPosition::(\w+)> (\w+);\s*pipe.InitBuffer\(\2, ([0-9]+)\);')
 pools={}
 for m in list(allocation.finditer(original)):
  pos,name,size=m.groups();size=int(size)
  if pos not in {'A1','A2','B2','CO1','VECCALC'} or size%32:raise ValueError('unsupported Cube pool')
  pools[name]=(pos,size)
  new=f'  AscendC::LocalTensor<float> {name}(AscendC::TPosition::{pos}, 0, {size//4});' if pos in {'A2','B2'} else ''
  sub(m[0],new)
 if set(pools)!={'ascend_l0a','ascend_l0b','ascend_l1','ascend_l0c','ascend_ub'}:raise ValueError('incomplete Cube allocation set')
 views=re.compile(r'  auto (\w+) = (\w+).GetWithOffset<([\w:]+)>\(([0-9]+), ([0-9]+)\);')
 for m in list(views.finditer(body)):
  name,pool,dtype,count,offset=m.groups();count=int(count);offset=int(offset);pos,size=pools[pool]
  width={'float':4,'half':2,'int32_t':4,'int8_t':1}[dtype]
  if offset%32 or count*width%32 or offset+count*width>size:raise ValueError('Cube tensor outside pool')
  sub(m[0],f'  AscendC::LocalTensor<{dtype}> {name}(AscendC::TPosition::{pos}, {offset}, {count});')
 sub('  AscendC::TPipe pipe;','  AscendC::InitSocState();');sub('  pipe.Destroy();','')
 if 'pipe.' in body or '.GetWithOffset<' in body:raise ValueError('unlowered Cube resource')
 for name,(pos,_) in pools.items():
  if pos not in {'A2','B2'} and name in body:raise ValueError('unbounded Cube pool consumer')
 restored=body
 for old,new in reversed(changes):
  if new:restored=restored.replace(new,old,1)
  else:
   # Empty declarations are restored through their stable source positions below.
   pass
 # Every operation outside matched resource spans must be byte-identical.
 def stripped(text):
  text=allocation.sub('',text);text=views.sub('',text)
  text=re.sub(r'  AscendC::LocalTensor<[^>]+> \w+\(AscendC::TPosition::\w+, [0-9]+, [0-9]+\);','',text)
  for token in ['  AscendC::TPipe pipe;','  AscendC::InitSocState();','  pipe.Destroy();']:text=text.replace(token,'')
  return ''.join(text.split())
 if stripped(original)!=stripped(body):raise ValueError('Cube arithmetic or ordering changed')
 return source[:start]+body+source[end:]
