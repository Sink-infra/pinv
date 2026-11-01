"""Bounded UB views with a caller-owned Vector event allocator."""
import re
from hashlib import sha256

def lower_function(function,capacity_bytes):
 original=function
 declarations=re.compile(r'(?m)^( +)AscendC::TPipe pipe;$')
 matches=list(declarations.finditer(function))
 if len(matches)!=1:raise ValueError('one Vector pipe owner required')
 allocation=re.compile(r'(?m)^ +AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer\(ascend_ub, ([0-9]+)\);$')
 allocations=list(allocation.finditer(function))
 if len(allocations)!=1 or int(allocations[0][1])>capacity_bytes:raise ValueError('child UB exceeds caller reservation')
 views=re.compile(r'(?m)^( +)auto (\w+) = ascend_ub.GetWithOffset<([\w:]+)>\(([0-9]+), ([0-9]+)\);$')
 widths={'float':4,'int':4,'int32_t':4,'uint32_t':4,'half':2,'uint8_t':1}
 records=[]
 def view(m):
  indent,name,dtype,count,offset=m.groups();count=int(count);offset=int(offset)
  if dtype not in widths or offset%32 or count*widths[dtype]%32 or offset+count*widths[dtype]>capacity_bytes:raise ValueError('unbounded borrowed view')
  records.append({'name':name,'offset':offset,'bytes':count*widths[dtype]})
  return indent+f'AscendC::LocalTensor<{dtype}> {name}(AscendC::TPosition::VECCALC, {offset}, {count});'
 function=allocation.sub('',function)
 function=views.sub(view,function)
 function=declarations.sub(lambda m:m[1]+'AscendC::TPipe &pipe=*::GetTPipePtr();',function)
 destroy=re.compile(r'(?m)^ +pipe.Destroy\(\);$')
 if len(destroy.findall(function))!=1:raise ValueError('one child teardown required')
 function=destroy.sub('',function)
 if 'ascend_ub' in function:raise ValueError('unknown child allocation consumer')
 def stripped(s):
  s=allocation.sub('',s);s=views.sub('',s);s=declarations.sub('',s);s=destroy.sub('',s)
  s=re.sub(r'(?m)^ +AscendC::TPipe &pipe=\*::GetTPipePtr\(\);$','',s)
  s=re.sub(r'(?m)^ +AscendC::LocalTensor<[^>]+> \w+\(AscendC::TPosition::VECCALC, [0-9]+, [0-9]+\);$','',s)
  return ''.join(s.split())
 if stripped(original)!=stripped(function):raise ValueError('arithmetic or synchronization changed')
 return function,{'capacity_bytes':capacity_bytes,'views':records,'unchanged_operations_sha256':sha256(stripped(original).encode()).hexdigest()}
