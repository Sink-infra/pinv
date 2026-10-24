"""Verified static UB lowering for explicit-event resident TileLang entries."""
import re
MARKER='#define PINV_STATIC_RESIDENT_UB 1\n'
ENTRY='extern "C" __global__ __aicore__ void linalg_pinv_tilelang_kernel('
BOUNDARY='\nvoid linalg_pinv_tilelang_kernel_tiling()'

def lower_source(source):
 if source.count(MARKER)!=1:raise ValueError('static storage requires one explicit policy marker')
 start=source.index(ENTRY);end=source.index(BOUNDARY,start);before=source[start:end];body=before
 if 'KERNEL_TYPE_AIV_ONLY' not in body or 'tl::ascend::pv_module_bind(' not in body or 'pv_sync_all(' in body:raise ValueError('static storage requires independent Vector jobs')
 if re.search(r'tl::ascend::pv_reduce_(?:sum|max)\(',body):raise ValueError('SDK aggregate reductions require an event allocator')
 if 'tl::ascend::cp_iota(' in body or 'CreateVecIndex(' in body:raise ValueError('A2 index generator requires an event allocator')
 allocation=re.compile(r'^  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer\(ascend_ub, ([0-9]+)\);$',re.M)
 allocations=list(allocation.finditer(body))
 if len(allocations)!=1:raise ValueError('one bounded UB allocation required')
 extent=int(allocations[0][1]);body=allocation.sub('',body)
 views=re.compile(r'^( +)auto (\w+) = ascend_ub.GetWithOffset<([\w:]+)>\(([0-9]+), ([0-9]+)\);$',re.M)
 widths={'float':4,'uint32_t':4,'int32_t':4,'int':4,'uint':4,'half':2,'int64_t':8,'uint64_t':8}
 records=[]
 def replace(m):
  indent,name,dtype,count,offset=m.groups();count=int(count);offset=int(offset)
  if dtype not in widths or count<=0 or offset%32 or (count*widths[dtype])%32 or offset+count*widths[dtype]>extent:raise ValueError('invalid static view')
  replacement=indent+f'AscendC::LocalTensor<{dtype}> {name}(AscendC::TPosition::VECCALC, {offset}, {count});'
  records.append((m[0],replacement));return replacement
 body=views.sub(replace,body)
 if not records or 'ascend_ub' in body:raise ValueError('unknown allocator use')
 if body.count('  AscendC::TPipe pipe;')!=1 or body.count('  pipe.Destroy();')!=1:raise ValueError('unexpected pipe lifetime')
 body=body.replace('  AscendC::TPipe pipe;','  AscendC::InitSocState();',1).replace('  pipe.Destroy();','',1)
 if re.search(r'\bpipe\b',body):raise ValueError('remaining event allocator consumer')
 expected=allocation.sub('',before).replace('  AscendC::TPipe pipe;','',1).replace('  pipe.Destroy();','',1)
 restored=body.replace('  AscendC::InitSocState();','',1)
 for original,replacement in records:restored=restored.replace(replacement,original,1)
 if restored!=expected:raise ValueError('numerical instructions changed during storage lowering')
 return source[:start]+body+source[end:]
