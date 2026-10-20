"""Allocator lifetime and runtime includes for a mixed TileLang composition.

AIC callback stages own their pipes. Only the outer AIV control owns a UB
pipe; nesting an AIC control pipe consumes its child's reserved events.
The callbacks restore the AIV parent pointer after ending their own pipe.
"""
from pathlib import Path
from code_dsl.op_kernel.backend import vector_only_backend
from contextlib import contextmanager
from tilelang.jit.adapter.wrapper import TLWrapper
@contextmanager
def _composition_includes():
 previous=TLWrapper.wrap
 def wrap(self,source):
  source=previous(self,source)
  token='#include "tl_templates/ascend/common.h"'
  assert source.count(token)==1

  source=source.replace(token,'',1)
  start=source.index('extern "C" __global__ __aicore__ void linalg_pinv_tilelang_kernel(')
  end=source.index('\nvoid linalg_pinv_tilelang_kernel_tiling(',start)
  body=source[start:end]
  resources=[]
  for line in body.splitlines():
   if line.strip()=='AscendC::TPipe pipe;' or 'ascend_ub' in line:
    resources.append(line)
  assert len(resources)==3,resources
  for line in resources:body=body.replace(line+'\n','',1)
  assert body.count('  pipe.Destroy();\n')==1
  body=body.replace('  pipe.Destroy();\n','',1)
  marker='  if ASCEND_IS_AIV {\n    tl::ascend::fr_bind'
  assert body.count(marker)==1
  body=body.replace(marker,'  if ASCEND_IS_AIV {\n'+'\n'.join('  '+line for line in resources)+'\n    tl::ascend::fr_bind',1)
  a=body.index('  if ASCEND_IS_AIV {\n    AscendC::TPipe pipe;')
  opening=body.index('{',a);depth=1;closing=opening+1
  while depth:
   if body[closing]=='{':depth+=1
   if body[closing]=='}':depth-=1
   closing+=1
  body=body[:closing-1]+'  pipe.Destroy();\n  '+body[closing-1:]
  source=source[:start]+body+source[end:]

  from code_dsl.op_kernel.kernels import FLAT_TENSOR_METADATA
  token='#include "'+Path(__file__).with_name("recovery_instructions.h").as_posix()+'"'
  source=source.removeprefix(FLAT_TENSOR_METADATA)
  return FLAT_TENSOR_METADATA+token+'\nusing namespace Catlass; using uint=unsigned int; using uchar=unsigned char; using ushort=unsigned short;\n'+source
 TLWrapper.wrap=wrap
 try:yield
 finally:TLWrapper.wrap=previous

@contextmanager
def composition_backend():
    with vector_only_backend(paired=True),_composition_includes():
        yield


def cube_component_source(source):
    from code_dsl.op_kernel.kernels import FLAT_TENSOR_METADATA
    token='#include "tl_templates/ascend/common.h"'
    if source.count(token)!=1:
        raise ValueError('unexpected Cube runtime include')
    source=source.replace(token,'#include "'+Path(__file__).with_name('recovery_instructions.h').as_posix()+'"',1)
    if not source.startswith(FLAT_TENSOR_METADATA):source=FLAT_TENSOR_METADATA+source
    start=source.index('extern "C" __global__ __aicore__ void linalg_pinv_tilelang_kernel(')
    boundary=source.index('\nvoid linalg_pinv_tilelang_kernel_tiling(',start)
    early='  pipe.Destroy();\n'
    if source.count(early)!=1 or not start<source.index(early)<boundary:
        raise ValueError('unexpected Cube allocator lifetime')
    source=source.replace(early,'',1)
    boundary=source.index('\nvoid linalg_pinv_tilelang_kernel_tiling(',start)
    close=source.rfind('\n}',start,boundary)
    return source[:close]+'\n  pipe.Destroy();'+source[close:]
