"""Elementary FP32-pair arithmetic, using explicit result buffers.

The pinned backend's custom-instruction hook emits statements. Numerical
control stays in DSL; these calls implement scalar two-component primitives.
"""
from code_dsl.op_kernel.intrinsics import emit


def pack(out,h,l):return emit('pv_pair_pack',out.access_ptr('w'),h,l)
def plus(out,a,b):return emit('pv_pair_plus',out.access_ptr('w'),a.access_ptr('r'),b.access_ptr('r'))
def minus(out,a,b):return emit('pv_pair_minus',out.access_ptr('w'),a.access_ptr('r'),b.access_ptr('r'))
def times(out,a,b):return emit('pv_pair_times',out.access_ptr('w'),a.access_ptr('r'),b.access_ptr('r'))
def divide(out,a,b):return emit('pv_pair_divide',out.access_ptr('w'),a.access_ptr('r'),b.access_ptr('r'))
