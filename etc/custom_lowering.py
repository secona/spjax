import jax
from jax.interpreters import mlir
from jax._src import core
from jax._src.lib.mlir.dialects import arith

custom_square_p = core.Primitive("custom_square")

def custom_square(x):
    return custom_square_p.bind(x)

@custom_square_p.def_impl
def custom_square_impl(x):
    return x * x

@custom_square_p.def_abstract_eval
def custom_square_abstract_eval(x):
    return core.ShapedArray(x.shape, x.dtype)

def custom_square_mlir_lowering(ctx: mlir.LoweringRuleContext, x, *args, **kwargs):
    result = arith.mulf(x, x)
    return [result]

mlir.register_lowering(custom_square_p, custom_square_mlir_lowering)

print(jax.jit(custom_square).lower(10.0).compiler_ir())
