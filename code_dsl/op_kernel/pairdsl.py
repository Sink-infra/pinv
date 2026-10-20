"""Expand pair expressions into TileLang statements with explicit UB results.

This is a syntactic frontend, not a numerical implementation. The pinned
Ascend backend supports external instructions as statements only. Every
``w_*`` variable is two FP32 values. All loops, branches and scalar decisions
remain in the expanded TileLang macro and hence in its TIR.
"""
import ast
import inspect
import linecache
import textwrap


class Expand(ast.NodeTransformer):
    def __init__(self):
        self.pairs=set();self.maximum=0;self.next=0

    def temporary(self):
        name=f'_pair_temp_{self.next}'
        self.next+=1;self.maximum=max(self.maximum,self.next)
        return ast.Name(id=name,ctx=ast.Load())

    def is_pair(self,node):
        return (isinstance(node,ast.Name) and node.id.startswith('w_')
                or isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in {'P','dot_pair','cert_dot_pair','root_pair'}
                or isinstance(node,ast.BinOp) and (self.is_pair(node.left) or self.is_pair(node.right))
                or isinstance(node,ast.UnaryOp) and self.is_pair(node.operand))

    def element(self,node,index):
        return ast.Subscript(value=node,slice=ast.Constant(index),ctx=ast.Load())

    def call(self,name,args):
        return ast.Expr(value=ast.Call(func=ast.Name(id=name,ctx=ast.Load()),args=args,keywords=[]))

    def pair(self,node,out=None):
        if isinstance(node,ast.Name) and node.id.startswith('w_'):
            self.pairs.add(node.id)
            if out is None:return [],node
            return [self.call('_pair_pack',[out,self.element(node,0),self.element(node,1)])],out
        out=out or self.temporary()
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
            name=node.func.id
            if name=='P':
                stmts=[];args=[]
                for a in node.args:
                    before,x=self.scalar(a);stmts+=before;args.append(x)
                return stmts+[self.call('_pair_pack',[out]+args)],out
            if name in {'dot_pair','cert_dot_pair'}:
                return [self.call('_'+name,[out]+node.args)],out
            if name=='root_pair':
                before,a=self.pair(node.args[0])
                return before+[self.call('_root_pair',[out,ast.Name(id='pool',ctx=ast.Load()),a])],out
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):
            before,a=self.pair(node.operand)
            return before+[self.call('_pair_pack',[out,ast.UnaryOp(op=ast.USub(),operand=self.element(a,0)),ast.UnaryOp(op=ast.USub(),operand=self.element(a,1))])],out
        if isinstance(node,ast.BinOp):
            left,a=self.pair(node.left) if self.is_pair(node.left) else self.pair(ast.Call(func=ast.Name(id='P',ctx=ast.Load()),args=[node.left,ast.Constant(0.0)],keywords=[]))
            right,b=self.pair(node.right) if self.is_pair(node.right) else self.pair(ast.Call(func=ast.Name(id='P',ctx=ast.Load()),args=[node.right,ast.Constant(0.0)],keywords=[]))
            name={ast.Add:'plus',ast.Sub:'minus',ast.Mult:'times',ast.Div:'divide'}[type(node.op)]
            return left+right+[self.call('_pair_'+name,[out,a,b])],out
        raise ValueError('unsupported pair expression: '+ast.unparse(node))

    def scalar(self,node):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='T' and node.func.attr=='abs':
            before,x=self.scalar(node.args[0])
            return before,ast.Call(func=ast.Attribute(value=ast.Name(id='T',ctx=ast.Load()),attr='max',ctx=ast.Load()),args=[x,ast.UnaryOp(op=ast.USub(),operand=x)],keywords=[])
        if isinstance(node,ast.Attribute) and self.is_pair(node.value) and node.attr in {'h','l'}:
            before,p=self.pair(node.value)
            return before,self.element(p,0 if node.attr=='h' else 1)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):
            if node.func.id=='value':
                before,p=self.pair(node.args[0]);return before,ast.BinOp(left=self.element(p,0),op=ast.Add(),right=self.element(p,1))
            if node.func.id=='root':
                before,x=self.scalar(node.args[0]);out=self.temporary()
                return before+[self.call('_root',[out,ast.Name(id='pool',ctx=ast.Load()),x])],self.element(out,0)
        before=[]
        for field,child in ast.iter_fields(node):
            if isinstance(child,ast.expr):
                stmts,new=self.scalar(child);before+=stmts;setattr(node,field,new)
            elif isinstance(child,list):
                result=[]
                for x in child:
                    if isinstance(x,ast.expr):
                        stmts,new=self.scalar(x);before+=stmts;result.append(new)
                    else:result.append(x)
                setattr(node,field,result)
        return before,node

    def visit_Assign(self,node):
        self.next=0
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id.startswith('w_'):
            target=node.targets[0];self.pairs.add(target.id)
            stmts,_=self.pair(node.value,ast.Name(id=target.id,ctx=ast.Load()))
            return stmts
        before,node.value=self.scalar(node.value)
        return before+[node]

    def visit_Expr(self,node):
        self.next=0;before,node.value=self.scalar(node.value)
        return before+[node]

    def visit_AugAssign(self,node):
        self.next=0;before,node.value=self.scalar(node.value)
        return before+[node]

    def visit_If(self,node):
        self.next=0;before,node.test=self.scalar(node.test)
        node.body=self.block(node.body);node.orelse=self.block(node.orelse)
        return before+[node]

    def visit_For(self,node):
        node.body=self.block(node.body);node.orelse=self.block(node.orelse)
        return node

    def block(self,nodes):
        result=[]
        for node in nodes:
            new=self.visit(node)
            result.extend(new if isinstance(new,list) else [new])
        return result


def pair_macro(function):
    import tilelang.language as T
    from code_dsl.op_kernel import dd
    from code_dsl.op_kernel.intrinsics import emit
    tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
    definition=tree.body[0];definition.decorator_list=[]
    lower=Expand();definition.body=lower.block(definition.body)
    names=sorted(lower.pairs)+[f'_pair_temp_{i}' for i in range(lower.maximum)]
    allocations=ast.parse('\n'.join(f"{name}=T.alloc_ub((8,),'float32')" for name in names)).body
    definition.body=allocations+definition.body
    ast.fix_missing_locations(tree)
    source=ast.unparse(tree)+'\n'
    filename='<pinv-pair-dsl:'+function.__name__+'>'
    linecache.cache[filename]=(len(source),None,source.splitlines(True),filename)
    scope=dict(function.__globals__)
    scope.update(inspect.getclosurevars(function).nonlocals)
    scope.update(T=T,_pair_pack=dd.pack,_pair_plus=dd.plus,_pair_minus=dd.minus,_pair_times=dd.times,_pair_divide=dd.divide)
    scope['_dot_pair']=lambda out,pool,*args:emit('pv_pair_dot',out.access_ptr('w'),pool.access_ptr('rw'),*args)
    scope['_cert_dot_pair']=lambda out,pool,*args:emit('pv_pair_cert_dot',out.access_ptr('w'),pool.access_ptr('rw'),*args)
    scope['_root_pair']=lambda out,pool,a:emit('pv_pair_root',out.access_ptr('w'),a.access_ptr('r'),pool.access_ptr('rw'))
    scope['_root']=lambda out,pool,x:emit('pv_root',out.access_ptr('w'),pool.access_ptr('rw'),x)
    exec(compile(source,filename,'exec'),scope)
    result=T.macro(scope[function.__name__])
    result.expanded_source=source
    return result
