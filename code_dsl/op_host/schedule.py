"""Runtime launch geometry for independent compact jobs."""

def launch_blocks(plan,batch):
    if plan.strategy=='local' and plan.fields['cols']<=64 and plan.fields['cols']%8==0:
        return min(plan.aiv,batch)
    if plan.strategy=='compact':
        return min(plan.aiv,max(batch,max(1,plan.aiv//4)))
    return plan.blocks
