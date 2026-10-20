"""Runtime launch geometry for independent compact jobs."""

def launch_blocks(plan,batch):
    if plan.strategy=='compact':
        return min(plan.aiv,max(batch,max(1,plan.aiv//4)))
    return plan.blocks
