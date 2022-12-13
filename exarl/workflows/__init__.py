from exarl.workflows.registration import register, make

register(
    id='sync',
    entry_point='exarl.workflows.workflow_vault:SYNC'
)

register(
    id='sync_ars',
    entry_point='exarl.workflows.workflow_vault:SYNC_ARS'
)

register(
    id='async_ars',
    entry_point='exarl.workflows.workflow_vault:ASYNC_ARS'
)

register(
    id='async',
    entry_point='exarl.workflows.workflow_vault:ASYNC'
)

register(
    id='rma',
    entry_point='exarl.workflows.workflow_vault:RMA'
)

register(
    id='tester',
    entry_point='exarl.workflows.workflow_vault:TESTER'
)

register(
    id='random',
    entry_point='exarl.workflows.workflow_vault:RANDOM'
)
