from .registration import register, make

register(
    id='DQN-v0',
    entry_point='agents.agent_vault:DQN'
)

register(
    id='DDPG-v0',
    entry_point='agents.agent_vault:DDPG'
)

register(
    id='DDDQN-v0',
    entry_point='agents.agent_vault:DDDQN'
)
