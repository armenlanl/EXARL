from gym.envs import registration
from gym.envs.registration import register
from exarl.utils.globals import ExaGlobals
try:
    env = ExaGlobals.lookup_params('env')
except:
    env = None

if env == 'ExaCH-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:CahnHilliardEnv',
    )

elif env == 'ExaCartPoleStatic-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaCartpoleStatic'
    )

elif env == 'ExaMountainCarContinuous-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaMountainCarContinuous'
    )

elif env == 'ExaPendulum-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaPendulum'
    )

elif env == 'ExaCovid-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaCOVID'
    )

elif env == 'ExaBoosterDiscrete-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaBooster'
    )

elif env == 'ExaBoosterNew-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaBooster'
    )

elif env == 'ExaWaterClusterDiscrete-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaWaterClusterDiscrete'
    )

elif env == 'ExaExaalt-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaExaalt'
    )

elif env == 'ExaExaaltBayes-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaExaaltBayes'
    )

elif env == 'ExaExaaltBayes-v1':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaExaaltBayes1'
    )

elif env == 'ExaExaaltBayesRL-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaExaaltBayesRL'
    )

elif env == 'ExaExaaltVE-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaExaaltVE'
    )

elif env == 'ExaExaaltGraph-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaExaaltGraph'
    )

elif env == 'ExaExaaltGraph-v1':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaExaaltGraphRL'
    )

elif env == 'ExaExaaltGraph-v2':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaExaaltGraphRLTask'
    )

elif env == 'ExaExaaltGraph-v3':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:ExaExaaltGraphRLSpace'
    )
   
elif env == 'Hadrec-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:HadrecWrapper'
    )

elif env == 'Bsuite-v0':
    register(
        id=env,
        entry_point='exarl.envs.env_vault:BsuiteWrapper'
    )
