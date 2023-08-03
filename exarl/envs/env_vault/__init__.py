from exarl.utils.globals import ExaGlobals
try:
    env = ExaGlobals.lookup_params('env')
except:
    env = None

if env == 'ExaCartPoleStatic-v0':
    from exarl.envs.env_vault.ExaCartpoleStatic import ExaCartpoleStatic
if env == 'ExaMountainCarContinuous-v0':
    from exarl.envs.env_vault.ExaMountainCarContinuous import ExaMountainCarContinuous
elif env == 'ExaCH-v0':
    from exarl.envs.env_vault.ExaCH import CahnHilliardEnv
elif env == 'ExaCovid-v0':
    from exarl.envs.env_vault.ExaCOVID import ExaCOVID
elif env == 'ExaBoosterDiscrete-v0':
    from exarl.envs.env_vault.ExaBoosterDiscrete import ExaBooster_v1 as ExaBooster
elif env == 'ExaBoosterNew-v0':
    from exarl.envs.env_vault.ExaBoosterNew import ExaBooster_v2 as ExaBooster
elif env == 'ExaWaterClusterDiscrete-v0':
    from exarl.envs.env_vault.ExaWaterClusterDiscrete import ExaWaterClusterDiscrete
elif env == 'Hadrec-v0':
    from exarl.envs.env_vault.HadrecWrapper import HadrecWrapper
elif env == 'Bsuite-v0':
    from exarl.envs.env_vault.BsuiteWrapper import BsuiteWrapper
elif env == 'ExaExaalt-v0':
    from exarl.envs.env_vault.ExaExaalt import ExaExaalt
elif env == 'ExaExaaltBayes-v0':
    from exarl.envs.env_vault.ExaExaaltBayes import ExaExaaltBayes
elif env == 'ExaExaaltBayes-v1':
    from exarl.envs.env_vault.ExaExaaltBayes1 import ExaExaaltBayes1
elif env == 'ExaExaaltBayesRL-v0':
    from exarl.envs.env_vault.ExaExaaltBayesRL import ExaExaaltBayesRL
elif env == 'ExaExaaltVE-v0':
    from exarl.envs.env_vault.ExaExaaltVE import ExaExaaltVE
elif env == 'ExaExaaltGraph-v0':
    from exarl.envs.env_vault.ExaExaaltGraph import ExaExaaltGraph
elif env == 'ExaExaaltGraph-v1':
    from exarl.envs.env_vault.ExaExaaltGraphRL import ExaExaaltGraphRL
elif env == 'ExaExaaltGraph-v2':
    from exarl.envs.env_vault.ExaExaaltGraphRLTask import ExaExaaltGraphRLTask
elif env == 'ExaExaaltGraph-v3':
    from exarl.envs.env_vault.ExaExaaltGraphRLSpace import ExaExaaltGraphRLSpace
elif env == 'ExaPendulum-v0':
    from exarl.envs.env_vault.ExaPendulum import ExaPendulum
elif env == 'ExaPendulum-v1':
    from exarl.envs.env_vault.ExaPendulumTuple import ExaPendulumTuple