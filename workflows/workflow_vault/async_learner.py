import time
import csv
import numpy as np
from random import randint
import exarl as erl
from utils.introspect import ib
from utils.profile import *
import utils.log as log
import utils.candleDriver as cd

logger = log.setup_logger(__name__, cd.run_params["log_level"])

from exarl.comm_base import ExaComm


class ASYNC(erl.ExaWorkflow):
    def __init__(self):
        print("Class ASYNC learner")

    @PROFILE
    def run(self, workflow):

        # MPI communicators
        agent_comm = ExaComm.agent_comm
        env_comm = ExaComm.env_comm

        # Set target model
        target_weights = None
        if mpi_settings.is_learner():
            workflow.agent.set_learner()
            target_weights = workflow.agent.get_weights()

        # Only agent_comm processes will run this try block
        try:
            # Send and set weights to all other agents
            current_weights = agent_comm.bcast(target_weights, root=0)
            workflow.agent.set_weights(current_weights)
        except:
            logger.debug('Does not contain an agent')

        # Variables for all
        episode = 0
        episode_done = 0
        episode_interim = 0

        # Round-Robin Scheduler
        if ExaComm.is_learner():
            start = agent_comm.time()
            agent_comm.time()
            worker_episodes = np.arange(1, agent_comm.size)
            logger.debug("worker_episodes:{}".format(worker_episodes))

            logger.info("Initializing ...\n")
            for s in range(1, agent_comm.size):
                # Send target weights
                rank0_epsilon = workflow.agent.epsilon
                target_weights = workflow.agent.get_weights()
                episode = worker_episodes[s - 1]
                agent_comm.send([episode, rank0_epsilon, target_weights], s)

            init_nepisodes = episode
            logger.debug("init_nepisodes:{}".format(init_nepisodes))

            logger.debug("Continuing ...\n")
            while episode_done < workflow.nepisodes:
                # print("Running scheduler/learner episode: {}".format(episode))

                # Receive the rank of the worker ready for more work
                recv_data = [0, 0, shape, 0] if hasShape else None
                recv_data = agent_comm.recv(recv_data)
                whofrom = recv_data[0]
                step = recv_data[1]
                batch = recv_data[2]
                policy_type = recv_data[3]
                done = recv_data[4]
                logger.debug('step:{}'.format(step))
                logger.debug('done:{}'.format(done))
                # Train
                workflow.agent.train(batch)
                # TODO: Double check if this is already in the DQN code
                workflow.agent.target_train()
                if policy_type == 0:
                    workflow.agent.epsilon_adj()
                epsilon = workflow.agent.epsilon

                # Send target weights
                logger.debug('rank0_epsilon:{}'.format(epsilon))

                target_weights = workflow.agent.get_weights()

                # Increment episode when starting
                if step == 0:
                    episode += 1
                    logger.debug("if episode:{}".format(episode))

                # Increment the number of completed episodes
                if done:
                    episode_done += 1
                    latest_episode = worker_episodes.max()
                    worker_episodes[whofrom - 1] = latest_episode + 1
                    logger.debug("episode_done:{}".format(episode_done))
                    ib.update("Async_Learner_Episode", 1)

                agent_comm.send([worker_episodes[whofrom - 1],
                                 epsilon, target_weights], dest=whofrom)

            logger.info("Finishing up ...\n")
            episode = -1
            for s in range(1, agent_comm.size):
                recv_data = [0, 0, shape, 0] if hasShape else None
                recv_data = agent_comm.recv(recv_data)
                whofrom = recv_data[0]
                step = recv_data[1]
                batch = recv_data[2]
                epsilon = recv_data[3]
                done = recv_data[4]
                logger.debug('step:{}'.format(step))
                logger.debug('done:{}'.format(done))
                # Train
                workflow.agent.train(batch)
                # TODO: Double check if this is already in the DQN code
                workflow.agent.target_train()
                agent_comm.send([episode, 0, 0], dest=s)

            logger.info("Learner time: {}".format(agent_comm.time() - start))

        else:
            if ExaComm.is_actor():
                # Setup logger
                filename_prefix = 'ExaLearner_' + 'Episodes%s_Steps%s_Rank%s_memory_v1' \
                    % (str(workflow.nepisodes), str(workflow.nsteps), str(agent_comm.rank))
                train_file = open(workflow.results_dir + '/' +
                                  filename_prefix + ".log", 'w')
                train_writer = csv.writer(train_file, delimiter=" ")

            start = env_comm.time()
            while episode != -1:
                # Add start jitter to stagger the jobs [ 1-50 milliseconds]
                # time.sleep(randint(0, 50) / 1000)
                # Reset variables each episode
                workflow.env.seed(0)
                # TODO: optimize some of these variables out for env processes
                current_state = workflow.env.reset()
                total_reward = 0
                steps = 0
                action = 0

                # Steps in an episode
                while steps < workflow.nsteps:
                    logger.debug('ASYNC::run() agent_comm.rank{}; step({} of {})'
                                 .format(agent_comm.rank, steps, (workflow.nsteps - 1)))
                    if mpi_settings.is_actor():
                        # Receive target weights
                        recv_data = [0, 0, target_weights]
                        recv_data = agent_comm.recv(recv_data, source=0)
                        # Update episode while beginning a new one i.e. step = 0
                        if steps == 0:
                            episode = recv_data[0]
                        # This variable is used for kill check
                        episode_interim = recv_data[0]

                    # Broadcast episode within env_comm
                    episode_interim = env_comm.bcast(episode_interim, 0)

                    if episode_interim == -1:
                        episode = -1
                        if ExaComm.is_actor():
                            logger.info(
                                "Rank[%s] - Episode/Step:%s/%s"
                                % (str(agent_comm.rank), str(episode), str(steps))
                            )
                        break

                    if mpi_settings.is_actor():
                        workflow.agent.epsilon = recv_data[1]
                        workflow.agent.set_weights(recv_data[2])

                        if workflow.action_type == 'fixed':
                            action, policy_type = 0, -11
                        else:
                            action, policy_type = workflow.agent.action(
                                current_state)

                    next_state, reward, done, _ = workflow.env.step(action)

                    if ExaComm.is_actor():
                        total_reward += reward
                        memory = (
                            current_state,
                            action,
                            reward,
                            next_state,
                            done,
                            total_reward,
                        )

                        workflow.agent.remember(
                            memory[0], memory[1], memory[2], memory[3], memory[4]
                        )

                        batch = next(workflow.agent.generate_data())
                        ib.update("Async_Env_Generate_Data", 1)

                        # batch_data = []
                        workflow.agent.remember(
                            memory[0], memory[1], memory[2], memory[3], memory[4])

                        batch_data = next(workflow.agent.generate_data())
                        logger.info(
                            "Rank[{}] - Generated data: {}".format(
                                agent_comm.rank, len(batch[0])
                            )
                        )
                        logger.info(
                            'Rank[{}] - Memories: {}'.format(agent_comm.rank, len(workflow.agent.memory)))

                    if steps >= workflow.nsteps - 1:
                        done = True

                    if ExaComm.is_actor():
                        # Send batched memories
                        pack = not hasShape
                        agent_comm.send(
                            [agent_comm.rank, steps, batch_data, policy_type, done], dest=0)

                        logger.info(
                            'Rank[%s] - Episode/Step/Status:%s/%s/%s' % (str(agent_comm.rank), str(episode), str(steps), str(done)))

                        train_writer.writerow([time.time(), current_state, action, reward, next_state, total_reward,
                                               done, episode, steps, policy_type, workflow.agent.epsilon])
                        train_file.flush()

                    # Update state and step
                    current_state = next_state
                    steps += 1

                    # Broadcast done
                    done = env_comm.bcast(done, 0)
                    # Break for loop if done
                    if done:
                        break
            ib.update("Async_Env_Episode", 1)
            logger.info("Worker time = {}".format(env_comm.time() - start))
            if ExaComm.is_actor():
                train_file.close()

        if mpi_settings.is_actor():
            logger.info(f'Agent[{agent_comm.rank}] timing info:\n')
            workflow.agent.print_timers()
