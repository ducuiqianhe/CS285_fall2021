import numpy as np

from .base_agent import BaseAgent
from cs285.policies.MLP_policy import MLPPolicyPG
from cs285.infrastructure.replay_buffer import ReplayBuffer

from cs285.infrastructure import utils


class PGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(PGAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']
        self.gae_lambda = self.agent_params['gae_lambda']

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, observations, actions, rewards_list, next_observations, terminals):

        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """

        # Done: update the PG actor/policy using the given batch of data, and
        # return the train_log obtained from updating the policy

        # HINT1: use helper functions to compute qvals and advantages
        # HINT2: look at the MLPPolicyPG class for how to update the policy
        # and obtain a train_log

        # step 1: calculate q values of each (s_t, a_t) point,
        # using rewards from that full rollout of length T: (r_0, ..., r_t, ..., r_{T-1})
        q_values = self.calculate_q_vals(rewards_list)

        # step 2: calculate advantages that correspond to each (s_t, a_t) point
        advantages = self.estimate_advantage(observations, rewards_list, q_values, terminals)

        # step 3:
        # pass the calculated values above into the actor/policy's update,
        # which will perform the actual PG update step
        logs = self.actor.update(observations, actions, advantages, q_values)

        return logs

    def calculate_q_vals(self, rewards_list):

        """
            Monte Carlo estimation of the Q function.
        """

        # Done: return the estimated qvals based on the given rewards, using
        # either the full trajectory-based estimator or the reward-to-go
        # estimator

        # HINT1: rewards_list is a list of lists of rewards. Each inner list
        # is a list of rewards for a single trajectory.
        # HINT2: use the helper functions self._discounted_return and
        # self._discounted_cumsum (you will need to implement these). These
        # functions should only take in a single list for a single trajectory.

        if not self.reward_to_go:
            q_vals = np.concatenate([self._discounted_return(r) for r in rewards_list])
        else:
            q_vals = np.concatenate([self._discounted_cumsum(r) for r in rewards_list])

        return q_vals

    def estimate_advantage(self, obs, rews_list, q_values, terminals):

        """
            Computes advantages by (possibly) using GAE, or subtracting a baseline from the estimated Q values
        """

        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the value function
        if self.nn_baseline:
            values_unnormalized = self.actor.run_baseline_prediction(obs)
            ## ensure that the value predictions and q_values have the same dimensionality
            ## to prevent silent broadcasting errors
            assert values_unnormalized.ndim == q_values.ndim
            ## Done: values were trained with standardized q_values, so ensure
            ## that the predictions have the same mean and standard deviation as
            ## the current batch of q_values
            values = utils.unnormalize(values_unnormalized, np.mean(q_values), np.std(q_values))

            if self.gae_lambda is not None:
                values = np.append(values, [0])

                ## combine rews_list into a single array
                rews = np.concatenate(rews_list)

                ## create empty numpy array to populate with GAE advantage
                ## estimates, with dummy T+1 value for simpler recursive calculation
                batch_size = obs.shape[0]
                advantages = np.zeros(batch_size + 1)

                for i in reversed(range(batch_size)):
                    ## Done: recursively compute advantage estimates starting from
                        ## timestep T.
                    ## HINT 1: use terminals to handle edge cases. terminals[i]
                        ## is 1 if the state is the last in its trajectory, and
                        ## 0 otherwise.
                    ## HINT 2: self.gae_lambda is the lambda value in the
                        ## GAE formula
                    if terminals[i]:
                        advantages[i] = rews[i] - values[i]
                    else:
                        td_error = self.gamma * values[i + 1] - values[i] + rews[i]
                        advantages[i] = advantages[i] * self.gamma * self.gae_lambda + td_error

                advantages = advantages[:-1]

            else:
                ## Done: compute advantage estimates using q_values, and values as baselines
                advantages = q_values - values

        # Else, just set the advantage to [Q]
        else:
            advantages = q_values.copy()

        # Normalize the resulting advantages
        if self.standardize_advantages:
            ## Done: standardize the advantages to have a mean of zero
            ## and a standard deviation of one
            advantages = utils.normalize(advantages, np.mean(advantages), np.std(advantages))

        return advantages

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _discounted_return(self, rewards):
        """
            Helper function

            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

            Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """

        # Done: create list_of_discounted_returns
        list_of_discounted_returns = np.array([np.sum([self.gamma ** t * r_t for t, r_t in zip(range(len(rewards)), rewards)])] * len(rewards))

        return list_of_discounted_returns

    def _discounted_cumsum(self, rewards):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """

        length = len(rewards)
        list_of_discounted_cumsum = []
        for t in range(length):
            print(np.asarray(rewards[t:]).shape)
            sum = np.sum((self.gamma ** np.arange(length - t)) * np.asarray(rewards[t:]))
            list_of_discounted_cumsum.append(sum)

        return np.array(list_of_discounted_cumsum)
