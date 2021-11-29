import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from RSA_communication_agents import RSAListener0, RSASpeaker0, RSAListener1, RSASpeaker1


def get_test_input_policy_single_agent(n, agent, ablation=False):
    """ For the single agent setting.
        Get the listeners' policies, so probabilities of selecting every possible state,
        when being presented with an input message that was withheld from training.
        The policies are calculated for the case that one message was excluded from training as well as
        for both messages separately in case two messages were excluded from training.
        :param n:           number of states and messages (in total)
        :param agent:       'L0' literal listener
                            'L1' pragmatic listener
        :param ablation:    indicates whether calculations are for the standard evaluation or the ablation test; for
                            the ablation study the pragmatic reasoning abilities of the agents are changed at test time.
        :return [p_missing, p_missing1, p_missing2]:    policies for all agents for the case that one message was
                                                        excluded from training (p_missing), or that two messages were
                                                        excluded from training (p_missing1, p_missing2)
    """

    n_messages = 2*n
    n_states = n

    agent_body = agent  # agent body determines whether the lexica of literal or pragmatic agents are used
    agent_head = agent  # agent head determined whether the literal or pragmatic reasoning is used at test time

    # ME: missing examples
    for ME in [1, 2]:

        # set file path
        if agent == 'L0':
            filename = ('data/bilingual/L0/' + str(n) + '_states/' + agent_body + '_' + str(ME) + 'missing_')
        elif agent == 'L1':
            filename = ('data/bilingual/L1/' + str(n) + '_states/' + agent_body + '_' + str(ME) + 'missing_5.0alpha_')

        # load final lexicon for every trained agent
        lexica_all = []
        for run in range(1, 101):
            lexica = np.load(filename + 'lexicon_run' + str(run) + '.npy')
            lexica_all.append(lexica[-1])

        # generate withheld / test examples
        if ME == 1:
            listener_input = np.zeros((1, n_messages, n_states), dtype=np.float32)
            listener_input[0, n - 1, :] = np.ones((1, n_states), dtype=np.float32)
            p_missing = []
        elif ME == 2:
            listener_input1 = np.zeros((1, n_messages, n_states), dtype=np.float32)
            listener_input2 = np.zeros((1, n_messages, n_states), dtype=np.float32)
            listener_input1[0, n - 1, :] = np.ones((1, n_states), dtype=np.float32)
            listener_input2[0, 0, :] = np.ones((1, n_states), dtype=np.float32)
            p_missing1 = []
            p_missing2 = []

        # determine whether pragmatic or literal reasoning is used at test time
        if ablation and agent_body == 'L0':
            agent_head = 'L1'
        if ablation and agent_body == 'L1':
            agent_head = 'L0'

        # iterate over the listeners' lexica and calculate the policy for an agent with that lexion
        for l in lexica_all:
            if agent_head == 'L0':
                listener = RSAListener0(n_states, n_messages, l)
            elif agent_head == 'L1':
                listener = RSAListener1(n_states, n_messages, l, alpha=5.)
            # calculate the policies for the state that was excluded from training in case one state was excluded
            if ME == 1:
                policy, _ = listener.get_states(listener_input)
                p_missing.append(np.squeeze(policy[:]))
            # calculate the policies for the states that were excluded from training separately,
            # in case two states were excluded
            elif ME == 2:
                policy1, _ = listener.get_states(listener_input1)
                policy2, _ = listener.get_states(listener_input2)
                p_missing1.append(np.squeeze(policy1[:]))
                p_missing2.append(np.squeeze(policy2[:]))

    # normalize the policies such that they are actual probabilities
    return_values = []
    for p in [p_missing, p_missing1, p_missing2]:
        p = np.squeeze(np.array(p))
        normalized_p = p / np.expand_dims(np.sum(p, axis=1), axis=1)
        return_values.append(normalized_p)

    return return_values


def get_test_input_policy_two_agents(n, agent, ablation=False):
    """ For the two agent setting.
        Get the listeners' policy, so probabilities of selecting every possible state,
        when the speaker is presented with an input state that was withheld from training.
        The policies are calculated for the case that one state was excluded from training as well as
        for both states separately in case two states were excluded from training.
        :param n:       number of states and messages (in total)
        :param agent:   'L0_S0' literal speaker-listener combination
                        'L1_S1' pragmatic speaker-listener combination
        :param ablation:    indicates whether calculations are for the standard evaluation or the ablation test; for
                            the ablation study the pragmatic reasoning abilities of the agents are changed at test time.
        :return [p_missing, p_missing1, p_missing2]:    policies for all agents for the case that one state was
                                                        excluded from training (p_missing), or that two states were
                                                        excluded from training (p_missing1, p_missing2)
    """

    agent_body = agent  # agent body determines whether the lexica of literal or pragmatic agents are used
    agent_head = agent  # agent head determined whether the literal or pragmatic reasoning is used at test time

    # ME: missing examples
    for ME in [1, 2]:

        # set file path
        if agent == 'L0_S0':
            filename = 'data/communication/' + str(n) + '_states/' + agent_body + '_' + str(ME) + 'missing_'
        elif agent == 'L1_S1':
            filename = 'data/communication/' + str(n) + '_states/' + agent_body + '_' + str(ME) + 'missing_5.0alpha_'

        # load final lexica for all speakers and listeners
        lexica_all_L = []
        lexica_all_S = []
        for run in range(1, 101):
            lexica_L = np.load(filename + 'lexicon_L_run' + str(run) + '.npy')
            lexica_S = np.load(filename + 'lexicon_S_run' + str(run) + '.npy')
            lexica_all_L.append(lexica_L[-1])
            lexica_all_S.append(lexica_S[-1])

        # generate withheld / test examples
        if ME == 1:
            speaker_input = np.zeros((1, n, n), dtype=np.float32)
            speaker_input[0, n - 1, :] = np.ones((1, n), dtype=np.float32)
            p_missing = []
        elif ME == 2:
            speaker_input1 = np.zeros((1, n, n), dtype=np.float32)
            speaker_input2 = np.zeros((1, n, n), dtype=np.float32)
            speaker_input1[0, n - 1, :] = np.ones((1, n), dtype=np.float32)
            speaker_input2[0, 0, :] = np.ones((1, n), dtype=np.float32)
            p_missing1 = []
            p_missing2 = []

        # determine whether pragmatic or literal reasoning is used at test time
        if ablation and agent_body == 'L0_S0':
            agent_head = 'L1_S1'
        if ablation and agent_body == 'L1_S1':
            agent_head = 'L0_S0'

        # Calculating the policy of the listener when the speaker is presented with one of the test states is
        # done in the following way:
        # 1) calculate the speaker's policy for that input state (conditional distribution over messages)
        # 2) calculate the listener's s policy for any input message (conditional distributions over states)
        # 3) multiply the speaker's probability of selecting a state given a message with the speaker's
        #    probability of producing that message
        # 4) sum up all the probabilities from 3) to obtain the expected probability of the listener for selecting
        #    the different states given the speaker is presented with a test input state.

        # iterate over all speaker-listener combinations
        for l_S, l_L in zip(lexica_all_S, lexica_all_L):
            if agent_head == 'L0_S0':
                speaker = RSASpeaker0(n, n, l_S)
                listener = RSAListener0(n, n, l_L)
            elif agent_head == 'L1_S1':
                speaker = RSASpeaker1(n, n, l_S, alpha=5.)
                listener = RSAListener1(n, n, l_L, alpha=5.)

            # calculate the policies for the state that was excluded from training in case one state was excluded
            if ME == 1:
                # do step 1)
                policy, _ = speaker.get_messages(speaker_input)
                policy = np.squeeze(policy[:])
                policy = policy / np.sum(policy)
                # do step 2)
                selections = np.zeros((n, n))
                for input_idx in range(n):
                    listener_input = np.zeros((1, n, n))
                    listener_input[0, input_idx, :] = np.ones((1, n))
                    selection, _ = listener.get_states(listener_input)
                    selection = np.squeeze(selection[:])
                    selection = selection / np.sum(selection)
                    # do step 3)
                    selections[input_idx, :] = policy[input_idx] * selection
                # do step 4)
                p_missing.append(np.sum(selections, axis=0))

            # calculate the policies for the states that were excluded from training separately,
            # in case two states were excluded
            elif ME == 2:
                # do step 1)
                policy1, _ = speaker.get_messages(speaker_input1)
                policy2, _ = speaker.get_messages(speaker_input2)
                for p, policy in enumerate([policy1, policy2]):
                    policy = np.squeeze(policy[:])
                    policy = policy / np.sum(policy)
                    # do step 2)
                    selections = np.zeros((n, n))
                    for input_idx in range(n):
                        listener_input = np.zeros((1, n, n))
                        listener_input[0, input_idx, :] = np.ones((1, n))
                        selection, _ = listener.get_states(listener_input)
                        selection = np.squeeze(selection[:])
                        selection = selection / np.sum(selection)
                        # do step 3)
                        selections[input_idx, :] = policy[input_idx] * selection
                    # do step 4)
                    [p_missing1, p_missing2][p].append(np.sum(selections, axis=0))

    # the values are already normalized and can be returned as probabilities
    return np.array(p_missing), np.array(p_missing1), np.array(p_missing2)


def me_index(agent, ablation=False):
    """ Calculate the ME index as defined in the paper for single or two agent setting.
    :param agent:       'L0' literal listener (single agent setting)
                        'L1' pragmatic listener (single agent setting)
                        'L0_S0' literal speaker-listener combination (two agent setting)
                        'L1_S1' pragmatic speaker-listener combination (two agent setting)
    :param ablation:    indicates whether calculations are for the standard evaluation or the ablation test; for
                        the ablation study the pragmatic reasoning abilities of the agents are changed at test time.
    """
    # n: number of states and messages
    for i, n in enumerate([3, 10]):
        # calculate ME index for one input state/message excluded from training

        if agent == 'L0' or agent == 'L1':
            p_missing, p_missing1, p_missing2 = get_test_input_policy_single_agent(n, agent, ablation=ablation)
        elif agent == 'L0_S0' or agent == 'L1_S1':
            p_missing, p_missing1, p_missing2 = get_test_input_policy_two_agents(n, agent, ablation=ablation)

        me1 = (p_missing[:, -1] - 1 / n) / ((n - 1) / n)
        mean_me1 = np.mean(me1)
        std_me1 = np.std(me1)

        print('N=' + str(n), 'K=1:', round(mean_me1, 3), round(std_me1, 3))

        # calculate ME index for two input states/messages excluded from training
        me2 = ((p_missing1[:, -1] + p_missing2[:, -1] + p_missing1[:, 0] + p_missing2[:, 0]) / 2 - 2 / n) / (
                (n - 2) / n)

        mean_me2 = np.mean(me2)
        std_me2 = np.std(me2)

        print('N=' + str(n), 'K=2:', round(mean_me2, 3), round(std_me2, 3))


def plot_rewards(agent, n=3, n_epochs=20):
    """ Plots the average rewards over time for the 100 listeners or speaker-listener combinations trained in
        Experiment 1. The left plots is for the agents that were trained with one input message (single agent setting)
        or state (two agent setting) excluded from training (K=1). The right plot is for the agents that were
        trained with two input messages or states excluded from training (K=2). The average is shown in a thick line,
        the range from minimal to maximal values as shaded region.
        :param agent:       'L0' literal listener (single agent setting)
                            'L1' pragmatic listener (single agent setting)
                            'L0_S0' literal speaker-listener combination (two agent setting)
                            'L1_S1' pragmatic speaker-listener combination (two agent setting)
        :param n:           number of states, here 3 or 10
        :param n_epochs:    number of epochs you would like to plot
    """

    fig = plt.figure(figsize=(10, 4))

    # set file path
    if agent == 'L0':
        filename1 = ('data/bilingual/L0/' + str(n) + '_states/' + agent + '_1missing_rewards_run')
        filename2 = ('data/bilingual/L0/' + str(n) + '_states/' + agent + '_2missing_rewards_run')
    elif agent == 'L1':
        filename1 = ('data/bilingual/L1/' + str(n) + '_states/' + agent + '_1' + 'missing_5.0alpha_rewards_run')
        filename2 = ('data/bilingual/L1/' + str(n) + '_states/' + agent + '_2' + 'missing_5.0alpha_rewards_run')
    if agent == 'L0_S0':
        filename1 = 'data/communication/' + str(n) + '_states/' + agent + '_1missing_rewards_run'
        filename2 = 'data/communication/' + str(n) + '_states/' + agent + '_2missing_rewards_run'
    elif agent == 'L1_S1':
        filename1 = ('data/communication/' + str(n) + '_states/' + agent + '_1missing_5.0alpha_rewards_run')
        filename2 = ('data/communication/' + str(n) + '_states/' + agent + '_2missing_5.0alpha_rewards_run')

    rewards_all_ME1 = []
    rewards_all_ME2 = []

    # load rewards
    for run in range(1, 101):
        rewards1 = np.load(filename1 + str(run) + '.npy')
        rewards2 = np.load(filename2 + str(run) + '.npy')
        rewards_all_ME1.append(rewards1)
        rewards_all_ME2.append(rewards2)

    # plot the rewards
    for i, rewards_all in enumerate([rewards_all_ME1, rewards_all_ME2]):
        ME = i + 1
        rewards_all = np.array(rewards_all)

        rewards_mean = np.mean(rewards_all, axis=0)
        rewards_min = np.min(rewards_all, axis=0)
        rewards_max = np.max(rewards_all, axis=0)

        ax = fig.add_subplot(1, 2, i + 1)
        ax.plot(rewards_mean[:n_epochs], color='black')
        ax.plot(rewards_max[:n_epochs], color='red')
        ax.plot(rewards_min[:n_epochs], color='blue')
        ax.legend(['mean', 'max', 'min'])
        ax.fill_between(np.arange(n_epochs), rewards_mean[:n_epochs], rewards_max[:n_epochs],
                        color='red', alpha=0.3)
        ax.fill_between(np.arange(n_epochs), rewards_min[:n_epochs], rewards_mean[:n_epochs],
                        color='blue', alpha=0.3)

        plt.xlim([0, n_epochs + 1])
        plt.xlabel('epoch')
        plt.ylabel('mean reward')
        plt.title(str(ME) + ' missing')


def plot_lexica_single_agent(agent, n_array = [3, 10]):
    """ Plots the average lexicon of listeners at the end of training in the single agent setting (Experiment 1).

        Four plots are created for N in {3,10} and K in {1,2}, with N being the number of states and messages,
        K being the number of messages withheld from training.
        Note that plotting the average lexica only makes sense in the single agent setting as the state-message
        mapping can evolve differently between different speaker-listener pairs in the two agent setting.
        :param agent:   'L0' literal listener
                        'L1' pragmatic listener
    """

    fig = plt.figure(figsize=(20, 4))

    # n: number of states and messages
    for i, n in enumerate(n_array):

        # ME: missing examples
        for j, ME in enumerate([1, 2]):

            # set filepath
            if agent == 'L0':
                filename = 'data/bilingual/L0/' + str(n) + '_states/' + agent + '_' + str(ME) + 'missing_'
            else:
                filename = 'data/bilingual/L1/' + str(n) + '_states/' + agent + '_' + str(ME) + 'missing_5.0alpha_'

            # load the final lexica and rearrange them such that entries for the first state become equivalent
            # to the entries in the second to last state for K=2. This is necessary as when leaving out 2
            # states from training we left out the first and the last one, but changed it to the last two for
            # the paper for easier readability.
            lexica_all = []

            for run in range(100):
                lexica = np.load(filename + 'lexicon_run' + str(run + 1) + '.npy')
                rewards = np.load(filename + 'rewards_run' + str(run + 1) + '.npy')
                if np.sum(np.equal(rewards, 1)) > 0:
                    if ME == 1:
                        # load final lexicon
                        lexica_all.append(lexica[-1])
                    elif ME == 2:
                        lexicon_rearranged = np.copy(lexica[-1])
                        # and rearrange in case two states were excluded
                        lexicon_rearranged[:, n - 2] = lexica[-1][:, 0]
                        lexicon_rearranged[:, 0] = lexica[-1][:, n - 2]
                        lexicon_rearranged[0, :] = lexica[-1][n - 2, :]
                        lexicon_rearranged[n - 2, :] = lexica[-1][0, :]
                        copy_lexicon = np.copy(lexicon_rearranged)
                        lexicon_rearranged[0, 0] = copy_lexicon[0, n - 2]
                        lexicon_rearranged[0, n - 2] = copy_lexicon[0, 0]
                        lexicon_rearranged[n - 2, 0] = copy_lexicon[n - 2, n - 2]
                        lexicon_rearranged[n - 2, n - 2] = copy_lexicon[n - 2, 0]
                        lexicon_rearranged[:, n] = copy_lexicon[:, 2 * n - 2]
                        lexicon_rearranged[:, 2 * n - 2] = copy_lexicon[:, n]
                        lexica_all.append(lexicon_rearranged)

            lexica = np.array(lexica_all)
            mean = sum(lexica) / len(lexica)

            # plot average lexicon
            plt.subplot(1, 4, 2 * i + j + 1)
            im = plt.imshow(mean / np.max(mean))
            plt.xticks(range(2*n), [k + 1 for k in range(2*n)], fontsize=18)
            plt.yticks(range(n), [k + 1 for k in range(n)], fontsize=18)
            plt.ylabel('state', fontsize=20)
            plt.xlabel('message', fontsize=20)
            plt.title(str(n) + ' states, ' + str(ME) + ' missing \n ', fontsize=25)

    fig.subplots_adjust(bottom=0.0, top=1.0, left=0.05, right=2.0,
                        wspace=0.0, hspace=0.0)
    cb_ax = fig.add_axes([0.0, 0.0, 0.01, 1.0])
    colorbar = fig.colorbar(im, cax=cb_ax)
    colorbar.ax.tick_params(labelsize=18)

def save_all_lexica(n = 10):
    """ Plots the average reward and ME index over time for the single agent setting. The plots include the simulations
        with three and ten states, where one message was left out from training.

        :param ablation:    indicates whether calculations are for the standard evaluation or the ablation test; for
                            the ablation study the pragmatic reasoning abilities of the agents are changed at test time.
        :param n_array:     size 2 array of n values to plot for
        :param n_ranges:    size 2 array of ranges of epochs to be plotted
    """

    filename = ('data/bilingual/L1/' + str(n) + '_states/L1_1missing_5.0alpha_')

    # load lexica and rewards until the maximum epoch that should be plotted
    lexica_all = []
    for run in range(1, 101):
        lexica = np.load(filename + 'lexicon_run' + str(run) + '.npy')
        lexica_all.append(lexica)

    lexica_mean = np.mean(lexica_all, axis=0)

    for epoch in range(100):
        fig = plt.figure(figsize=(10, 6))
        plt.subplot(1,1,1)
        # plot average lexicon
        im = plt.imshow(lexica_mean[epoch] / np.max(lexica_mean[epoch]), vmin=0., vmax=1.)
        plt.xticks(range(2*n), [k + 1 for k in range(2*n)], fontsize=16)
        plt.yticks(range(n), [k + 1 for k in range(n)], fontsize=16)
        plt.ylabel('state', fontsize=20)
        plt.xlabel('message', fontsize=20)
        plt.title(str(n) + ' states, 1 missing \n ', fontsize=25)
        plt.gcf().text(0.87, 0.9, 'epoch ' + str(epoch), fontsize=16)

        fig.subplots_adjust(bottom=0.0, top=1.0, left=0.2, right=0.98,
                            wspace=0.0, hspace=0.0)
        cb_ax = fig.add_axes([0.02, 0.1, 0.01, 0.8])
        colorbar = fig.colorbar(im, cax=cb_ax)
        colorbar.ax.tick_params(labelsize=18)
        
        fig.savefig('plots/bilingual/L1/10 states/e_' + str(epoch) + '.png')
        plt.close(fig)

def bar_plot_me_bias(agent, ablation=False):
    """ Plots the listeners' average selection probability for all states given a novel example (that was not part of
        the training. For the single agent setting a novel example is a message that was not part of training, for the
        two agent setting a novel example is a state that was not part of training (and is presented to the speaker
        who produces a message based on which the listener makes a selection).
        Four plots are generated for the different combinations of three or ten states in total and 1 or 2 states being
        excluded from training.
        :param agent:       'L0' literal listener (single agent setting)
                            'L1' pragmatic listener (single agent setting)
                            'L0_S0' literal speaker-listener combination (two agent setting)
                            'L1_S1' pragmatic speaker-listener combination (two agent setting)
        :param ablation:    indicates whether calculations are for the standard evaluation or the ablation test; for
                            the ablation study the pragmatic reasoning abilities of the agents are changed at test time.
    """

    fig = plt.figure(figsize=(19, 3))

    # n: number of states and messages
    for i, n in enumerate([3, 10]):

        if agent == 'L0' or agent == 'L1':
            p_missing, p_missing1, p_missing2 = get_test_input_policy_single_agent(n, agent, ablation=ablation)
        elif agent == 'L0_S0' or agent == 'L1_S1':
            p_missing, p_missing1, p_missing2 = get_test_input_policy_two_agents(n, agent, ablation=ablation)

        # ME: missing examples
        for j, ME in enumerate([1, 2]):

            ax = fig.add_subplot(1, 4, 2 * i + j + 1)
            ax.set_axisbelow(True)

            # calculate mean and std if one input was excluded from training
            if ME == 1:
                mean_p_missing = np.mean(p_missing, axis=0)
                std_p_missing = np.std(p_missing, axis=0)
                ax.bar(range(1, n + 1), mean_p_missing, yerr=std_p_missing, color='blue', edgecolor='k',
                       width=0.6, capsize=3, alpha=1)

            # calculate mean and std if two inputs were excluded from training
            elif ME == 2:
                positions1 = np.arange(1, n + 1) - 0.17
                positions2 = np.arange(1, n + 1) + 0.17
                mean_p_missing1 = np.mean(p_missing1, axis=0)
                mean_p_missing2 = np.mean(p_missing2, axis=0)
                std_p_missing1 = np.std(p_missing1, axis=0)
                std_p_missing2 = np.std(p_missing2, axis=0)

                # rearrange the order in case two states were missing
                mean_p_missing1[[0, n - 2]] = mean_p_missing1[[n - 2, 0]]
                std_p_missing1[[0, n - 2]] = std_p_missing1[[n - 2, 0]]
                mean_p_missing2[[0, n - 2]] = mean_p_missing2[[n - 2, 0]]
                std_p_missing2[[0, n - 2]] = std_p_missing2[[n - 2, 0]]

                ax.bar(positions1, mean_p_missing1, yerr=std_p_missing1, color='blue',
                       edgecolor='k', width=0.3, capsize=3, alpha=1)
                ax.bar(positions2, mean_p_missing2, yerr=std_p_missing2, color='orangered',
                       edgecolor='k', width=0.3, capsize=3, alpha=1, )

                plt.legend(['message ' + str(n - 1), 'message ' + str(n)], fontsize=15)
                leg = ax.get_legend()
                leg.legendHandles[0].set_color('blue')
                leg.legendHandles[1].set_color('orangered')

            plt.ylim([-0.1, 1.1])
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            plt.xlabel('state', fontsize=20)
            if i == 0 and j == 0:
                plt.ylabel('selection probability', fontsize=20)
            plt.xticks(range(1, n + 1), [k for k in range(1, n + 1)], fontsize=18)
            plt.yticks(fontsize=18)
            plt.title(str(n) + ' states, ' + str(ME) + ' missing', fontsize=20)


def plot_rewards_plus_me_index_single_agent(ablation=False, n_array = [3, 10], n_ranges = [26, 41]):
    """ Plots the average reward and ME index over time for the single agent setting. The plots include the simulations
        with three and ten states, where one message was left out from training.

        :param ablation:    indicates whether calculations are for the standard evaluation or the ablation test; for
                            the ablation study the pragmatic reasoning abilities of the agents are changed at test time.
        :param n_array:     size 2 array of n values to plot for
        :param n_ranges:    size 2 array of ranges of epochs to be plotted
    """

    colors = [['red', 'red'], ['blue', 'blue']]

    fig = plt.figure(figsize=(15, 4))

    # n: number of states
    for n_index, n in enumerate(n_array):

        n_messages = 2*n
        n_states = n

        # which epochs should be plotted for three and ten states
        indices = [list(range(n_ranges[0])), list(range(n_ranges[1]))][n_index]
        # at what step size the error bars should be plotted for three and ten states
        error_step = np.floor_divide(n_ranges, 10)[n_index]

        ax = fig.add_subplot(1, 2, n_index + 1)

        for a, agent in enumerate(['L0', 'L1']):

            # determine which lexica to evaluate and which reasoning ability at test time
            agent_body = agent  # agent body determines whether the lexica of literal or pragmatic agents are used
            agent_head = agent  # agent head determined whether the literal or pragmatic reasoning is used at test time
            if ablation and agent_body == 'L0':
                agent_head = 'L1'
            elif ablation and agent_body == 'L1':
                agent_head = 'L0'

            # set file path
            if agent_body == 'L0':
                filename = ('data/bilingual/L0/' + str(n) + '_states/' + agent_body + '_1missing_')
            elif agent_body == 'L1':
                filename = ('data/bilingual/L1/' + str(n) + '_states/' + agent_body + '_1missing_5.0alpha_')

            # load lexica and rewards until the maximum epoch that should be plotted
            lexica_all = []
            rewards_all = []
            for run in range(1, 101):
                rewards = np.load(filename + 'rewards_run' + str(run) + '.npy')
                lexica = np.load(filename + 'lexicon_run' + str(run) + '.npy')
                lexica_all.append(lexica[indices])
                rewards_all.append(rewards[indices])

            # determine the test input as the example that was left out from training
            agent_input = np.zeros((1, n_messages, n_states), dtype=np.float32)
            agent_input[0, n - 1, :] = np.ones((1, n_states), dtype=np.float32)

            # re-sort rewards and indices such that the values of all agents are pooled together per time step
            # similarly calculate the ME index for every agent at every time step and pool per time step
            me_index_all = np.zeros((len(indices), 100))
            rewards_sorted = np.zeros((len(indices), 100))

            for idx in indices:
                me_index = np.zeros((100))  # unnormalized ME index

                for run in range(100):

                    rewards_sorted[idx, run] = rewards_all[run][idx]
                    lexicon = lexica_all[run][idx]

                    if agent_head == 'L0':
                        listener = RSAListener0(n_states, n_messages, lexicon)
                    elif agent_head == 'L1':
                        listener = RSAListener1(n_states, n_messages, lexicon, alpha=5.)
                    policy, _ = listener.get_states(agent_input)
                    policy = np.squeeze(policy[:])
                    policy = policy / np.sum(policy)
                    me_index[run] = policy[-1]

                # normalize the ME index for this agent and time step and append to the ME indices of the other
                # agents for that time step
                me_index_all[idx, :] = (me_index - (1 / n)) / ((n - 1) / n)

            # plot mean and stds for the rewards and ME indices over time
            mean_rewards = np.mean(rewards_sorted, axis=1)
            std_rewards = np.std(rewards_sorted, axis=1)
            mean_me_index = np.mean(me_index_all, axis=1)
            std_me_index = np.std(me_index_all, axis=1)

            ax.errorbar(indices, mean_rewards, yerr=std_rewards, errorevery=error_step, color=colors[a][0],
                        linewidth=3.0)
            ax.errorbar(indices, mean_me_index, yerr=std_me_index, errorevery=error_step, color=colors[a][1],
                        linewidth=3.0, linestyle='dashed')

            # if (n == 10):
            #     print(mean_me_index)

            plt.xticks(fontsize=20)
            plt.yticks([0, 1], fontsize=20)
            plt.xlabel('epoch', fontsize=25)
            plt.ylabel('mean reward /\n ME index', fontsize=25)

        if n == n_array[1]:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(['literal reward', 'literal $I_{ME}$', 'pragmatic reward', 'pragmatic $I_{ME}$'],
                      loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
        fig.tight_layout()

def plot_rewards_plus_me_index_pragmatic_agent(ablation=False, n_array = [6, 7, 8, 9, 10]):
    """ Plots the average reward and ME index over time for the single agent setting (pragmatic agent only). 
        The plots include the simulations with n_array states, where one message was left out from training.

        :param ablation:    indicates whether calculations are for the standard evaluation or the ablation test; for
                            the ablation study the pragmatic reasoning abilities of the agents are changed at test time.
        :param n_array:     array of n values to plot for
    """

    colors = ['#3cf49c', '#13eac9', '#0dc5d0', '#0986b5', '#055199']

    fig = plt.figure(figsize=(8, 4))

    # n: number of states
    for n_index, n in enumerate(n_array):

        n_messages = 2*n
        n_states = n

        # which epochs should be plotted for three and ten states
        indices = range(100)
        # at what step size the error bars should be plotted for three and ten states
        error_step = 10

        # ax = fig.add_subplot(1, 2, n_index + 1)

        agent = 'L1'

        # for a, agent in enumerate(['L0', 'L1']):

        # determine which lexica to evaluate and which reasoning ability at test time
        agent_body = agent  # agent body determines whether the lexica of literal or pragmatic agents are used
        agent_head = agent  # agent head determined whether the literal or pragmatic reasoning is used at test time
        if ablation and agent_body == 'L0':
            agent_head = 'L1'
        elif ablation and agent_body == 'L1':
            agent_head = 'L0'

        # set file path
        if agent_body == 'L0':
            filename = ('data/bilingual/L0/' + str(n) + '_states/' + agent_body + '_1missing_')
        elif agent_body == 'L1':
            filename = ('data/bilingual/L1/' + str(n) + '_states/' + agent_body + '_1missing_5.0alpha_')

        # load lexica and rewards until the maximum epoch that should be plotted
        lexica_all = []
        rewards_all = []
        for run in range(1, 101):
            rewards = np.load(filename + 'rewards_run' + str(run) + '.npy')
            lexica = np.load(filename + 'lexicon_run' + str(run) + '.npy')
            lexica_all.append(lexica[indices])
            rewards_all.append(rewards[indices])

        # determine the test input as the example that was left out from training
        agent_input = np.zeros((1, n_messages, n_states), dtype=np.float32)
        agent_input[0, n - 1, :] = np.ones((1, n_states), dtype=np.float32)

        # re-sort rewards and indices such that the values of all agents are pooled together per time step
        # similarly calculate the ME index for every agent at every time step and pool per time step
        me_index_all = np.zeros((len(indices), 100))
        rewards_sorted = np.zeros((len(indices), 100))

        for idx in indices:
            me_index = np.zeros((100))  # unnormalized ME index

            for run in range(100):

                rewards_sorted[idx, run] = rewards_all[run][idx]
                lexicon = lexica_all[run][idx]

                if agent_head == 'L0':
                    listener = RSAListener0(n_states, n_messages, lexicon)
                elif agent_head == 'L1':
                    listener = RSAListener1(n_states, n_messages, lexicon, alpha=5.)
                policy, _ = listener.get_states(agent_input)
                policy = np.squeeze(policy[:])
                policy = policy / np.sum(policy)
                me_index[run] = policy[-1]

            # normalize the ME index for this agent and time step and append to the ME indices of the other
            # agents for that time step
            me_index_all[idx, :] = (me_index - (1 / n)) / ((n - 1) / n)

        # plot mean and stds for the rewards and ME indices over time
        #mean_rewards = np.mean(rewards_sorted, axis=1)
        #std_rewards = np.std(rewards_sorted, axis=1)
        mean_me_index = np.mean(me_index_all, axis=1)
        std_me_index = np.std(me_index_all, axis=1)

        maxval = max(mean_me_index[0:36])
        maxvalidx = np.squeeze(np.where(mean_me_index == maxval))
        minval = min(mean_me_index[maxvalidx:100])
        minvalidx = np.squeeze(np.where(mean_me_index == minval))
        print(str(n) + "\t" + str(maxvalidx) + ":" + str(maxval) + "; " + str(minvalidx) + ":" + str(minval))

        #plt.errorbar(indices, mean_rewards, yerr=std_rewards, errorevery=error_step, color=colors[n_index],
        #            linewidth=3.0)
        plt.errorbar(indices, mean_me_index, yerr=std_me_index, errorevery=error_step, color=colors[n_index],
                    linewidth=3.0, linestyle='dashed')

        plt.xticks(fontsize=20)
        plt.yticks([0, 1], fontsize=20)
        plt.xlabel('epoch', fontsize=25)
        plt.ylabel('mean ME index', fontsize=25)

    # if n == 10:
    #box = plt.get_position()
    #plt.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.legend(['n=' + str(i) for i in n_array],
                loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    fig.tight_layout()

def plot_rewards_plus_me_index_two_agents(ablation=False):
    """ Plots the average reward and ME index over time for the two agent setting.
        The plots include the simulations with three and ten states where one state was left out from training.

        :param ablation:    indicates whether calculations are for the standard evaluation or the ablation test; for
                            the ablation study the pragmatic reasoning abilities of the agents are changed at test time.
    """

    colors = [['red', 'red'], ['blue', 'blue']]

    fig = plt.figure(figsize=(15, 4))

    # n: number of states
    for n_index, n in enumerate([3, 10]):

        # which epochs should be plotted for three and ten states
        indices = [list(range(100)), list(range(1000))][n_index]
        # at what step size the error bars should be plotted for three and ten states
        error_step = [10, 100][n_index]

        ax = fig.add_subplot(1, 2, n_index + 1)

        for a, agent in enumerate(['L0_S0', 'L1_S1']):

            # determine which lexica to evaluate and which reasoning ability at test time
            agent_body = agent  # agent body determines whether the lexica of literal or pragmatic agents are used
            agent_head = agent  # agent head determined whether the literal or pragmatic reasoning is used at test time
            if ablation and agent_body == 'L0_S0':
                agent_head = 'L1_S1'
            elif ablation and agent_body == 'L1_S1':
                agent_head = 'L0_S0'

            # set file path
            if agent == 'L0_S0':
                filename = 'data/communication/' + str(n) + '_states/' + agent_body + '_1missing_'
            if agent == 'L1_S1':
                filename = 'data/communication/' + str(n) + '_states/' + agent_body + '_1missing_5.0alpha_'

            # load lexica and rewards until the maximum epoch that should be plotted
            lexica_all_speaker = []
            lexica_all_listener = []
            rewards_all = []
            for run in range(1, 101):
                rewards = np.load(filename + 'rewards_run' + str(run) + '.npy')
                lexica_speaker = np.load(filename + 'lexicon_S_run' + str(run) + '.npy')
                lexica_listener = np.load(filename + 'lexicon_L_run' + str(run) + '.npy')
                lexica_all_speaker.append(lexica_speaker)
                lexica_all_listener.append(lexica_listener)
                rewards_all.append(rewards)

            # determine the test input as the example that was left out from training
            agent_input = np.zeros((1, n, n), dtype=np.float32)
            agent_input[0, n - 1, :] = np.ones((1, n), dtype=np.float32)

            me_index_all = np.zeros((len(indices), 100))
            rewards_sorted = np.zeros((len(indices), 100))

            # re-sort rewards and indices such that the values of all agents are pooled together per time step
            # similarly calculate the ME index for every agent at every time step and pool per time step
            for count_indices, index in enumerate(indices):

                me_index = np.zeros(100)  # unnormalized ME index

                for run in range(100):

                    rewards_sorted[count_indices, run] = rewards_all[run][index]

                    lexicon_speaker = lexica_all_speaker[run][index]
                    lexicon_listener = lexica_all_listener[run][index]

                    if agent_head == 'L0_S0':
                        listener = RSAListener0(n, n, lexicon_listener)
                        speaker = RSASpeaker0(n, n, lexicon_speaker)
                    elif agent_head == 'L1_S1':
                        listener = RSAListener1(n, n, lexicon_listener, alpha=5.)
                        speaker = RSASpeaker1(n, n, lexicon_speaker, alpha=5.)

                    # get speaker's policy for test input state
                    policy, _ = speaker.get_messages(agent_input)
                    policy = np.squeeze(policy[:])
                    policy = policy / np.sum(policy)

                    # get listener's policy for every input message weighted by probability that speaker generates
                    # that message given the test input state
                    selections = np.zeros((n, n))
                    for input_idx in range(n):
                        listener_input = np.zeros((1, n, n))
                        listener_input[0, input_idx, :] = np.ones((1, n))
                        selection, _ = listener.get_states(listener_input)
                        selection = np.squeeze(selection[:])
                        selection = selection / np.sum(selection)
                        selections[input_idx, :] = policy[input_idx] * selection
                    # sum up to get the probability of the speaker selecting any state given that the speaker gets the
                    # test input state
                    selections = np.sum(selections, axis=0)

                    # normalize the ME index for this agent and time step and append to the ME indices of the other
                    # agents for that time step
                    me_index[run] = (selections[-1] / np.sum(selections) - 1 / n) / ((n - 1) / n)

                me_index_all[count_indices, :] = me_index

            # plot mean and stds for the rewards and ME indices over time
            mean_rewards = np.mean(rewards_sorted, axis=1)
            std_rewards = np.std(rewards_sorted, axis=1)
            mean_me_index = np.mean(me_index_all, axis=1)
            std_me_index = np.std(me_index_all, axis=1)

            if n == 10 and agent_body == 'L1_S1':
                error_step = 90
            ax.errorbar(indices, mean_rewards, yerr=std_rewards, errorevery=error_step, color=colors[a][0],
                        linewidth=3.0)
            if n == 10 and agent_body == 'L1_S1':
                error_step = 95
            ax.errorbar(indices, mean_me_index, yerr=std_me_index, errorevery=error_step, color=colors[a][1],
                        linewidth=3.0, linestyle='dashed')

            plt.xticks(fontsize=20)
            plt.yticks([0, 1], fontsize=20)
            plt.xlabel('epoch', fontsize=25)
            plt.ylabel('mean reward /\n ME index', fontsize=25)

        if n == 10:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(['literal reward', 'literal $I_{ME}$', 'pragmatic reward', 'pragmatic $I_{ME}$'],
                      loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
        fig.tight_layout()
