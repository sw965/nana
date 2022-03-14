import copy
import numpy as np
import tensorflow as tf
import parrot
import seviper

class Network:
    INPUT_HEIGHT = 38
    INPUT_WIDTH = 38
    INPUT_DEPTH = 6
    ACTOR_OUTPUT_SIZE = 6
    CRITIC_OUTPUT_SIZE = 1

    def __init__(self, session):
        self.session = session
        self.input_holder = tf.placeholder(tf.float32, [None, 28, 28, 1])
        self.actor_target_holder = tf.placeholder(tf.float32, [None, Network.ACTOR_OUTPUT_SIZE])
        self.critic_target_holder = tf.placeholder(tf.float32, [None, Network.CRITIC_OUTPUT_SIZE])

        self.is_training_holder = tf.placeholder(tf.bool)
        stddev = 0.01

        filter1 = parrot.random_variable([3, 3, 1, 108], stddev=stddev)
        filter2 = parrot.random_variable([3, 3, 108, 108], stddev=stddev)
        filter3 = parrot.random_variable([3, 3, 108, 108], stddev=stddev)
        filter4 = parrot.random_variable([3, 3, 108, 108], stddev=stddev)
        filter5 = parrot.random_variable([3, 3, 108, 108], stddev=stddev)
        filter6 = parrot.random_variable([3, 3, 108, 108], stddev=stddev)

        actor_filter1 = parrot.random_variable([3, 3, 108, 108], stddev=stddev)
        actor_filter2 = parrot.random_variable([1, 1, 108, Network.ACTOR_OUTPUT_SIZE], stddev=stddev)

        critic_filter1 = parrot.random_variable([3, 3, 108, 108], stddev=stddev)
        critic_filter2 = parrot.random_variable([1, 1, 108, Network.CRITIC_OUTPUT_SIZE], stddev=stddev)

        conv1 = parrot.conv2d(self.input_holder, filter1)
        bn1 = parrot.batch_normalization(conv1, self.is_training_holder)
        act1 = parrot.tanh_exp(bn1)

        conv2 = parrot.conv2d(act1, filter2)
        bn2 = parrot.batch_normalization(conv2, self.is_training_holder)
        act2 = parrot.tanh_exp(bn2)
        max_pool2 = parrot.max_pool(act2)

        conv3 = parrot.conv2d(max_pool2, filter3)
        bn3 = parrot.batch_normalization(conv3, self.is_training_holder)
        act3 = parrot.tanh_exp(bn3)

        conv4 = parrot.conv2d(act3, filter4)
        bn4 = parrot.batch_normalization(conv4, self.is_training_holder)
        act4 = parrot.tanh_exp(bn4)
        max_pool4 = parrot.max_pool(act4)

        conv5 = parrot.conv2d(max_pool4, filter5)
        bn5 = parrot.batch_normalization(conv5, self.is_training_holder)
        act5 = parrot.tanh_exp(bn5)

        conv6 = parrot.conv2d(act5, filter6)
        bn6 = parrot.batch_normalization(conv6, self.is_training_holder)
        act6 = parrot.tanh_exp(bn6)

        actor_conv1 = parrot.conv2d(act6, actor_filter1)
        actor_bn1 = parrot.batch_normalization(actor_conv1, self.is_training_holder)
        actor_act1 = parrot.tanh_exp(actor_bn1)

        actor_conv2 = parrot.conv2d(actor_act1, actor_filter2)
        actor_bn2 = parrot.batch_normalization(actor_conv2, self.is_training_holder)
        actor_act2 = parrot.tanh_exp(actor_bn2)
        actor_gap = parrot.GAP(actor_act2)

        critic_conv1 = parrot.conv2d(act6, critic_filter1)
        critic_bn1 = parrot.batch_normalization(critic_conv1, self.is_training_holder)
        critic_act1 = parrot.tanh_exp(critic_bn1)

        critic_conv2 = parrot.conv2d(critic_act1, critic_filter2)
        critic_bn2 = parrot.batch_normalization(critic_conv2, self.is_training_holder)
        critic_act2 = parrot.tanh_exp(critic_bn2)
        critic_gap = parrot.GAP(critic_act2)

        self.actor_output = tf.nn.tanh(actor_gap)
        self.critic_output = tf.nn.tanh(critic_gap)

        self.actor_loss = parrot.mean_squared_error(self.actor_output, self.actor_target_holder)
        self.critic_loss = parrot.mean_squared_error(self.critic_output, self.critic_target_holder)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.actor_train = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(self.actor_loss)
            self.critic_train = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(self.critic_loss)
        self.saver = tf.train.Saver()

    def run_actor_output(self, input_data):
        feed_dict = {self.input_holder:input_data, self.is_training_holder:False}
        return self.session.run(self.actor_output, feed_dict=feed_dict)

    def run_critic_output(self, input_data):
        feed_dict = {self.input_holder:input_data, self.is_training_holder:False}
        return self.session.run(self.critic_output, feed_dict=feed_dict)

    def run_actor_train(self, input_data, target):
        feed_dict = {self.input_holder:input_data,
                     self.actor_target_holder:target,
                     self.is_training_holder:True}
        self.session.run(self.actor_train, feed_dict=feed_dict)

    def run_critic_train(self, input_data, target):
        feed_dict = {self.input_holder:input_data,
                     self.critic_target_holder:target,
                     self.is_training_holder:True}
        self.session.run(self.critic_train, feed_dict=feed_dict)

    def run_loss(self, input_data, target):
        feed_dict = {self.input_holder:input_data,
                     self.target_holder:target,
                     self.is_training_holder:False}
        return self.session.run(self.loss, feed_dict=feed_dict)

    def run_accuracy(self, input_data, target):
        feed_dict = {self.input_holder:input_data,
                     self.actor_target_holder:target,
                     self.is_training_holder:False}
        return self.session.run(parrot.accuracy(self.actor_output, self.actor_target_holder), feed_dict=feed_dict)

    def save(self, folder_path, file_name):
        self.saver.save(self.session, folder_path + file_name)

    def load(self, folder_path, file_name):
        self.saver.restore(self.session, folder_path + file_name)

    def actor(self, battle, temperature_parameter):
        action_commands = battle.p1_fighters[0].sorted_move_names + [pokemon.name for pokemon in battle.p1_fighters[1:]]
        legal_action_commands = battle.p1_fighters[0].legal_action_commands()

        input_data = battle.to_feature_array_3d()
        output = parrot.self.run_actor_output(input_data)
        output = [v if action_command in legal_action_commands else 0.0 for action_command in action_commands]
        index = parrot.boltzmann_random(output[0], temperature_parameter)
        return action_commands[i]

    def trainer(self, temperature_parameter):
        def result(battle):
            return self.actor(battle, temperature_parameter)
        return result

if __name__ == "__main__":
    ...
    # train_data, target_data, test_data, test_target = parrot.load_mnist(flatten=True, binary=True)
    # train_data = train_data.reshape(60000, 28, 28, 1)
    # test_data = test_data.reshape(10000, 28, 28, 1)
    # session = tf.Session()
    # network = Network(session)
    # session.run(tf.global_variables_initializer())
    #
    # for i in range(12800):
    #     indices = np.random.choice(60000, 128)
    #     network.run_actor_train(train_data[indices], target_data[indices])
    #     if i%100 == 0:
    #         indices = np.random.choice(10000, 128)
    #         print(network.run_accuracy(test_data[indices], test_target[indices]))
    #         print(network.run_actor_output(test_data[0].reshape(1, 28, 28, 1)))
