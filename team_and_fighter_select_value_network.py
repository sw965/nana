import copy
import json
import os
import random
import numpy as np
import tensorflow as tf
import boa
import parrot
import seviper

NO_FEATURE_VALUE = 0
FEATURE_VALUE = 1

POKE_NAME_FEATURES = [seviper.EMPTY] + seviper.POKE_NAMES
MOVE_NAME_FEATURES = [seviper.EMPTY] + seviper.MOVE_NAMES
FIGHTER_SELECT_INDEX_FEATURES = [0, 1, 2, 3, 4, 5]

INPUT_STATUS_LENGTH = sum([
    len(POKE_NAME_FEATURES),
    len(seviper.LEVELS),
    len(seviper.NATURES),
    len(MOVE_NAME_FEATURES) * seviper.MAX_MOVESET_NUM,
    len(seviper.IVS) * seviper.STATUS_NUM,
    len(seviper.VALID_EVS) * seviper.STATUS_NUM,
    len(POKE_NAME_FEATURES),
    len(seviper.LEVELS),
]) * seviper.MAX_TEAM_NUM
INPUT_STATUS_LENGTH += sum([len(FIGHTER_SELECT_INDEX_FEATURES)]) * seviper.FIGHTERS_NUM

INPUT_INDEX_GENE = (i for i in range(INPUT_STATUS_LENGTH))
INPUT_INDEX_TO_FEATURE = []

class InputInfo:
    def __init__(self, features, length):
        self.features = features
        self.input_ranges = [[next(INPUT_INDEX_GENE) for _ in range(len(features))]\
                              for _ in range(length)]

        for i in range(length):
            for feature_i, feature in enumerate(features):
                index = self.input_ranges[i][feature_i]
                INPUT_INDEX_TO_FEATURE.append(feature)

TEAM_POKE_NAME_INPUT_INFO = InputInfo(POKE_NAME_FEATURES, seviper.MAX_TEAM_NUM)
LEVEL_INPUT_INFO = InputInfo(seviper.LEVELS, seviper.MAX_TEAM_NUM)
NATURE_INPUT_INFO = InputInfo(seviper.NATURES, seviper.MAX_TEAM_NUM)

MOVE1_NAME_INPUT_INFO = InputInfo(MOVE_NAME_FEATURES, seviper.MAX_TEAM_NUM)
MOVE2_NAME_INPUT_INFO = InputInfo(MOVE_NAME_FEATURES, seviper.MAX_TEAM_NUM)
MOVE3_NAME_INPUT_INFO = InputInfo(MOVE_NAME_FEATURES, seviper.MAX_TEAM_NUM)
MOVE4_NAME_INPUT_INFO = InputInfo(MOVE_NAME_FEATURES, seviper.MAX_TEAM_NUM)

HP_IV_INPUT_INFO = InputInfo(seviper.IVS, seviper.MAX_TEAM_NUM)
ATK_IV_INPUT_INFO = InputInfo(seviper.IVS, seviper.MAX_TEAM_NUM)
DEF_IV_INPUT_INFO = InputInfo(seviper.IVS, seviper.MAX_TEAM_NUM)
SP_ATK_IV_INPUT_INFO = InputInfo(seviper.IVS, seviper.MAX_TEAM_NUM)
SP_DEF_IV_INPUT_INFO = InputInfo(seviper.IVS, seviper.MAX_TEAM_NUM)
SPEED_IV_INPUT_INFO = InputInfo(seviper.IVS, seviper.MAX_TEAM_NUM)

HP_EV_INPUT_INFO = InputInfo(seviper.VALID_EVS, seviper.MAX_TEAM_NUM)
ATK_EV_INPUT_INFO = InputInfo(seviper.VALID_EVS, seviper.MAX_TEAM_NUM)
DEF_EV_INPUT_INFO = InputInfo(seviper.VALID_EVS, seviper.MAX_TEAM_NUM)
SP_ATK_EV_INPUT_INFO = InputInfo(seviper.VALID_EVS, seviper.MAX_TEAM_NUM)
SP_DEF_EV_INPUT_INFO = InputInfo(seviper.VALID_EVS, seviper.MAX_TEAM_NUM)
SPEED_EV_INPUT_INFO = InputInfo(seviper.VALID_EVS, seviper.MAX_TEAM_NUM)

P2_TEAM_POKE_NAME_INPUT_INFO = InputInfo(POKE_NAME_FEATURES, seviper.MAX_TEAM_NUM)
P2_LEVEL_INPUT_INFO = InputInfo(seviper.LEVELS, seviper.MAX_TEAM_NUM)
FIGHTER_SELECT_INDEX_INPUT_INFO = InputInfo(FIGHTER_SELECT_INDEX_FEATURES, seviper.FIGHTERS_NUM)

try:
    next(INPUT_INDEX_GENE)
    assert False
except StopIteration:
    del INPUT_INDEX_GENE

def make_init_input_status():
    return np.array([NO_FEATURE_VALUE for _ in range(Network.INPUT_SIZE)])

def input_status_to_2d(input_status):
    return input_status.reshape(1, Network.INPUT_SIZE)

def input_status_to_feature(input_status, input_info, input_ranges_index):
    input_range = input_info.input_ranges[input_ranges_index]
    features = [INPUT_INDEX_TO_FEATURE[input_index] for input_index in input_range \
                if input_status[input_index] == FEATURE_VALUE]
    if len(features) == 0:
        return None
    assert len(features) == 1
    return features[0]

def input_status_to_team_poke_name(input_status, team_index):
    return input_status_to_feature(input_status, TEAM_POKE_NAME_INPUT_INFO, team_index)

def input_status_to_team_poke_names(input_status):
    return [input_status_to_team_poke_name(input_status, i) for i in range(seviper.MAX_TEAM_NUM)]

def input_status_to_team_filled_indices(input_status):
    team_poke_names = input_status_to_team_poke_names(input_status)
    return [i for i, poke_name in enumerate(team_poke_names) if poke_name != seviper.EMPTY and poke_name is not None]

def input_status_to_level(input_status, team_index):
    return input_status_to_feature(input_status, LEVEL_INPUT_INFO, team_index)

def input_status_to_levels(input_status):
    return [input_status_to_level(input_status, i) for i in range(seviper.MAX_TEAM_NUM)]

def input_status_to_nature(input_status, team_index):
    return input_status_to_feature(input_status, NATURE_INPUT_INFO, team_index)

def input_status_to_natures(input_status):
    return [input_status_to_nature(input_status, i) for i in range(seviper.MAX_TEAM_NUM)]

def input_status_to_move1_name(input_status, team_index):
    return input_status_to_feature(input_status, MOVE1_NAME_INPUT_INFO, team_index)

def input_status_to_move1_names(input_status):
    return [input_status_to_move1_name(input_status, i) for i in range(seviper.MAX_TEAM_NUM)]

def input_status_to_move2_name(input_status, team_index):
    return input_status_to_feature(input_status, MOVE2_NAME_INPUT_INFO, team_index)

def input_status_to_move2_names(input_status):
    return [input_status_to_move2_name(input_status, i) for i in range(seviper.MAX_TEAM_NUM)]

def input_status_to_move3_name(input_status, team_index):
    return input_status_to_feature(input_status, MOVE3_NAME_INPUT_INFO, team_index)

def input_status_to_move3_names(input_status):
    return [input_status_to_move3_name(input_status, i) for i in range(seviper.MAX_TEAM_NUM)]

def input_status_to_move4_name(input_status, team_index):
    return input_status_to_feature(input_status, MOVE4_NAME_INPUT_INFO, team_index)

def input_status_to_move4_names(input_status):
    return [input_status_to_move4_name(input_status, i) for i in range(seviper.MAX_TEAM_NUM)]

def input_status_to_moveset(input_status, team_index):
    return [input_status_to_move1_name(input_status, team_index),
            input_status_to_move2_name(input_status, team_index),
            input_status_to_move3_name(input_status, team_index),
            input_status_to_move4_name(input_status, team_index)]

def input_status_to_hp_iv(input_status, team_index):
    return input_status_to_feature(input_status, HP_IV_INPUT_INFO, team_index)

def input_status_to_hp_ivs(input_status):
    return [input_status_to_hp_iv(input_status, i) for i in range(seviper.MAX_TEAM_NUM)]

def input_status_to_atk_iv(input_status, team_index):
    return input_status_to_feature(input_status, ATK_IV_INPUT_INFO, team_index)

def input_status_to_atk_ivs(input_status):
    return [input_status_to_atk_iv(input_status, i) for i in range(seviper.MAX_TEAM_NUM)]

def input_status_to_def_iv(input_status, team_index):
    return input_status_to_feature(input_status, DEF_IV_INPUT_INFO, team_index)

def input_status_to_def_ivs(input_status):
    return [input_status_to_def_iv(input_status, i) for i in range(seviper.MAX_TEAM_NUM)]

def input_status_to_sp_atk_iv(input_status, team_index):
    return input_status_to_feature(input_status, SP_ATK_IV_INPUT_INFO, team_index)

def input_status_to_sp_atk_ivs(input_status):
    return [input_status_to_sp_atk_iv(input_status, i) for i in range(seviper.MAX_TEAM_NUM)]

def input_status_to_sp_def_iv(input_status, team_index):
    return input_status_to_feature(input_status, SP_DEF_IV_INPUT_INFO, team_index)

def input_status_to_sp_def_ivs(input_status):
    return [input_status_to_sp_def_iv(input_status, i) for i in range(seviper.MAX_TEAM_NUM)]

def input_status_to_speed_iv(input_status, team_index):
    return input_status_to_feature(input_status, SPEED_IV_INPUT_INFO, team_index)

def input_status_to_speed_ivs(input_status):
    return [input_status_to_speed_iv(input_status, i) for i in range(seviper.MAX_TEAM_NUM)]

def input_status_to_ivs(input_status, team_index):
    hp = input_status_to_hp_iv(input_status, team_index)
    atk = input_status_to_atk_iv(input_status, team_index)
    defe = input_status_to_def_iv(input_status, team_index)
    sp_atk = input_status_to_sp_atk_iv(input_status, team_index)
    sp_def = input_status_to_sp_def_iv(input_status, team_index)
    speed = input_status_to_speed_iv(input_status, team_index)
    result = [hp, atk, defe, sp_atk, sp_def, speed]
    return [value if value is not None else 0 for value in result]

def input_status_to_hp_ev(input_status, team_index):
    return input_status_to_feature(input_status, HP_EV_INPUT_INFO, team_index)

def input_status_to_hp_evs(input_status):
    return [input_status_to_hp_ev(input_status, i) for i in range(seviper.MAX_TEAM_NUM)]

def input_status_to_atk_ev(input_status, team_index):
    return input_status_to_feature(input_status, ATK_EV_INPUT_INFO, team_index)

def input_status_to_atk_evs(input_status):
    return [input_status_to_atk_ev(input_status, i) for i in range(seviper.MAX_TEAM_NUM)]

def input_status_to_def_ev(input_status, team_index):
    return input_status_to_feature(input_status, DEF_EV_INPUT_INFO, team_index)

def input_status_to_def_evs(input_status):
    return [input_status_to_def_ev(input_status, i) for i in range(seviper.MAX_TEAM_NUM)]

def input_status_to_sp_atk_ev(input_status, team_index):
    return input_status_to_feature(input_status, SP_ATK_EV_INPUT_INFO, team_index)

def input_status_to_sp_atk_evs(input_status):
    return [input_status_to_sp_atk_ev(input_status, i) for i in range(seviper.MAX_TEAM_NUM)]

def input_status_to_sp_def_ev(input_status, team_index):
    return input_status_to_feature(input_status, SP_DEF_EV_INPUT_INFO, team_index)

def input_status_to_sp_def_evs(input_status):
    return [input_status_to_sp_def_ev(input_status, i) for i in range(seviper.MAX_TEAM_NUM)]

def input_status_to_speed_ev(input_status, team_index):
    return input_status_to_feature(input_status, SPEED_EV_INPUT_INFO, team_index)

def input_status_to_speed_evs(input_status):
    return [input_status_to_speed_ev(input_status, i) for i in range(seviper.MAX_TEAM_NUM)]

def input_status_to_evs(input_status, team_index):
    hp = input_status_to_hp_ev(input_status, team_index)
    atk = input_status_to_atk_ev(input_status, team_index)
    defe = input_status_to_def_ev(input_status, team_index)
    sp_atk = input_status_to_sp_atk_ev(input_status, team_index)
    sp_def = input_status_to_sp_def_ev(input_status, team_index)
    speed = input_status_to_speed_ev(input_status, team_index)
    result = [hp, atk, defe, sp_atk, sp_def, speed]
    return [value if value is not None else 0 for value in result]

def input_status_to_json_team(input_status):
    team_poke_names = input_status_to_team_poke_names(input_status)
    levels = input_status_to_levels(input_status)
    natures = input_status_to_natures(input_status)

    move1_names = input_status_to_move1_names(input_status)
    move2_names = input_status_to_move2_names(input_status)
    move3_names = input_status_to_move3_names(input_status)
    move4_names = input_status_to_move4_names(input_status)

    hp_ivs = input_status_to_hp_ivs(input_status)
    atk_ivs = input_status_to_atk_ivs(input_status)
    def_ivs = input_status_to_def_ivs(input_status)
    sp_atk_ivs = input_status_to_sp_atk_ivs(input_status)
    sp_def_ivs = input_status_to_sp_def_ivs(input_status)
    speed_ivs = input_status_to_speed_ivs(input_status)

    hp_evs = input_status_to_hp_evs(input_status)
    atk_evs = input_status_to_atk_evs(input_status)
    def_evs = input_status_to_def_evs(input_status)
    sp_atk_evs = input_status_to_sp_atk_evs(input_status)
    sp_def_evs = input_status_to_sp_def_evs(input_status)
    speed_evs = input_status_to_speed_evs(input_status)

    data = {"PokeNames":team_poke_names, "Levels":levels, "Natures":natures,
            "Move1Names":move1_names, "Move2Names":move2_names,
            "Move3Names":move3_names, "Move4Names":move4_names,
            "HPIVs":hp_ivs, "AtkIVs":atk_ivs, "DefIVs":def_ivs,
            "SpAtkIVs":sp_atk_ivs, "SpDefIVs":sp_def_ivs, "SpeedIVs":speed_ivs,
            "HPEVs":hp_evs, "AtkEVs":atk_evs, "DefEVs":def_evs,
            "SpAtkEVs":sp_atk_evs, "SpDefEVs":sp_def_evs, "SpeedEVs":speed_evs}
    return json.dumps(data, ensure_ascii=False, indent=4)

def input_status_to_p2_team_poke_names(input_status):
    return [input_status_to_feature(input_status, P2_TEAM_POKE_NAME_INPUT_INFO, i) for i in range(seviper.MAX_TEAM_NUM)]

def input_status_to_p2_levels(input_status):
    return [input_status_to_feature(input_status, P2_LEVEL_INPUT_INFO, i) for i in range(seviper.MAX_TEAM_NUM)]

def input_p2_select_phase_features(input_status, p2_input_status):
    p2_team_poke_names = input_status_to_team_poke_names(p2_input_status)
    assert None not in p2_team_poke_names
    p2_levels = input_status_to_levels(p2_input_status)
    result = copy.deepcopy(input_status)

    def input_features(features, input_info):
        for i, feature in enumerate(features):
            if feature is None:
                continue
            j = input_info.features.index(feature)
            input_index = input_info.input_ranges[i][j]
            assert result[input_index] == NO_FEATURE_VALUE
            result[input_index] = FEATURE_VALUE

    input_features(p2_team_poke_names, P2_TEAM_POKE_NAME_INPUT_INFO)
    input_features(p2_levels, P2_LEVEL_INPUT_INFO)
    return result

def is_team_build_completed(input_status):
    end = SPEED_EV_INPUT_INFO.input_ranges[-1][-1] + 1
    input_status = input_status[:end]

    team_poke_names = input_status_to_team_poke_names(input_status)
    if team_poke_names.count(None) != 0:
        return False

    team_num = len(input_status_to_team_filled_indices(input_status))
    return sum(input_status) == seviper.MAX_TEAM_NUM + (team_num * 18)

class Network:
    """
    入力データの情報

    ポケモンの名前
    レベル
    性格

    技1
    技2
    技3
    技4

    HP個体値
    攻撃個体値
    防御個体値
    特攻個体値
    特防個体値
    素早さ個体値

    HP努力値
    攻撃努力値
    防御努力値
    特攻努力値
    特防努力値
    素早さ努力値

    相手のポケモン名
    相手のレベル
    ファイター選択インデックス
    """

    INPUT_SIZE = INPUT_STATUS_LENGTH
    OUTPUT_SIZE = 1
    INSTANCE_COUNT = 0

    def __init__(self, session, l2_rate):
        Network.INSTANCE_COUNT += 1
        assert Network.INSTANCE_COUNT == 1

        self.session = session

        self.input_status_holder = tf.placeholder(tf.float32, [None, Network.INPUT_SIZE])
        self.target_value_holder = tf.placeholder(tf.float32, [None, Network.OUTPUT_SIZE])
        self.is_training_holder = tf.placeholder(tf.bool)
        self.keep_holder = tf.placeholder(tf.float32)

        stddev = 0.05
        w1_size = 128
        w2_size = 128
        w3_size = 128

        layers1, param1 = parrot.matmul_BN_prelu_dropout_with_param(self.input_status_holder, w1_size, stddev,
                                                                  self.is_training_holder, self.keep_holder)
        layers2, param2 = parrot.matmul_BN_prelu_dropout_with_param(layers1, w2_size, stddev,
                                                                  self.is_training_holder, self.keep_holder)
        layers3, param3 = parrot.matmul_BN_prelu_dropout_with_param(layers2, w3_size, stddev,
                                                                  self.is_training_holder, self.keep_holder)
        self.output, output_param = parrot.matmul_prelu_dropout_output_with_param(layers3, Network.OUTPUT_SIZE,
                                                                                stddev, self.keep_holder, tf.nn.tanh)

        l2 = sum([tf.nn.l2_loss(w) for w in [param1["w"], param2["w"], param3["w"], output_param["w"]]])
        self.loss = parrot.mean_squared_error(self.output, self.target_value_holder) + (l2 * l2_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train = parrot.AdaBoundOptimizer(final_lr=1.0).minimize(self.loss)
        self.saver = tf.train.Saver()

    def run_output(self, input_status):
        feed_dict = {self.input_status_holder:input_status, self.is_training_holder:False, self.keep_holder:1.0}
        return self.session.run(self.output, feed_dict=feed_dict)

    def run_train(self, input_status, target_value, keep):
        feed_dict = {self.input_status_holder:input_status,
                     self.target_value_holder:target_value,
                     self.is_training_holder:True,
                     self.keep_holder:keep}
        self.session.run(self.train, feed_dict=feed_dict)

    def run_loss(self, input_status, target_value):
        feed_dict = {self.input_status_holder:input_status, self.target_value:target_value,
                     self.is_training_holder:False, self.keep_holder:1.0}
        return self.session.run(self.loss, feed_dict=feed_dict)

    def save(self, folder_path, file_name):
        self.saver.save(self.session, folder_path + file_name)

    def load(self, folder_path, file_name):
        self.saver.restore(self.session, folder_path + file_name)

    def team_evaluation_info(self, input_status, input_info, team_index, exclusion_features):
        input_range = input_info.input_ranges[team_index]
        input_statuses = [copy.deepcopy(input_status) for _ in range(len(input_range))]

        for i, input_index in enumerate(input_range):
            assert input_statuses[i][input_index] == NO_FEATURE_VALUE
            input_statuses[i][input_index] = FEATURE_VALUE

        evaluations = [value[0] for value in self.run_output(input_statuses)]
        evaluations_result = [evaluations[i] for i, input_index in enumerate(input_range) \
                              if INPUT_INDEX_TO_FEATURE[input_index] not in exclusion_features]
        input_range_result = [input_index for input_index in input_range \
                              if INPUT_INDEX_TO_FEATURE[input_index] not in exclusion_features]
        assert len(evaluations_result) != 0
        return {"evaluations":evaluations_result, "input_range":input_range_result}

    def team_poke_name_evaluation_info(self, input_status, team_index):
        team_poke_names = input_status_to_team_poke_names(input_status)
        team_poke_names = [poke_name for poke_name in team_poke_names if poke_name is not None]
        assert len(team_poke_names) == team_index

        exclusion_poke_names = [poke_name for poke_name in team_poke_names if poke_name != seviper.EMPTY]
        if len(team_poke_names) <= 2:
            exclusion_poke_names += [seviper.EMPTY]
        return self.team_evaluation_info(input_status, TEAM_POKE_NAME_INPUT_INFO, team_index, exclusion_poke_names)

    def level_evaluation_info(self, input_status, team_index):
        poke_name = input_status_to_team_poke_name(input_status, team_index)
        level = input_status_to_level(input_status, team_index)
        exclusion_levels = [level for level in LEVEL_INPUT_INFO.features \
                            if not seviper.is_valid_level(poke_name, level)]
        return self.team_evaluation_info(input_status, LEVEL_INPUT_INFO, team_index, exclusion_levels)

    def nature_evaluation_info(self, input_status, team_index):
        return self.team_evaluation_info(input_status, NATURE_INPUT_INFO, team_index, [])

    def move_name_evaluation_info(self, input_status, move_name_input_info, team_index):
        poke_name = input_status_to_team_poke_name(input_status, team_index)
        level = input_status_to_level(input_status, team_index)
        assert poke_name != seviper.EMPTY
        assert poke_name is not None

        moveset = input_status_to_moveset(input_status, team_index)
        moveset = [move_name for move_name in moveset if move_name is not None]

        exclusion_move_names = [move_name for move_name in moveset if move_name != seviper.EMPTY]
        if len(moveset) == 0:
            exclusion_move_names += [seviper.EMPTY]
        exclusion_move_names += [move_name for move_name in seviper.MOVE_NAMES if move_name in moveset]
        exclusion_move_names += [move_name for move_name in seviper.MOVE_NAMES \
                                 if not seviper.can_learn_move_name(poke_name, move_name, level)]
        return self.team_evaluation_info(input_status, move_name_input_info, team_index, exclusion_move_names)

    def move1_name_evaluation_info(self, input_status, team_index):
        return self.move_name_evaluation_info(input_status, MOVE1_NAME_INPUT_INFO, team_index)

    def move2_name_evaluation_info(self, input_status, team_index):
        return self.move_name_evaluation_info(input_status, MOVE2_NAME_INPUT_INFO, team_index)

    def move3_name_evaluation_info(self, input_status, team_index):
        return self.move_name_evaluation_info(input_status, MOVE3_NAME_INPUT_INFO, team_index)

    def move4_name_evaluation_info(self, input_status, team_index):
        return self.move_name_evaluation_info(input_status, MOVE4_NAME_INPUT_INFO, team_index)

    def hp_iv_evaluation_info(self, input_status, team_index):
        return self.team_evaluation_info(input_status, HP_IV_INPUT_INFO, team_index, [])

    def atk_iv_evaluation_info(self, input_status, team_index):
        return self.team_evaluation_info(input_status, ATK_IV_INPUT_INFO, team_index, [])

    def def_iv_evaluation_info(self, input_status, team_index):
        return self.team_evaluation_info(input_status, DEF_IV_INPUT_INFO, team_index, [])

    def sp_atk_iv_evaluation_info(self, input_status, team_index):
        return self.team_evaluation_info(input_status, SP_ATK_IV_INPUT_INFO, team_index, [])

    def sp_def_iv_evaluation_info(self, input_status, team_index):
        return self.team_evaluation_info(input_status, SP_DEF_IV_INPUT_INFO, team_index, [])

    def speed_iv_evaluation_info(self, input_status, team_index):
        return self.team_evaluation_info(input_status, SPEED_IV_INPUT_INFO, team_index, [])

    def ev_evaluation_info(self, input_status, ev_input_info, team_index):
        evs = input_status_to_evs(input_status, team_index)
        sum_ev = sum(evs)
        exclusion_ev = [ev for ev in seviper.VALID_EVS if (sum_ev + ev) > seviper.MAX_SUM_EV]
        return self.team_evaluation_info(input_status, ev_input_info, team_index, exclusion_ev)

    def hp_ev_evaluation_info(self, input_status, team_index):
        return self.ev_evaluation_info(input_status, HP_EV_INPUT_INFO, team_index)

    def atk_ev_evaluation_info(self, input_status, team_index):
        return self.ev_evaluation_info(input_status, ATK_EV_INPUT_INFO, team_index)

    def def_ev_evaluation_info(self, input_status, team_index):
        return self.ev_evaluation_info(input_status, DEF_EV_INPUT_INFO, team_index)

    def sp_atk_ev_evaluation_info(self, input_status, team_index):
        return self.ev_evaluation_info(input_status, SP_ATK_EV_INPUT_INFO, team_index)

    def sp_def_ev_evaluation_info(self, input_status, team_index):
        return self.ev_evaluation_info(input_status, SP_DEF_EV_INPUT_INFO, team_index)

    def speed_ev_evaluation_info(self, input_status, team_index):
        return self.ev_evaluation_info(input_status, SPEED_EV_INPUT_INFO, team_index)

    def select_team_parameter(self, input_status, team_index, evaluation_info_func, select_func):
        evaluation_info = evaluation_info_func(input_status, team_index)
        evaluations = evaluation_info["evaluations"]
        input_range = evaluation_info["input_range"]
        sigmoid_evaluations = np.vectorize(parrot.tanh_to_sigmoid)(evaluations)
        input_range_index = select_func(sigmoid_evaluations)
        return input_range[input_range_index]

    def boltzmann_select_team_parameter(self, input_status, team_index, evaluation_info_func,
                                        temperature_parameter):
        return self.select_team_parameter(
            input_status, team_index, evaluation_info_func,
            lambda sigmoid_evaluations:parrot.boltzmann_random(sigmoid_evaluations, temperature_parameter),
        )

    def team_evaluation_info_funcs(self):
        return [self.team_poke_name_evaluation_info,
                self.level_evaluation_info,
                self.nature_evaluation_info,

                self.move1_name_evaluation_info,
                self.move2_name_evaluation_info,
                self.move3_name_evaluation_info,
                self.move4_name_evaluation_info,

                self.hp_iv_evaluation_info,
                self.atk_iv_evaluation_info,
                self.def_iv_evaluation_info,
                self.sp_atk_iv_evaluation_info,
                self.sp_def_iv_evaluation_info,
                self.speed_iv_evaluation_info,

                self.hp_ev_evaluation_info,
                self.atk_ev_evaluation_info,
                self.def_ev_evaluation_info,
                self.sp_atk_ev_evaluation_info,
                self.sp_def_ev_evaluation_info,
                self.speed_ev_evaluation_info]

    def build_team(self, pokemon_temperature_parameter, poke_param_temperature_parameter, random_num):
        assert random_num >= 0 or random_num == -1

        if random_num == 0:
            assert poke_param_temperature_parameter == 0

        if random_num == -1:
            assert poke_param_temperature_parameter == -1

        input_status = make_init_input_status()
        team_build_status = []

        def select_pokemon():
            temperature_parameters = [
                pokemon_temperature_parameter, pokemon_temperature_parameter, pokemon_temperature_parameter,
                0, 0, 0,
            ]
            random.shuffle(temperature_parameters)

            for i, team_index in enumerate(range(seviper.MAX_TEAM_NUM)):
                input_index = self.boltzmann_select_team_parameter(input_status, team_index,
                                                                   self.team_poke_name_evaluation_info,
                                                                   temperature_parameters[i])
                assert input_status[input_index] == NO_FEATURE_VALUE
                input_status[input_index] = FEATURE_VALUE
                team_build_status.append(copy.deepcopy(input_status))

        select_pokemon()

        """ポケモンが空ではない枠を特定するteam_filled_indicesを作成する"""
        team_filled_indices = input_status_to_team_filled_indices(input_status)

        def make_poke_param_temperature_parameter_gane(length):
            if random_num != -1:
                random_list = [False for _ in range(length - random_num)] + [True for _ in range(random_num)]
            else:
                random_list = [True for _ in range(length)]
            random.shuffle(random_list)
            return (poke_param_temperature_parameter if is_random else 0 for is_random in random_list)

        def select_pokemon_parameter(evaluation_info_func):
            for team_index in team_filled_indices:
                input_index = self.boltzmann_select_team_parameter(input_status, team_index,
                                                                   evaluation_info_func,
                                                                   next(poke_param_temperature_parameter_gane))
                assert input_status[input_index] == NO_FEATURE_VALUE
                input_status[input_index] = FEATURE_VALUE
                team_build_status.append(copy.deepcopy(input_status))

        team_evaluation_info_funcs = self.team_evaluation_info_funcs()[1:]
        gene_length = len(team_filled_indices) * len(team_evaluation_info_funcs)
        poke_param_temperature_parameter_gane = make_random_percent_gane(gene_length)

        """ポケモンが空ではない枠のポケモンのパラメーターを決める"""
        for func in team_evaluation_info_funcs:
            select_pokemon_parameter(func)

        try:
            next(poke_param_temperature_parameter_gane)
            assert False
        except StopIteration:
            pass

        assert all([value == input_status[i] for i, value in enumerate(team_build_status[-1])])
        assert all([sum(value) == i + 1 for i, value in enumerate(team_build_status)])
        assert len(team_build_status) == (seviper.MAX_TEAM_NUM + gene_length)
        return team_build_status

    def build_teams(self, team_num, temperature_parameter, random_num):
        return [self.build_team(temperature_parameter, random_num) for _ in range(team_num)]

    def fighter_select_indices_evaluation_info(self, input_status):
        team_poke_names = input_status_to_team_poke_names(input_status)
        brute_force_select_indices = [
            [i, j, k] for i in range(seviper.MAX_TEAM_NUM) \
                      for j in range(seviper.MAX_TEAM_NUM) \
                      for k in range(seviper.MAX_TEAM_NUM) \
            if seviper.EMPTY not in [team_poke_names[i], team_poke_names[j], team_poke_names[k]] \
            and i != j and i != k and j != k \
        ]

        input_ranges = FIGHTER_SELECT_INDEX_INPUT_INFO.input_ranges
        input_indices_list = [[input_ranges[0][select_indices[0]],
                               input_ranges[1][select_indices[1]],
                               input_ranges[2][select_indices[2]]]
                              for select_indices in brute_force_select_indices]

        length = len(input_indices_list)
        input_statuses = [copy.deepcopy(input_status) for _ in range(length)]
        for i in range(length):
            input_indices = input_indices_list[i]
            for input_index in input_indices:
                assert input_statuses[i][input_index] == NO_FEATURE_VALUE
                input_statuses[i][input_index] = FEATURE_VALUE

        evaluations = [value[0] for value in self.run_output(input_statuses)]
        return {"evaluations":evaluations, "input_indices_list":input_indices_list}

    def fighter_select_indices(self, input_status, temperature_parameter):
        assert is_team_build_completed(input_status)

        p2_team_poke_names = input_status_to_p2_team_poke_names(input_status)
        assert p2_team_poke_names.count(None) == 0

        p2_levels = input_status_to_p2_levels(input_status)
        assert p2_levels.count(None) <= seviper.MIN_TEAM_NUM

        evaluation_info = self.fighter_select_indices_evaluation_info(input_status)
        evaluations = evaluation_info["evaluations"]
        input_indices_list = evaluation_info["input_indices_list"]
        sigmoid_evaluations = np.vectorize(parrot.tanh_to_sigmoid)(evaluations)
        indices_i = parrot.boltzmann_random(sigmoid_evaluations, temperature_parameter)
        return [FIGHTER_SELECT_INDEX_INPUT_INFO.input_ranges[fighter_index].index(input_index)\
                for fighter_index, input_index in enumerate(input_indices_list[indices_i])]

    def brute_force_battle_fighter_select_indices(self, team_build_statuses,
                                                  p1_temperature_parameter, p2_temperature_parameter):
        length = len(team_build_statuses)
        team_p1_and_p2_indices_list = [[i, j + i + 1] for i in range(length) \
                                                      for j in range(length - i - 1)]
        fighter_select_indices = {}
        fighter_select_status_with_team_index = {}

        for indices in team_p1_and_p2_indices_list:
            p1_team_index = indices[0]
            if p1_team_index not in fighter_select_indices:
                fighter_select_indices[p1_team_index] = {}

            if p1_team_index not in fighter_select_status_with_team_index:
                fighter_select_status_with_team_index[p1_team_index] = {}

            p2_team_index = indices[1]
            assert p2_team_index not in fighter_select_indices[p1_team_index]
            assert p2_team_index not in fighter_select_status_with_team_index[p1_team_index]
            assert p1_team_index != p2_team_index

            p1_fighter_select_status = team_build_statuses[p1_team_index][-1]
            p2_fighter_select_status = team_build_statuses[p2_team_index][-1]

            p1_fighter_select_status = input_p2_select_phase_features(p1_fighter_select_status, p2_fighter_select_status)
            p2_fighter_select_status = input_p2_select_phase_features(p2_fighter_select_status, p1_fighter_select_status)

            fighter_select_status_with_team_index[p1_team_index][p2_team_index] = \
                [p1_fighter_select_status, p2_fighter_select_status]

            p1_select_indices = self.fighter_select_indices(p1_fighter_select_status, p1_temperature_parameter)
            p2_select_indices = self.fighter_select_indices(p2_fighter_select_status, p2_temperature_parameter)
            fighter_select_indices[p1_team_index][p2_team_index] = [p1_select_indices, p2_select_indices]
        return fighter_select_indices, fighter_select_status_with_team_index

    def mini_batch_learning(self, input_status, target_value, batch_size, epoch, keep):
        input_status_length = len(input_status)
        assert len(target_value) == input_status_length
        iter_num = (input_status_length // batch_size) * epoch

        for _ in range(iter_num):
            indices = np.random.choice(input_status_length, batch_size)
            self.run_train(input_status[indices], target_value[indices], keep)

def make_team_teacher_data(team_build_statuses, brute_force_win_rates):
    teacher_input_status = np.array([input_status for team_build_status in team_build_statuses \
                                     for input_status in team_build_status])

    target_value = np.array([[parrot.sigmoid_to_tanh(brute_force_win_rates[i])] \
                            for i, team_build_status in enumerate(team_build_statuses) \
                            for _ in team_build_status])
    return teacher_input_status, target_value

def make_fighter_select_teacher_data(fighter_select_status_with_team_index, individual_win_values):
    fighter_select_status = []
    fighter_select_target_value = []
    for p1_team_index, p2_team_index_and_fighter_select_status in fighter_select_status_with_team_index.items():
        for p2_team_index, p1_and_p2_fighter_select_status in p2_team_index_and_fighter_select_status.items():
            fighter_select_status.append(p1_and_p2_fighter_select_status[0])
            fighter_select_status.append(p1_and_p2_fighter_select_status[1])
            str_p1_team_index = str(p1_team_index)
            str_p2_team_index = str(p2_team_index)
            p1_target_value = parrot.sigmoid_to_tanh(individual_win_values[str_p1_team_index][str_p2_team_index])
            p2_target_value = parrot.sigmoid_to_tanh(1.0 - individual_win_values[str_p1_team_index][str_p2_team_index])
            fighter_select_target_value.append([p1_target_value])
            fighter_select_target_value.append([p2_target_value])
    return np.array(fighter_select_status), np.array(fighter_select_target_value)

def make_teacher_data(team_build_statuses, brute_force_win_rates, fighter_select_status_with_team_index, individual_win_values):
    team_teacher_data = make_team_teacher_data(team_build_statuses, brute_force_win_rates)
    fighter_select_teacher_data = make_fighter_select_teacher_data(fighter_select_status_with_team_index, individual_win_values)
    teacher_input_status = np.r_[team_teacher_data[0], fighter_select_teacher_data[0]]
    target_value = np.r_[team_teacher_data[1], fighter_select_teacher_data[1]]
    return {"input_status":teacher_input_status, "target_value":target_value}

if __name__ == "__main__":
    session = tf.Session()
    network = Network(session, 0.0001)
    session.run(tf.global_variables_initializer())
    for i in range(128):
        team_build_status = network.build_team(1, 5)
