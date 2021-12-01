import copy
import json
import random
import numpy as np
import tensorflow as tf
import boa.src as boa
import crow.src as crow
import nana.team_and_fighter_select_value_network as tafsvn
import nana.local_socket_server as lss

def main():
    session = tf.Session()
    network = tafsvn.Network(session, 1 / 1000)
    session.run(tf.global_variables_initializer())

    total_step = 1280
    one_step_team_build_num = 16
    assert one_step_team_build_num > 1
    high_win_rate_sample_num = 8

    print("クライアントの接続待ちだぞ(はぁと")
    server = lss.LocalSocketServer(7777, network, total_step, one_step_team_build_num)
    print("クライアントの接続完了だぞ(はぁと")

    high_win_rate_team_build_statuses = []
    test_team_build_statuses = [network.build_team(0, 0), network.build_team(0, 0)]

    for i in range(total_step):
        team_build_statuses = copy.deepcopy(high_win_rate_team_build_statuses)
        team_build_statuses += [network.build_team(0, 0), network.build_team(-1, -1)]
        team_build_statuses += network.build_teams(one_step_team_build_num - len(team_build_statuses), 1, 1)

        fighter_select_status_with_team_index = \
            server.send_json_brute_force_battle_fighter_select_indices(team_build_statuses, 0, 0)

        for team_build_status in team_build_statuses:
            server.send_json_team(team_build_status[-1])

        individual_win_values = server.recv_json_to_dict(9600)
        brute_force_win_rates = server.recv_brute_force_win_rates(one_step_team_build_num)

        teacher_data = tafsvn.make_teacher_data(team_build_statuses, brute_force_win_rates,
                                                fighter_select_status_with_team_index, individual_win_values)

        network.mini_batch_learning(teacher_data["input_status"], teacher_data["target_value"],
                                    batch_size=64, epoch=1, keep=0.9)
        network.save("C:/Python35/pyckage/nana/team_and_fighter_select_value_model/", str(i) + "step")

        del high_win_rate_team_build_statuses[:]
        high_win_rate_order = np.argsort(brute_force_win_rates)[::-1]
        for index in high_win_rate_order[:high_win_rate_sample_num]:
            high_win_rate_team_build_statuses.append(team_build_statuses[index])

        del test_team_build_statuses[0]
        test_team_build_statuses.append(network.build_team(0, 0))
        server.send_json_brute_force_battle_fighter_select_indices(test_team_build_statuses, 0, 0)
        for test_team_build_status in test_team_build_statuses:
            server.send_json_team(test_team_build_status[-1])
        boa.dump_pickle(tafsvn.input_status_to_json_team(test_team_build_statuses[0][-1]),
                        "C:/Python35/pyckage/nana/build_team_log/" + str(i) + ".pkl")

        for key, value in json.loads(tafsvn.input_status_to_json_team(test_team_build_statuses[0][-1])).items():
            print(key, value)

        print("i = \n", i)

if __name__ == "__main__":
    main()
