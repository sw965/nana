import json
import boa.src as boa
import nana.team_and_fighter_select_value_network as tafsvn

class LocalSocketServer:
    def __init__(self, port, network, total_step, one_step_team_build_num):
        self.server = boa.SimpleLocalSocketServer(port)
        self.network = network
        self.server.send(str(total_step))
        assert self.server.recv(128) == "totalStep"
        self.server.send(str(one_step_team_build_num))
        assert self.server.recv(128) == "oneStepTeamBuildNum"

    def recv(self, byte_size):
        return self.server.recv(byte_size)

    def send(self, msg):
        self.server.send(msg)

    def send_json_brute_force_battle_fighter_select_indices(self, team_build_statuses, p1_temperature_parameter):
        data, fighter_select_status_with_team_index = \
            self.network.brute_force_battle_fighter_select_indices(team_build_statuses, p1_temperature_parameter)
        json_data = json.dumps(data, ensure_ascii=False, indent=4)
        self.send(json_data)
        assert self.recv(512) == "jsonBruteForceBattleFighterSelectIndices"
        return fighter_select_status_with_team_index

    def send_json_team(self, input_status):
        json_team = tafsvn.input_status_to_json_team(input_status)
        self.send(json_team)
        assert self.recv(128) == "jsonTeam"

    def recv_json_to_dict(self, byte_size):
        json_data = self.recv(byte_size)
        self.send("json_dict")
        return json.loads(json_data)

    def recv_brute_force_win_rates(self, size):
        result = []
        for _ in range(size):
            msg = self.server.recv(512)[:7]
            self.send("brute_force_win_rates")
            target_value = float(msg)
            result.append(target_value)
        return result
