import numpy as np
import pandas as pd


class DispatchController:
    def __init__(self, simulator):
        self.sim = simulator

        # === 目标水位配置 ===
        self.target_z_min = 553.5
        self.target_z_max = 553.8
        self.target_z_mid = (self.target_z_min + self.target_z_max) / 2

        # 紧急水位 (用于跳过惰性检查，但仍需遵守爬坡约束)
        self.safe_z_min = 553.4
        self.safe_z_max = 553.9

        # === 设备参数 ===
        self.gate_max = 80.0
        self.unit_max_load = 348.0
        self.single_unit_capacity = 58.0

        # === 物理硬约束 (爬坡率) ===
        # 相邻时段(5min)最大变化量
        self.RAMP_LOAD_MAX = 50.0  # MW
        self.RAMP_GATE_MAX = 4.0  # m

        # === 搜索步长 ===
        self.load_step_search = 5.0
        self.gate_step_search = 0.5

        # 预测视界
        self.check_steps = 12  # 1小时

        self.gate_order = [1, 2, 0, 3, 4]

    def distribute_units(self, total_load):
        """均匀分配负荷"""
        loads = [0.0] * 6
        remaining = total_load
        for i in range(6):
            if remaining <= 0: break
            alloc = min(remaining, self.single_unit_capacity)
            loads[i] = alloc
            remaining -= alloc
        return loads

    def distribute_gates(self, total_opening):
        """按顺序分配闸门"""
        gates = [0.0] * 5
        remaining = total_opening

        # 1. 优先避振区 (开度>=2.0)
        full_round = False
        while remaining > 0.001 and not full_round:
            filled_count = 0
            for idx in self.gate_order:
                if remaining <= 0.001: break
                current = gates[idx]
                if current < 2.0:
                    need = 2.0 - current
                    give = min(remaining, need)
                    gates[idx] += give
                    remaining -= give
                else:
                    filled_count += 1
            if filled_count == 5: full_round = True

        # 2. 剩余均匀分配
        if remaining > 0.001:
            while remaining > 0.001:
                for idx in self.gate_order:
                    if remaining <= 0.001: break
                    give = min(remaining, 0.5)
                    gates[idx] += give
                    remaining -= give
        return gates

    def _simulate_trajectory(self, start_z, start_z_down, future_inflows, units, gates, dt):
        """推演未来水位轨迹"""
        trajectory = []
        curr_z = start_z
        curr_z_down = start_z_down
        total_outflow = 0.0

        for i in range(self.check_steps):
            inflow = future_inflows[i] if i < len(future_inflows) else future_inflows[-1]
            res = self.sim.calculate_step(curr_z, curr_z_down, inflow, units, gates, dt)

            curr_z = res['cal_next_z_up']
            curr_z_down = res.get('cal_next_z_down', curr_z_down)

            total_outflow += res['cal_outflow']
            trajectory.append(curr_z)

        return trajectory, total_outflow / self.check_steps

    def get_max_generation_discharge(self, current_z, z_down, dt):
        """计算机组满发流量"""
        max_units = self.distribute_units(self.unit_max_load)
        zero_gates = [0.0] * 5
        res = self.sim.calculate_step(current_z, z_down, 0.0, max_units, zero_gates, dt)
        return res['cal_outflow']

    def find_optimal_plan(self, current_z, z_down, future_inflows, current_total_gate, current_total_load, dt=300):

        # === 1. 惰性检查 (Baseline Check) ===
        # 只有在非紧急水位时才允许保持现状
        curr_units_list = self.distribute_units(current_total_load)
        curr_gates_list = self.distribute_gates(current_total_gate)

        is_emergency = (current_z > self.safe_z_max) or (current_z < self.safe_z_min)

        if not is_emergency:
            traj_baseline, avg_outflow_baseline = self._simulate_trajectory(
                current_z, z_down, future_inflows, curr_units_list, curr_gates_list, dt
            )
            # 如果水位全程都在目标范围内，直接保持不变，节省计算
            if all(self.target_z_min <= z <= self.target_z_max for z in traj_baseline):
                return {
                    'gate_total': current_total_gate,
                    'load_total': current_total_load,
                    'gates': curr_gates_list,
                    'units': curr_units_list,
                    'next_z': traj_baseline[0],
                    'water_rate': avg_outflow_baseline / (current_total_load + 1e-5) if current_total_load > 1 else 0,
                    'dist': 0.0,
                    'outflow': avg_outflow_baseline
                }

        # === 2. 确定硬约束搜索边界 ===
        # 无论策略如何，必须严格遵守爬坡约束

        # 负荷可行域 [Load_min, Load_max]
        search_load_min = max(0.0, current_total_load - self.RAMP_LOAD_MAX)
        search_load_max = min(self.unit_max_load, current_total_load + self.RAMP_LOAD_MAX)

        # 闸门可行域 [Gate_min, Gate_max]
        search_gate_min = max(0.0, current_total_gate - self.RAMP_GATE_MAX)
        search_gate_max = min(self.gate_max, current_total_gate + self.RAMP_GATE_MAX)

        # === 3. 判别调节模式 ===
        max_gen_flow = self.get_max_generation_discharge(current_z, z_down, dt)
        avg_inflow = np.mean(future_inflows[:6])

        schemes = []  # (gate, load)

        # 逻辑判断：入流能否被满发机组吃下？
        # 注意：为了避免频繁切换，如果当前已经在开闸泄洪且入流接近满发流量，保持Mode B

        mode_load_opt = False
        if avg_inflow < max_gen_flow and current_z < self.safe_z_max:
            mode_load_opt = True

        if mode_load_opt:
            # === 模式 A: 负荷调节 (Load Optimization) ===
            # 目标：闸门尽可能关（趋向0），负荷寻找最优值

            # 闸门动作：在可行域内尽可能小。
            # 如果可行域包含0，则设为0；否则设为下限（全速关闭）
            fixed_gate = search_gate_min

            # 负荷搜索：在可行域内搜索
            load_candidates = np.arange(search_load_min, search_load_max + 0.01, self.load_step_search)
            # 确保边界值被包含 (特别是当前值，防止震荡)
            load_candidates = np.unique(
                np.append(load_candidates, [current_total_load, search_load_min, search_load_max]))

            for l in load_candidates:
                schemes.append((fixed_gate, l))

        else:
            # === 模式 B: 闸门调节 (Gate Optimization) ===
            # 目标：负荷尽可能大（趋向满发），闸门寻找最优值

            # 负荷动作：在可行域内尽可能大。
            # 如果可行域包含满发，则设为满发；否则设为上限（全速加载）
            fixed_load = search_load_max

            # 闸门搜索：在可行域内搜索
            gate_candidates = np.arange(search_gate_min, search_gate_max + 0.01, self.gate_step_search)
            gate_candidates = np.unique(
                np.append(gate_candidates, [current_total_gate, search_gate_min, search_gate_max]))

            for g in gate_candidates:
                schemes.append((g, fixed_load))

        # === 4. 预测与择优 ===
        valid_schemes = []

        for g, l in schemes:
            g_list = self.distribute_gates(g)
            l_list = self.distribute_units(l)

            # 只需要算未来的轨迹
            traj, avg_outflow = self._simulate_trajectory(
                current_z, z_down, future_inflows, l_list, g_list, dt
            )

            # 罚函数
            violation = 0.0
            for z in traj:
                if z > self.target_z_max:
                    violation += (z - self.target_z_max) * 10
                elif z < self.target_z_min:
                    violation += (self.target_z_min - z) * 10

            dist_mid = abs(traj[-1] - self.target_z_mid)

            # 稳定性惩罚：即使在允许范围内，也不要无意义地乱动
            move_cost = abs(l - current_total_load) * 0.001 + abs(g - current_total_gate) * 0.01

            res = {
                'gate_total': g, 'load_total': l,
                'gates': g_list, 'units': l_list,
                'next_z': traj[0],
                'violation': violation,
                'score': violation * 1000 + dist_mid + move_cost,
                'outflow': avg_outflow
            }
            valid_schemes.append(res)

        # 选分值最小的
        valid_schemes.sort(key=lambda x: x['score'])

        best = valid_schemes[0]
        water_rate = best['outflow'] / (best['load_total'] + 1e-5) if best['load_total'] > 1 else 0

        return {
            'gate_total': best['gate_total'],
            'load_total': best['load_total'],
            'gates': best['gates'],
            'units': best['units'],
            'next_z': best['next_z'],
            'water_rate': water_rate,
            'dist': 0.0,  # 仅作占位
            'outflow': best['outflow']
        }
