import csv
import math
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

# OBJ 없이 35초 주행을 목표로 하는 큰 자동차 연습용 시뮬레이션. 실제 HW2 제출 조건의 대체물은 아님.
DT = 1 / 240
DRIVE_END = 35.0
DISPLAY_END = 40.0
SIZE = 100.0
N = 201
A = 1.8
WAVELENGTH = 32.0
START = np.array([3.0, 3.0])
GOAL = np.array([48.0, 48.0])  # 시작 전 지정한 목표점
CAR_LENGTH = 1.5  # racecar URDF 실제 크기는 별도 확인 필요


def height(x, y):
    return A * math.sin(2 * math.pi * x / WAVELENGTH) * math.cos(2 * math.pi * y / WAVELENGTH)


def gradient(x, y):
    k = 2 * math.pi / WAVELENGTH
    return (A * k * math.cos(k * x) * math.cos(k * y),
            -A * k * math.sin(k * x) * math.sin(k * y))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def main():
    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(DT)
    p.setPhysicsEngineParameter(numSolverIterations=80)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

    # heightfield 데이터는 y행, x열 순서로 생성
    xs = np.linspace(0, SIZE, N)
    ys = np.linspace(0, SIZE, N)
    zdata = [height(float(x), float(y)) for y in ys for x in xs]
    terrain_shape = p.createCollisionShape(
        p.GEOM_HEIGHTFIELD, meshScale=[SIZE / (N - 1), SIZE / (N - 1), 1],
        heightfieldData=zdata, numHeightfieldRows=N, numHeightfieldColumns=N)
    terrain = p.createMultiBody(0, terrain_shape)
    # Heightfield의 로컬 중심을 실제 0~100 좌표로 옮김
    p.resetBasePositionAndOrientation(terrain, [SIZE / 2, SIZE / 2, 0], [0, 0, 0, 1])
    p.changeDynamics(terrain, -1, lateralFriction=1.0, restitution=0)
    p.changeVisualShape(terrain, -1, rgbaColor=[0.45, 0.65, 0.45, 1])

    # 목표점, 중앙 영역 표시
    corners = [[42, 42], [58, 42], [58, 58], [42, 58]]
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        p.addUserDebugLine([*a, height(*a) + 0.3], [*b, height(*b) + 0.3], [1, 0.55, 0], 3)
    p.addUserDebugText('GOAL', [*GOAL, height(*GOAL) + 1], [1, 0, 0], 1.5)
    p.addUserDebugLine([*START, height(*START) + 0.2], [*GOAL, height(*GOAL) + 0.2], [0, 0.7, 1], 2)

    yaw0 = math.atan2(GOAL[1] - START[1], GOAL[0] - START[0])
    quat = p.getQuaternionFromEuler([0, 0, yaw0])
    car = p.loadURDF('racecar/racecar.urdf', [*START, height(*START) + 1.3], quat, globalScaling=2.5)
    print('JOINTS:', [(i, p.getJointInfo(car, i)[1].decode()) for i in range(p.getNumJoints(car))])
    steering_joints = [4, 6]
    drive_joints = [2, 3, 5, 7]
    for j in drive_joints:
        p.changeDynamics(car, j, lateralFriction=1.0, restitution=0)
        p.setJointMotorControl2(car, j, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

    # 착지 안정화. 준비 시간은 주행시간에서 제외
    for _ in range(720):
        p.stepSimulation()
    pos, _ = p.getBasePositionAndOrientation(car)
    print('READY:', tuple(round(v, 3) for v in pos))

    path = Path('hw2_bigcar_35s_log.csv')
    passed_at = None
    stop_since = None
    stopped_at = None
    last_speed = 0.0
    total_distance = 0.0
    previous_xy = np.array(p.getBasePositionAndOrientation(car)[0][:2])
    rows = []
    debug_id = -1
    max_roll = 0.0

    for step in range(int(DISPLAY_END / DT) + 1):
        t = step * DT
        pos, orn = p.getBasePositionAndOrientation(car)
        vel, _ = p.getBaseVelocity(car)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        xy = np.array(pos[:2])
        speed = float(np.linalg.norm(vel))
        total_distance += float(np.linalg.norm(xy - previous_xy))
        previous_xy = xy.copy()
        max_roll = max(max_roll, abs(math.degrees(roll)))
        distance = float(np.linalg.norm(GOAL - xy))
        heading = math.atan2(GOAL[1] - xy[1], GOAL[0] - xy[0])
        err_heading = wrap(heading - yaw)
        dzdx, dzdy = gradient(xy[0], xy[1])
        slope = math.degrees(math.atan(math.hypot(dzdx, dzdy)))
        steer = float(np.clip(1.6 * err_heading, -0.35, 0.35))

        # 0~4초 가속, 이후 중앙 통과, 31초부터 최종 감속
        # 거리/시간 기반 목표속도. 실제 동역학에 따라 목표 도달 보장 X.
        remaining_time = max(0.1, DRIVE_END - t)
        cruise = min(4.5, distance / remaining_time * 1.15)
        if t < 4:
            cruise = min(cruise, 4.5 * t / 4)
        if t >= 31:
            cruise = min(cruise, max(0.0, distance / remaining_time))
        if t >= DRIVE_END:
            cruise = 0.0
        rule = 'NORMAL'
        if slope > 6:
            cruise = min(cruise, 3.0)
            rule = 'SLOPE_LIMIT'
        if abs(steer) > 0.18:
            cruise = min(cruise, 2.4)
            rule += '+TURN_LIMIT'
        if distance < 0.7:
            cruise = 0.0
            rule = 'GOAL_BRAKE'
        if t >= DRIVE_END:
            rule = 'FINAL_BRAKE'

        # 휠 각속도 제어. 목표속도는 차체 속도가 아닌 휠 명령으로 변환됨.
        wheel_target = cruise / 0.125  # racecar URDF 바퀴 반경 근사
        for j in steering_joints:
            p.setJointMotorControl2(car, j, p.POSITION_CONTROL, targetPosition=steer, force=60)
        for j in drive_joints:
            p.setJointMotorControl2(car, j, p.VELOCITY_CONTROL,
                                    targetVelocity=wheel_target, force=35.0 if cruise else 80.0)

        if 42 <= xy[0] <= 58 and 42 <= xy[1] <= 58 and passed_at is None and t <= 35:
            passed_at = t
            print(f'CENTER FIRST PASS: {t:.2f}s')
        contact_count = sum(bool(p.getContactPoints(car, terrain, linkIndexA=j)) for j in drive_joints)
        if t >= 31 and speed <= 0.2 and contact_count >= 1:
            if stop_since is None:
                stop_since = t
            if stopped_at is None and t - stop_since >= 0.5:
                stopped_at = stop_since
                print(f'STOP CONFIRMED: {stopped_at:.2f}s')
        else:
            stop_since = None

        if step % 24 == 0:
            accel = (speed - last_speed) / (24 * DT) if step else 0
            last_speed = speed
            rows.append([round(t, 3), *[round(v, 3) for v in pos], round(speed, 3),
                         round(accel, 3), round(math.degrees(roll), 2),
                         round(math.degrees(pitch), 2), round(math.degrees(yaw), 2),
                         round(steer, 3), round(slope, 2), round(cruise, 3),
                         round(distance, 3), contact_count, rule, int(passed_at is not None)])
            if step % 240 == 0:
                print(f't={t:4.0f}s pos=({xy[0]:5.1f},{xy[1]:5.1f}) '
                      f'speed={speed:.2f}m/s dist={distance:.1f}m slope={slope:.1f}deg {rule}')
            debug_id = p.addUserDebugText(
                f't={t:.1f}s | v={speed:.2f}m/s | goal={distance:.1f}m | {rule}',
                [pos[0], pos[1], pos[2] + 0.8], [1, 1, 0], 1,
                lifeTime=0.2, replaceItemUniqueId=debug_id)
            p.resetDebugVisualizerCamera(7, math.degrees(yaw) - 90, -35, pos)
        p.stepSimulation()
        time.sleep(DT)

    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['time_s', 'x_m', 'y_m', 'z_m', 'speed_mps', 'accel_mps2',
                         'roll_deg', 'pitch_deg', 'yaw_deg', 'steer_rad', 'slope_deg',
                         'target_speed_mps', 'goal_distance_m', 'wheel_contacts', 'rule', 'center_passed'])
        writer.writerows(rows)
    print('\n=== SUMMARY ===')
    print('center first pass:', passed_at)
    print('confirmed stop start:', stopped_at)
    print('final goal error:', round(distance, 2), 'm')
    print('total distance:', round(total_distance, 2), 'm')
    print('max abs roll:', round(max_roll, 2), 'deg')
    print('CSV:', path.resolve())
    print('NOTE: 실제 35초/중앙통과/정지 조건 충족 여부는 로그 확인 필수')
    p.disconnect()


if __name__ == '__main__':
    main()
