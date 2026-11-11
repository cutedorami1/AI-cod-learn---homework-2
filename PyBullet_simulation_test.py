
import csv
import math
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data


# =========================================================
# 1. 기본 설정
# =========================================================

DT = 1.0 / 120.0

DRIVE_END = 35.0
DISPLAY_END = 40.0

SIZE = 100.0
N = 201

A = 1.8
WAVELENGTH = 32.0

START = np.array([3.0, 3.0], dtype=float)
GOAL = np.array([48.0, 48.0], dtype=float)

CAR_SCALE = 2.5

# 부드러운 속도 및 조향 제어
MAX_ACCEL = 1.5
MAX_DECEL = 2.5
MAX_STEER_RATE = 0.65

MAX_SPEED = 4.5
MAX_STEER = 0.35

WHEEL_RADIUS = 0.125

# 카메라 갱신: 120 / 3 = 40Hz
CAMERA_INTERVAL = 3
CAMERA_SMOOTH = 0.16
CAMERA_YAW_SMOOTH = 0.12

# CSV 기록: 120 / 12 = 10Hz
LOG_INTERVAL = 12

# 콘솔 출력: 1초마다
PRINT_INTERVAL = 120


# =========================================================
# 2. 곡면 지형 함수
# =========================================================

def height(x, y):
    k = 2.0 * math.pi / WAVELENGTH
    return A * math.sin(k * x) * math.cos(k * y)


def gradient(x, y):
    k = 2.0 * math.pi / WAVELENGTH

    dzdx = A * k * math.cos(k * x) * math.cos(k * y)
    dzdy = -A * k * math.sin(k * x) * math.sin(k * y)

    return dzdx, dzdy


def wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def move_toward(current, target, max_delta):
    return current + float(
        np.clip(target - current, -max_delta, max_delta)
    )


# =========================================================
# 3. 시뮬레이션 시작
# =========================================================

def main():

    cid = p.connect(p.GUI)

    if cid < 0:
        raise RuntimeError("PyBullet GUI 연결 실패")

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.setGravity(0, 0, -9.81)
    p.setTimeStep(DT)

    p.setPhysicsEngineParameter(
        numSolverIterations=60,
        fixedTimeStep=DT
    )

    p.configureDebugVisualizer(
        p.COV_ENABLE_SHADOWS, 0
    )

    p.configureDebugVisualizer(
        p.COV_ENABLE_GUI, 0
    )

    # =====================================================
    # 4. 곡면 Heightfield 생성
    # =====================================================

    xs = np.linspace(0, SIZE, N)
    ys = np.linspace(0, SIZE, N)

    zdata = [
        height(float(x), float(y))
        for y in ys
        for x in xs
    ]

    terrain_shape = p.createCollisionShape(
        shapeType=p.GEOM_HEIGHTFIELD,
        meshScale=[
            SIZE / (N - 1),
            SIZE / (N - 1),
            1.0
        ],
        heightfieldData=zdata,
        numHeightfieldRows=N,
        numHeightfieldColumns=N
    )

    terrain = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=terrain_shape
    )

    p.resetBasePositionAndOrientation(
        terrain,
        [SIZE / 2, SIZE / 2, 0],
        [0, 0, 0, 1]
    )

    p.changeDynamics(
        terrain,
        -1,
        lateralFriction=1.3,
        restitution=0.0
    )

    p.changeVisualShape(
        terrain,
        -1,
        rgbaColor=[0.40, 0.64, 0.42, 1]
    )

    # =====================================================
    # 5. 중앙 영역 및 목표점 표시
    # =====================================================

    corners = [
        [42, 42],
        [58, 42],
        [58, 58],
        [42, 58]
    ]

    for i in range(4):

        a = corners[i]
        b = corners[(i + 1) % 4]

        p.addUserDebugLine(
            [a[0], a[1], height(*a) + 0.3],
            [b[0], b[1], height(*b) + 0.3],
            [1, 0.55, 0],
            3
        )

    p.addUserDebugText(
        "GOAL",
        [GOAL[0], GOAL[1], height(*GOAL) + 1],
        [1, 0, 0],
        1.5
    )

    p.addUserDebugLine(
        [START[0], START[1], height(*START) + 0.3],
        [GOAL[0], GOAL[1], height(*GOAL) + 0.3],
        [0, 0.7, 1],
        2
    )

    # =====================================================
    # 6. 큰 자동차 생성
    # =====================================================

    yaw0 = math.atan2(
        GOAL[1] - START[1],
        GOAL[0] - START[0]
    )

    quat = p.getQuaternionFromEuler(
        [0, 0, yaw0]
    )

    car = p.loadURDF(
        "racecar/racecar.urdf",
        [
            START[0],
            START[1],
            height(*START) + 1.3
        ],
        quat,
        globalScaling=CAR_SCALE
    )

    steering_joints = [4, 6]
    drive_joints = [2, 3, 5, 7]

    print("\n=== CAR JOINTS ===")

    for i in range(p.getNumJoints(car)):
        print(
            i,
            p.getJointInfo(car, i)[1].decode()
        )

    # 자동차 바퀴 접지력
    for j in drive_joints:

        p.changeDynamics(
            car,
            j,
            lateralFriction=1.4,
            restitution=0.0
        )

        p.setJointMotorControl2(
            car,
            j,
            p.VELOCITY_CONTROL,
            targetVelocity=0,
            force=0
        )

    # =====================================================
    # 7. 자동차 착지 안정화
    # =====================================================

    print("\n자동차 착지 안정화 중...")

    for _ in range(360):
        p.stepSimulation()

    pos, orn = p.getBasePositionAndOrientation(car)

    print(
        "READY:",
        tuple(round(v, 3) for v in pos)
    )

    # =====================================================
    # 8. 주행 변수 초기화
    # =====================================================

    smooth_steer = 0.0
    smooth_speed = 0.0

    camera_target = np.array(pos, dtype=float)
    camera_yaw = math.degrees(yaw0) - 90.0

    passed_at = None
    stopped_at = None
    stop_since = None

    max_roll = 0.0
    total_distance = 0.0

    last_speed = 0.0

    previous_xy = np.array(pos[:2], dtype=float)

    rows = []

    debug_id = -1

    path = Path("hw2_bigcar_smooth_log.csv")

    print("\n=== SIMULATION START ===")

    # =====================================================
    # 9. 주행 반복
    # =====================================================

    for step in range(int(DISPLAY_END / DT) + 1):

        loop_start = time.perf_counter()

        t = step * DT

        pos, orn = p.getBasePositionAndOrientation(car)
        vel, _ = p.getBaseVelocity(car)

        roll, pitch, yaw = p.getEulerFromQuaternion(orn)

        xy = np.array(pos[:2], dtype=float)

        speed = float(np.linalg.norm(vel))

        total_distance += float(
            np.linalg.norm(xy - previous_xy)
        )

        previous_xy = xy.copy()

        max_roll = max(
            max_roll,
            abs(math.degrees(roll))
        )

        distance = float(
            np.linalg.norm(GOAL - xy)
        )

        # =================================================
        # 10. 목표 방향 계산
        # =================================================

        heading = math.atan2(
            GOAL[1] - xy[1],
            GOAL[0] - xy[0]
        )

        err_heading = wrap(heading - yaw)

        target_steer = float(
            np.clip(
                1.6 * err_heading,
                -MAX_STEER,
                MAX_STEER
            )
        )

        # 조향각 변화량 제한
        smooth_steer = move_toward(
            smooth_steer,
            target_steer,
            MAX_STEER_RATE * DT
        )

        # =================================================
        # 11. 경사각 계산
        # =================================================

        dzdx, dzdy = gradient(
            xy[0],
            xy[1]
        )

        slope = math.degrees(
            math.atan(
                math.hypot(dzdx, dzdy)
            )
        )

        # =================================================
        # 12. 목표속도 계산
        # =================================================

        remaining_time = max(
            0.1,
            DRIVE_END - t
        )

        cruise = min(
            MAX_SPEED,
            distance / remaining_time * 1.15
        )

        # 출발 시 서서히 가속
        if t < 4.0:
            cruise = min(
                cruise,
                MAX_SPEED * t / 4.0
            )

        # 31초 이후 감속
        if t >= 31.0:
            cruise = min(
                cruise,
                distance / remaining_time
            )

        rule = "NORMAL"

        # 경사 제한
        if slope > 6.0:

            cruise = min(
                cruise,
                3.0
            )

            rule = "SLOPE_LIMIT"

        # 회전 제한
        if abs(target_steer) > 0.18:

            cruise = min(
                cruise,
                2.4
            )

            rule += "+TURN_LIMIT"

        # 목표점 근처 감속
        if distance < 0.7:

            cruise = 0.0
            rule = "GOAL_BRAKE"

        # 35초 이후 정지 명령
        if t >= DRIVE_END:

            cruise = 0.0
            rule = "FINAL_BRAKE"

        # =================================================
        # 13. 부드러운 가속 및 감속
        # =================================================

        if cruise > smooth_speed:
            max_change = MAX_ACCEL * DT
        else:
            max_change = MAX_DECEL * DT

        smooth_speed = move_toward(
            smooth_speed,
            cruise,
            max_change
        )

        smooth_speed = max(
            0.0,
            smooth_speed
        )

        # =================================================
        # 14. 자동차 바퀴 제어
        # =================================================

        wheel_target = (
            smooth_speed / WHEEL_RADIUS
        )

        for j in steering_joints:

            p.setJointMotorControl2(
                car,
                j,
                p.POSITION_CONTROL,
                targetPosition=float(smooth_steer),
                force=60
            )

        for j in drive_joints:

            p.setJointMotorControl2(
                car,
                j,
                p.VELOCITY_CONTROL,
                targetVelocity=float(wheel_target),
                force=35.0 if smooth_speed > 0.01 else 80.0
            )

        # =================================================
        # 15. 중앙 영역 통과 판정
        # =================================================

        if (
            42 <= xy[0] <= 58
            and 42 <= xy[1] <= 58
            and passed_at is None
            and t <= DRIVE_END
        ):

            passed_at = t

            print(
                f"CENTER FIRST PASS: {t:.2f}s"
            )

        # =================================================
        # 16. 바퀴 접지 및 정지 판정
        # =================================================

        contact_count = sum(
            bool(
                p.getContactPoints(
                    car,
                    terrain,
                    linkIndexA=j
                )
            )
            for j in drive_joints
        )

        if (
            t >= 31.0
            and speed <= 0.2
            and contact_count >= 1
        ):

            if stop_since is None:
                stop_since = t

            if (
                stopped_at is None
                and t - stop_since >= 0.5
            ):

                stopped_at = stop_since

                print(
                    f"STOP CONFIRMED: {stopped_at:.2f}s"
                )

        else:
            stop_since = None

        # =================================================
        # 17. CSV 데이터 기록
        # =================================================

        if step % LOG_INTERVAL == 0:

            accel = (
                (speed - last_speed)
                / (LOG_INTERVAL * DT)
                if step else 0.0
            )

            last_speed = speed

            rows.append([
                round(t, 3),
                round(pos[0], 3),
                round(pos[1], 3),
                round(pos[2], 3),
                round(speed, 3),
                round(accel, 3),
                round(math.degrees(roll), 2),
                round(math.degrees(pitch), 2),
                round(math.degrees(yaw), 2),
                round(smooth_steer, 3),
                round(slope, 2),
                round(smooth_speed, 3),
                round(distance, 3),
                contact_count,
                rule,
                int(passed_at is not None)
            ])

        # =================================================
        # 18. 콘솔 출력
        # =================================================

        if step % PRINT_INTERVAL == 0:

            print(
                f"t={t:5.1f}s "
                f"pos=({xy[0]:6.2f},{xy[1]:6.2f}) "
                f"speed={speed:.2f}m/s "
                f"dist={distance:.2f}m "
                f"slope={slope:.1f}deg "
                f"{rule}"
            )

        # =================================================
        # 19. 부드러운 카메라 추적
        # =================================================

        if step % CAMERA_INTERVAL == 0:

            camera_target += CAMERA_SMOOTH * (
                np.array(pos) - camera_target
            )

            desired_yaw = (
                math.degrees(yaw) - 90.0
            )

            yaw_difference = (
                (desired_yaw - camera_yaw + 180)
                % 360 - 180
            )

            camera_yaw += (
                CAMERA_YAW_SMOOTH * yaw_difference
            )

            p.resetDebugVisualizerCamera(
                cameraDistance=9,
                cameraYaw=camera_yaw,
                cameraPitch=-35,
                cameraTargetPosition=camera_target.tolist()
            )

        # =================================================
        # 20. 상태 텍스트
        # =================================================

        if step % 24 == 0:

            debug_id = p.addUserDebugText(
                (
                    f"Time: {t:.1f}s | "
                    f"Speed: {speed:.2f}m/s | "
                    f"Goal: {distance:.1f}m | "
                    f"{rule}"
                ),
                [
                    pos[0],
                    pos[1],
                    pos[2] + 1.0
                ],
                [1, 1, 0],
                1.2,
                lifeTime=0.3,
                replaceItemUniqueId=debug_id
            )

        # =================================================
        # 21. 물리 계산 및 시간 동기화
        # =================================================

        p.stepSimulation()

        elapsed = time.perf_counter() - loop_start

        sleep_time = DT - elapsed

        if sleep_time > 0:
            time.sleep(sleep_time)

    # =====================================================
    # 22. CSV 파일 저장
    # =====================================================

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "time_s",
            "x_m",
            "y_m",
            "z_m",
            "speed_mps",
            "accel_mps2",
            "roll_deg",
            "pitch_deg",
            "yaw_deg",
            "steer_rad",
            "slope_deg",
            "target_speed_mps",
            "goal_distance_m",
            "wheel_contacts",
            "rule",
            "center_passed"
        ])

        writer.writerows(rows)

    # =====================================================
    # 23. 최종 결과
    # =====================================================

    final_pos, _ = p.getBasePositionAndOrientation(car)

    final_vel, _ = p.getBaseVelocity(car)

    final_distance = float(
        np.linalg.norm(
            GOAL - np.array(final_pos[:2])
        )
    )

    final_speed = float(
        np.linalg.norm(final_vel)
    )

    print("\n================================")
    print("       SIMULATION SUMMARY")
    print("================================")

    print("Center first pass:", passed_at)
    print("Confirmed stop start:", stopped_at)

    print(
        "Final goal error:",
        round(final_distance, 2),
        "m"
    )

    print(
        "Final speed:",
        round(final_speed, 3),
        "m/s"
    )

    print(
        "Total distance:",
        round(total_distance, 2),
        "m"
    )

    print(
        "Max abs roll:",
        round(max_roll, 2),
        "deg"
    )

    print("CSV:", path.resolve())

    print("\n=== CONDITION CHECK ===")

    print(
        "Center passed:",
        passed_at is not None
    )

    print(
        "Stop confirmed:",
        stopped_at is not None
    )

    print(
        "Goal within 1m:",
        final_distance <= 1.0
    )

    print(
        "Final speed <= 0.2m/s:",
        final_speed <= 0.2
    )

    print(
        "\nNOTE: 목표 도달 및 정지 조건은 "
        "실제 실행 결과를 확인해야 합니다."
    )

    print("\n시뮬레이션 종료")

    p.disconnect()


if __name__ == "__main__":
    main()
