import time
import numpy as np
import pybullet as p
import pybullet_data

# ==========================================
# 1. 시뮬레이션 환경 초기화
# ==========================================
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 그림자 및 GUI 렌더링 최적화 설정
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

# ==========================================
# 2. 곡면 Heightfield 지형 데이터 생성
# ==========================================
num_rows = 128
num_cols = 128
length_x = 30.0  # 전장 (m)
length_y = 30.0  # 전폭 (m)

dx = length_x / (num_rows - 1)
dy = length_y / (num_cols - 1)

# 지형 매개변수
amplitude = 1.2      # A: 언덕 최대 높이 (m)
lambda_x = 15.0      # x축 파장 (m)
lambda_y = 15.0      # y축 파장 (m)

x_coords = np.linspace(-length_x / 2.0, length_x / 2.0, num_rows)
y_coords = np.linspace(-length_y / 2.0, length_y / 2.0, num_cols)
X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')

# 해석적 정현파 지형 계산: z = f(x, y)
Z = amplitude * np.sin(2.0 * np.pi * X / lambda_x) * np.cos(2.0 * np.pi * Y / lambda_y)

# 1차원 리스트로 직렬화 (C-order flattening)
heightfield_data = Z.flatten().tolist()

# Heightfield 충돌체 생성
terrain_shape = p.createCollisionShape(
    shapeType=p.GEOM_HEIGHTFIELD,
    meshScale=[dx, dy, 1.0],
    heightfieldTextureScaling=(num_rows - 1) / 2,
    heightfieldData=heightfield_data,
    numHeightfieldRows=num_rows,
    numHeightfieldColumns=num_cols
)

terrain_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=terrain_shape)
p.resetBasePositionAndOrientation(terrain_body, [0, 0, 0], [0, 0, 0, 1])

# 지면 마찰력(Lateral Friction) 증대 (미끄러짐 방지)
p.changeDynamics(terrain_body, -1, lateralFriction=1.2, restitution=0.0)
p.changeVisualShape(terrain_body, -1, rgbaColor=[0.45, 0.65, 0.45, 1.0])

# ==========================================
# 3. 자동차 모델(URDF) 로드 및 구동계 구성
# ==========================================
# 시작 위치 (언덕 중앙 안장점 근처)
car_start_x = 0.0
car_start_y = 0.0
# 안전한 착지를 위해 지형 높이보다 0.4m 위에서 스폰
car_start_z = amplitude * np.sin(2.0 * np.pi * car_start_x / lambda_x) * np.cos(2.0 * np.pi * car_start_y / lambda_y) + 0.4

car_id = p.loadURDF("racecar/racecar.urdf", basePosition=[car_start_x, car_start_y, car_start_z])

# racecar.urdf의 조인트 인덱스 정의
# 조향 링크: 4 (앞좌측 킹핀), 6 (앞우측 킹핀)
# 구동 휠: 2 (후륜 메인 구동축)
# 유동 휠: 3, 5, 7 (베어링 공회전 휠)
steering_joints = [4, 6]
drive_joint = 2
idle_wheels = [3, 5, 7]

# 공회전 휠의 구동력 제로화 (자유 회전 보장)
for wheel in idle_wheels:
    p.setJointMotorControl2(car_id, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

# ==========================================
# 4. 실시간 인터랙티브 GUI 슬라이더 생성
# ==========================================
slider_velocity = p.addUserDebugParameter("Target Velocity (rad/s)", -40.0, 40.0, 15.0)
slider_steering = p.addUserDebugParameter("Steering Angle (rad)", -0.5, 0.5, 0.0)
slider_force = p.addUserDebugParameter("Max Motor Force (N*m)", 0.0, 50.0, 20.0)

# ==========================================
# 5. 주 시뮬레이션 루프
# ==========================================
debug_line_id = -1
text_id = -1

sim_dt = 1.0 / 240.0
p.setTimeStep(sim_dt)

print("시뮬레이션을 시작합니다. GUI의 슬라이더를 통해 속도와 조향각을 제어하세요.")

while True:
    # 1) 슬라이더 제어 입력 취득
    target_v = p.readUserDebugParameter(slider_velocity)
    steer_angle = p.readUserDebugParameter(slider_steering)
    max_f = p.readUserDebugParameter(slider_force)

    # 2) 모터 제어 명령 인가 (전자제어 액추에이터 구동)
    # 조향 제어: Position Control
    for joint in steering_joints:
        p.setJointMotorControl2(car_id, joint, controlMode=p.POSITION_CONTROL, targetPosition=steer_angle)

    # 주행 제어: Velocity Control (인가 토크 상한 설정)
    p.setJointMotorControl2(car_id, drive_joint, controlMode=p.VELOCITY_CONTROL, 
                            targetVelocity=target_v, force=max_f)

    # 3) 물리 엔진 시간 적분 갱신
    p.stepSimulation()

    # 4) 차량 상태 계측
    car_pos, car_orn = p.getBasePositionAndOrientation(car_id)
    car_vel, car_ang_vel = p.getBaseVelocity(car_id)
    lin_speed = np.linalg.norm(car_vel)

    # 5) 지형의 해석적 편도함수 및 법선 벡터 연산
    px, py, pz = car_pos[0], car_pos[1], car_pos[2]
    dz_dx = amplitude * (2.0 * np.pi / lambda_x) * np.cos(2.0 * np.pi * px / lambda_x) * np.cos(2.0 * np.pi * py / lambda_y)
    dz_dy = -amplitude * (2.0 * np.pi / lambda_y) * np.sin(2.0 * np.pi * px / lambda_x) * np.sin(2.0 * np.pi * py / lambda_y)

    norm_mag = np.sqrt(1.0 + dz_dx**2 + dz_dy**2)
    normal_vector = np.array([-dz_dx, -dz_dy, 1.0]) / norm_mag

    # 경사각 계산 (deg)
    slope_angle_deg = np.degrees(np.arccos(normal_vector[2]))

    # 6) 시각화: 지형 법선 벡터를 3D 공간에 녹색 선으로 표시
    line_start = [px, py, pz]
    line_end = [px + normal_vector[0] * 1.5, py + normal_vector[1] * 1.5, pz + normal_vector[2] * 1.5]
    debug_line_id = p.addUserDebugLine(line_start, line_end, lineColorRGB=[0, 1, 0], lineWidth=3, 
                                       lifeTime=sim_dt * 2, replaceItemUniqueId=debug_line_id)

    # 7) 실시간 텔레메트리 텍스트 HUD 출력
    hud_text = f"Speed: {lin_speed:4.2f} m/s | Slope: {slope_angle_deg:4.1f} deg"
    text_id = p.addUserDebugText(hud_text, [px, py, pz + 0.6], textColorRGB=[1, 1, 0], textSize=1.1,
                                 lifeTime=sim_dt * 2, replaceItemUniqueId=text_id)

    # 8) 3인칭 동적 추적 카메라(Chase Camera) 업데이트
    euler = p.getEulerFromQuaternion(car_orn)
    yaw_deg = np.degrees(euler[2]) + 90.0
    p.resetDebugVisualizerCamera(
        cameraDistance=2.8,
        cameraPitch=-25.0,
        cameraYaw=yaw_deg,
        cameraTargetPosition=car_pos
    )

    time.sleep(sim_dt)
