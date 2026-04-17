# Canable 2.0 Actuator Python Controller

A demo project for controlling actuators using the [`robstride-dynamics`](https://pypi.org/project/robstride-dynamics/) and DM Python library over Canable 2.0 USB adapter.

## Hardware Requirements

- **Robstride RS03** or **DM-J4310-2EC** motor with power supply
- **CANable 2.0** USB-to-CAN adapter (candlelight firmware)
- **Raspberry Pi 4** (or any Linux board with USB and SocketCAN support)
- CAN bus wiring (CAN_H, CAN_L) and a common GND connection

## Wiring

| CANable 2.0 | RS03 Motor        |
|--------------|-------------------|
| CAN_H        | CAN_H             |
| CAN_L        | CAN_L             |
| GND          | Power supply GND* |

> \* The RS03 CAN connector only has two pins (H and L). Wire the CANable GND to the **negative terminal of the motor's power supply** to establish a common ground reference. Without this, CAN communication will be unreliable.

### Termination

Enable the **120-ohm termination resistor** on the CANable 2.0 (solder jumper or switch). On a short two-device bus (CANable + RS03), this is required for reliable communication.

## Software Setup

### 1. Bring up the CAN interface

```bash
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
```

Verify the interface is active:

```bash
ip -details link show can0
```

You should see `state UP` and `can state ERROR-ACTIVE` (this is the normal healthy state).

### 2. Install dependencies

```bash
uv add robstride-dynamics
```

Or using `uv`:

```bash
uv sync
```

## Robstride

### Finding your motor ID

If you don't know the motor's CAN ID (factory default is typically `1`), scan the bus:

```bash
python -c "from robstride_dynamics import RobstrideBus; print(RobstrideBus.scan_channel('can0'))"
```

This probes Robstride actuator IDs 1--254 and reports which ones respond.

### Changing the motor ID

To assign a new CAN bus ID to a motor, connect with its current ID and call `write_id()`:

```python
from robstride_dynamics import RobstrideBus, Motor

bus = RobstrideBus("can0", {"rs03": Motor(id=1, model="rs-03")})
bus.connect()
bus.write_id("rs03", new_id=127)
bus.disconnect()
```

The motor will respond to the new ID immediately. Update `MOTOR_ID` in `main.py` to match.

### Homing to zero position

Use `zero_pos.py` to slowly move one or more motors back to their zero position:

```bash
# Single motor
python robstride/homing.py 1

# Multiple motors
python robstride/homing.py 1 2 3

# Custom speed (default: 0.2 rad/s)
python robstride/homing.py 1 --speed 0.1

# Custom model and gains
python robstride/homing.py 1 2 --model rs-01 --kp 20.0 --kd 3.0
```

| Option      | Default  | Description                       |
|-------------|----------|-----------------------------------|
| `--speed`   | 0.2      | Max homing speed in rad/s         |
| `--model`   | rs-03    | Motor model                       |
| `--kp`      | 30.0     | Position gain (stiffness)         |
| `--kd`      | 5.0      | Damping gain (smoothness)         |
| `--channel` | can0     | CAN channel                       |

The motor ramps slowly from its current position to zero. Press `Ctrl+C` to stop at any time.

#### Shortest-path homing after power cycle

After a power cycle the motor's multi-turn counter resets, so the reported position can be far from the true offset (e.g. +4.711 rad instead of -0.662 rad). To avoid a dangerous long travel, `zero_pos.py` uses shortest-path homing:

- **Start**: actual reported position (e.g. +4.711 rad) -- no jump
- **Goal**: nearest multiple of 2π (e.g. +6.283 rad) -- only ~1.57 rad away
- **Ramp**: smoothly from start toward goal at `--speed`

For example: +4.711 rad → nearest zero is +2π (+6.283), so the motor only travels ~1.57 rad forward instead of 4.71 rad backward. Both reach the same physical zero position.

## RS03 Specifications

| Parameter       | Value             |
|-----------------|-------------------|
| Position range  | +/- 12.57 rad (4pi) |
| Max velocity    | 50 rad/s          |
| Max torque      | 60 Nm             |
| Max Kp          | 5000 Nm/rad       |
| Max Kd          | 100 Nm*s/rad      |
| CAN bitrate     | 1 Mbps            |

## Damiao

### Parameters

- **CAN_ID**（寄存器 8，ESC_ID）— 执行器的接收/命令 ID，即主控端发送指令时使用的 CAN 地址。每个执行器在同一总线上必须有唯一的 CAN_ID，否则会产生冲突。修改后需调用 `store_parameters()` 保存到 Flash，断电后才能生效。
- **MASTER_ID**（寄存器 7，MST_ID）— 执行器的反馈 ID，即执行器回传状态数据时使用的 ID。主控端根据此 ID 识别是哪个执行器的反馈。默认值通常为 `0x00`，多个执行器可以共用同一个 MASTER_ID，但建议根据需要区分。
- **kp**（刚度/比例增益）— 控制电机的"弹簧硬度"。kp 越大，电机到达目标位置的力越强，但太大会导致振荡（抖动）。
- **kd**（阻尼/微分增益）— 控制电机的"阻力"，像刹车一样抑制速度。kd 越大，运动越平滑，但太大会让电机反应迟钝。

### MIT Mode Control Limits

| Parameter | DM-J4310-2EC | DM-J4340P-2EC |
|-----------|-------------|---------------|
| Position  | ±12.5 rad   | ±12.5 rad     |
| Velocity  | ±30 rad/s   | ±8 rad/s      |
| Torque    | ±10 Nm      | ±28 Nm        |
| Kp (max)  | 500         | 500           |
| Kd (max)  | 5.0         | 5.0           |

### `get_register` 注意事项

`damiao-motor` 库的 `get_register()` 方法发送寄存器读取请求后，依赖控制器的后台轮询（`poll_feedback()`）来接收回复。如果没有主动轮询总线，`get_register()` 会超时返回错误。

正确用法：先发送所有读取请求，然后主动轮询总线，最后读取缓存值：

```python
# 1. 发送读取请求
motor.request_register_reading(8)  # CAN_ID
motor.request_register_reading(7)  # MASTER_ID

# 2. 主动轮询总线，等待回复
for _ in range(100):
    controller.poll_feedback()
    time.sleep(0.01)

# 3. 读取缓存值（短超时即可）
can_id = motor.get_register(8, timeout=0.1)
master_id = motor.get_register(7, timeout=0.1)
```

如果直接调用 `get_register()` 而不轮询，寄存器值会显示为 `?`（超时）。

### Scan

扫描 CAN 总线上所有已连接的达妙执行器，显示完整寄存器参数表及 ID 摘要：

```bash
# 扫描 can0，默认探测 ID 0x01-0x1F
python damiao/scan.py

# 指定通道
python damiao/scan.py can1

# 自定义 ID 范围
python damiao/scan.py can0 1 64
```

扫描会自动检测所有型号的执行器（4310、4340P、6006 等），无需指定型号。扫描结果包括：
- 每个执行器的完整寄存器参数表
- **ID 摘要表**：清晰显示每个执行器的 CAN_ID 和 MASTER_ID

### Changing actuator ID

使用交互式脚本修改执行器的 CAN_ID 和 MASTER_ID：

```bash
python damiao/chg_id.py
```

脚本会依次引导你：
1. 选择执行器型号
2. 扫描并列出已连接的执行器
3. 选择要修改的执行器
4. 输入新的 CAN_ID（1-127）
5. 输入新的 MASTER_ID（默认 0x00）
6. 写入寄存器并保存到 Flash
7. 重新扫描验证修改是否生效

> **注意**：修改 CAN_ID 后会立即生效，脚本会自动保存到 Flash，断电后不会丢失。

### Homing to zero position

缓慢将执行器移动回零位：

```bash
# 单个执行器（master_id:can_id 格式）
python damiao/homing.py 0:1

# 多个执行器
python damiao/homing.py 0:1 1:2

# 自定义速度
python damiao/homing.py 0:1 --speed 0.1
```

| Option        | Default | Description               |
|---------------|---------|---------------------------|
| `--speed`     | 0.2     | 最大归零速度 (rad/s)       |
| `--kp`        | 100.0   | 刚度增益 (max 500)         |
| `--kd`        | 3.0     | 阻尼增益 (max 5.0)        |
| `--tolerance` | 0.15    | 位置容差 (rad)             |
| `--channel`   | can0    | CAN 通道                   |

脚本使用最短路径归零：断电重启后多圈计数器会复位，脚本会自动计算到最近 2π 倍数的距离，避免长距离危险运动。

## Troubleshooting

### "No response from the motor"

- **Check power** -- the RS03 needs its own power supply; it does not power through CAN
- **Check wiring** -- verify CAN_H and CAN_L are not swapped
- **Check common GND** -- connect CANable GND to the motor power supply GND
- **Check termination** -- enable the 120-ohm resistor on the CANable 2.0
- **Check motor ID** -- run the scan command above to discover the actual ID
- **Check CAN interface** -- run `ip -details link show can0` and confirm `state UP`
