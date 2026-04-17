import time
import struct
import argparse
import serial
import serial.tools.list_ports
from serial import EIGHTBITS, PARITY_NONE, STOPBITS_ONE

# Macro definition parameters
PI = 3.1415926
FRAME_HEAD = str('fc')
FRAME_END = str('fd')
TYPE_IMU = str('40')
TYPE_AHRS = str('41')
TYPE_INSGPS = str('42')
TYPE_GEODETIC_POS = str('5c')
TYPE_GROUND = str('f0')
TYPE_SYS_STATE = str('50')
TYPE_BODY_ACCELERATION = str('62')
TYPE_ACCELERATION = str('61')
TYPE_MSG_BODY_VEL = str('60')
IMU_LEN = str('38')  # //56
AHRS_LEN = str('30')  # //48
INSGPS_LEN = str('48')  # //72
GEODETIC_POS_LEN = str('20')  # //32
SYS_STATE_LEN = str('64')  # // 100
BODY_ACCELERATION_LEN = str('10') #// 16
ACCELERATION_LEN = str('0c')  # 12
PI = 3.141592653589793
DEG_TO_RAD = 0.017453292519943295
isrun = True


# Get command line input parameters
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, default='/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0', help='Serial port receive data')
    parser.add_argument('--bps', type=int, default=921600, help='the models baud rate set; default: 921600')
    parser.add_argument('--timeout', type=int, default=20, help='set the serial port timeout; default: 20')
    parser.add_argument('--rate', type=float, default=10.0, help='output rate in Hz; default: 10.0')

    receive_params = parser.parse_known_args()[0] if known else parser.parse_args()
    return receive_params


def dump_imu_data(port='/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0', baudrate=921600, timeout=1, output_rate=10.0):
    try:
        serial_ = serial.Serial(port=port, baudrate=baudrate, bytesize=EIGHTBITS, parity=PARITY_NONE, stopbits=STOPBITS_ONE, timeout=timeout)
        print(f"Connected to IMU on port: {port}")
        print(f"Baud rate: {serial_.baudrate}")
        print(f"Output rate: {output_rate} Hz")
        print("=" * 80)
        print("Starting IMU data dump... (Press Ctrl+C to stop)")
        print("=" * 80)
    except Exception as e:
        print(f"Error: Unable to open port {port}. {e}")
        exit(1)

    temp1 = False
    temp2 = False
    last_output_time = 0
    output_interval = 1.0 / output_rate

    result = {
        "Accelerometer_X": 0,
        "Accelerometer_Y": 0,
        "Accelerometer_Z": 0,
        "RollSpeed": 0,
        "PitchSpeed": 0,
        "HeadingSpeed": 0,
        "Roll": 0,
        "Pitch": 0,
        "Heading": 0,
        "qw": 0,
        "qx": 0,
        "qy": 0,
        "qz": 0,
    }

    try:
        while serial_.isOpen():
            check_head = serial_.read().hex()
            # Verify frame header
            if check_head != FRAME_HEAD:
                continue
            head_type = serial_.read().hex()
            # Verify data type
            if (head_type != TYPE_IMU and head_type != TYPE_AHRS and head_type != TYPE_INSGPS and
                    head_type != TYPE_GEODETIC_POS and head_type != 0x50 and head_type != TYPE_GROUND and
                    head_type != TYPE_SYS_STATE and  head_type!=TYPE_MSG_BODY_VEL and head_type!=TYPE_BODY_ACCELERATION and head_type!=TYPE_ACCELERATION):
                continue
            check_len = serial_.read().hex()
            # Verify data type length
            if head_type == TYPE_IMU and check_len != IMU_LEN:
                continue
            elif head_type == TYPE_AHRS and check_len != AHRS_LEN:
                continue
            elif head_type == TYPE_INSGPS and check_len != INSGPS_LEN:
                continue
            elif head_type == TYPE_GEODETIC_POS and check_len != GEODETIC_POS_LEN:
                continue
            elif head_type == TYPE_SYS_STATE and check_len != SYS_STATE_LEN:
                continue
            elif head_type == TYPE_GROUND or head_type == 0x50:
                continue
            elif head_type == TYPE_MSG_BODY_VEL and check_len != ACCELERATION_LEN:
                print("check head type "+str(TYPE_MSG_BODY_VEL)+" failed;"+" check_LEN:"+str(check_len))
                continue
            elif head_type == TYPE_BODY_ACCELERATION and check_len != BODY_ACCELERATION_LEN:
                print("check head type "+str(TYPE_BODY_ACCELERATION)+" failed;"+" check_LEN:"+str(check_len))
                continue
            elif head_type == TYPE_ACCELERATION and check_len != ACCELERATION_LEN:
                print("check head type "+str(TYPE_ACCELERATION)+" failed;"+" ckeck_LEN:"+str(check_len))
                continue
            check_sn = serial_.read().hex()
            head_crc8 = serial_.read().hex()
            crc16_H_s = serial_.read().hex()
            crc16_L_s = serial_.read().hex()

            # Read and parse IMU data
            if head_type == TYPE_IMU:
                data_s = serial_.read(int(IMU_LEN, 16))
                IMU_DATA = struct.unpack('12f ii',data_s[0:56])
                result["Accelerometer_X"] = IMU_DATA[3]
                result["Accelerometer_Y"] = IMU_DATA[4]
                result["Accelerometer_Z"] = IMU_DATA[5]
                temp1 = True

            # Read and parse AHRS data
            elif head_type == TYPE_AHRS:
                data_s = serial_.read(int(AHRS_LEN, 16))
                AHRS_DATA = struct.unpack('10f ii',data_s[0:48])
                # Coordinate transformation
                result["RollSpeed"] = AHRS_DATA[1]
                result["PitchSpeed"] = AHRS_DATA[0] * -1
                result["HeadingSpeed"] = AHRS_DATA[2]
                r = AHRS_DATA[4]
                p = AHRS_DATA[3] * -1
                h = AHRS_DATA[5]
                result["Roll"] = r
                result["Pitch"] = p
                result["Heading"] = h
                result["qw"] = AHRS_DATA[6]
                result["qx"] = AHRS_DATA[7]
                result["qy"] = AHRS_DATA[8]
                result["qz"] = AHRS_DATA[9]
                temp2 = True

            # When complete IMU and AHRS data is collected, output to terminal
            if temp1 and temp2:
                current_time = time.time()
                if current_time - last_output_time >= output_interval:
                    print(f"\n[{time.strftime('%H:%M:%S', time.localtime(current_time))}] IMU Data:")
                    print(f"  Accelerometer (m/s²): X={result['Accelerometer_X']:8.4f}, Y={result['Accelerometer_Y']:8.4f}, Z={result['Accelerometer_Z']:8.4f}")
                    print(f"  Angular Velocity (rad/s): Roll={result['RollSpeed']:8.4f}, Pitch={result['PitchSpeed']:8.4f}, Heading={result['HeadingSpeed']:8.4f}")
                    print(f"  Orientation (rad): Roll={result['Roll']:8.4f}, Pitch={result['Pitch']:8.4f}, Heading={result['Heading']:8.4f}")
                    print(f"  Quaternion: w={result['qw']:8.4f}, x={result['qx']:8.4f}, y={result['qy']:8.4f}, z={result['qz']:8.4f}")
                    
                    last_output_time = current_time
                
                temp1 = False
                temp2 = False

    except KeyboardInterrupt:
        print("\n\nIMU data dump stopped by user.")
    except Exception as e:
        print(f"\nError during data reading: {e}")
    finally:
        if serial_.isOpen():
            serial_.close()
            print("Serial port closed.")


if __name__ == "__main__":
    args = parse_opt()
    dump_imu_data(port=args.port, baudrate=args.bps, timeout=args.timeout, output_rate=args.rate)