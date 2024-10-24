import csv
import datetime
import time
import threading
from pynput import keyboard
from pyparrot.Bebop import Bebop

# Counters for each movement function
move_forward_count = 0
move_backwards_count = 0
move_right_count = 0
move_left_count = 0
move_up_count = 0
move_down_count = 0
takeoff_count = 0
land_count = 0

# Bebop object
bebop = Bebop(drone_type="Bebop2")
# Data gathering variables
record = False
flight_permission = False
verbose = False

def generate_file_name():
    return f"flight_data_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.csv"

def record_flight():
    global record
    global verbose

    while True:
        if record:
            file_name = generate_file_name()
            csv_file = open(file_name, "w", newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Timestamp", "Battery", "Flying State", "Sensors"])  # CSV header

            try:
                start_time = time.time()  # Start time
                count = 0  # Counter for number of lines written

                while record:
                    # Get the values
                    battery = bebop.sensors.battery  # Battery status
                    flying_state = bebop.sensors.flying_state  # Flying state
                    sensors_dict = bebop.sensors.sensors_dict  # All sensors
                    timestamp = time.time()  # Timestamp

                    # Write the values to the CSV file
                    csv_writer.writerow([timestamp, battery, flying_state, sensors_dict])

                    # Print verbose information
                    if verbose:
                        print(f"<INFO> Timestamp: {timestamp}\nBattery: {battery}\nFlying State: {flying_state}\nSensors: {sensors_dict}\n")

                    count += 1  # Increment the line counter

                    # Wait delay seconds before updating again
                    time.sleep(0.1)
            finally:
                csv_file.close()
                print(f"<INFO> Finished recording. Wrote {count} lines to {file_name}\n")
        else:
            # Se a gravação não estiver habilitada, aguarda 1 segundo antes de verificar novamente
            time.sleep(1)

def enable_commands():
    global flight_permission
    flight_permission = not flight_permission
    status = 'enabled' if flight_permission else 'disabled'
    print(f'Commands {status}\n')

def move_forward():
    if flight_permission:
        bebop.move_relative(0.5, 0, 0, 0)
        global move_forward_count
        move_forward_count += 1
    else:
        print('No permission\n')

def move_backwards():
    if flight_permission:
        bebop.move_relative(-0.5, 0, 0, 0)
        global move_backwards_count
        move_backwards_count += 1
    else:
        print('No permission\n')

def move_right():
    if flight_permission:
        bebop.move_relative(0, 0.5, 0, 0)
        global move_right_count
        move_right_count += 1
    else:
        print('No permission\n')

def move_left():
    if flight_permission:
        bebop.move_relative(0, -0.5, 0, 0)
        global move_left_count
        move_left_count += 1
    else:
        print('No permission\n')

def move_up():
    if flight_permission:
        bebop.move_relative(0, 0, -0.5, 0)
        global move_up_count
        move_up_count += 1
    else:
        print('No permission\n')

def move_down():
    if flight_permission:
        bebop.move_relative(0, 0, 0.5, 0)
        global move_down_count
        move_down_count += 1
    else:
        print('No permission\n')

def takeoff():
    if flight_permission:
        bebop.safe_takeoff(3)
        global takeoff_count
        takeoff_count += 1
    else:
        print('No permission\n')

def land():
    if flight_permission:
        bebop.safe_land(3)
        global land_count
        land_count += 1
    else:
        print('No permission\n')

def end_flight():
    if flight_permission:
        bebop.emergency_land()
        print('End flight\n')

def start_recording():
    global record
    if not record:
        record = True
        print('Starting to record data\n')

def stop_recording():
    global record
    if record:
        record = False
        print('Finishing recording\n')

def print_and_reset_counters():
    global move_forward_count, move_backwards_count, move_right_count, move_left_count, move_up_count, move_down_count, takeoff_count, land_count
    print(f"Move forward used: {move_forward_count} times")
    print(f"Move backwards used: {move_backwards_count} times")
    print(f"Move right used: {move_right_count} times")
    print(f"Move left used: {move_left_count} times")
    print(f"Move up used: {move_up_count} times")
    print(f"Move down used: {move_down_count} times")
    print(f"Takeoff used: {takeoff_count} times")
    print(f"Land used: {land_count} times")

    # Reset counters
    move_forward_count = 0
    move_backwards_count = 0
    move_right_count = 0
    move_left_count = 0
    move_up_count = 0
    move_down_count = 0
    takeoff_count = 0
    land_count = 0

def on_press(key):
    try:
        if key.char == 'w':
            move_forward()
        elif key.char == 's':
            move_backwards()
        elif key.char == 'd':
            move_right()
        elif key.char == 'a':
            move_left()
        elif key.char == 'e':
            move_up()
        elif key.char == 'q':
            move_down()
        elif key.char == 't':
            takeoff()
        elif key.char == 'l':
            land()
        elif key.char == 'u':
            end_flight()
        elif key.char == 'c':
            start_recording()
        elif key.char == 'v':
            stop_recording()
        elif key.char == 'b':
            enable_commands()
        elif key.char == 'p':
            print_and_reset_counters()
    except AttributeError:
        pass  # Handle special keys if needed

def keyboard_control():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def main():
    # Start the recording thread
    thread_data = threading.Thread(target=record_flight)
    thread_data.daemon = True
    thread_data.start()

    # Start the keyboard control thread
    thread_control = threading.Thread(target=keyboard_control)
    thread_control.start()
    thread_control.join()

# Main execution
try:
    success = bebop.connect(5)
    if success:
        print('Connected to Drone\n')
        main()
    else:
        print('\n<INFO> Error connecting to Drone\n')
finally:
    bebop.disconnect()
    print('Disconnected from Drone\n')