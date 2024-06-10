from picamera2 import Picamera2
from time import sleep
from lobe import ImageModel
import serial
import threading

# إعدادات المنفذ التسلسلي
serial_port = '/dev/serial0'
baud_rate = 115200

# محاولة الاتصال بالـ Arduino
try:
    ser = serial.Serial(serial_port, baud_rate)
    print("Connected to Arduino Uno on", serial_port)
except serial.SerialException as e:
    print("Error connecting:", e)
    exit()

camera = Picamera2()
camera.start()
sleep(2)

# Load Lobe TF model
# --> Change model file path as needed
model = ImageModel.load('TFModel')

def send_command(command):
    """Sends a single character command to the Arduino."""
    try:
        ser.write(command.encode())  # Encode the command character for transmission
        print("Sent command:", command)
    except serial.SerialException as e:
        print("Error sending command:", e)

def read_line():
    line = b''  # Initialize an empty byte array
    try:
        while True:
            char = ser.read(1)  # Read one byte at a time
            if char == b'\n':  # Check for newline character
                line = b''
                response = line.decode('utf-8').strip()
                print(f"Received response: {response}")
            line += char
    except UnicodeDecodeError:  # Handle potential decoding errors
        print("Error decoding received data. Check your Arduino's encoding.")
    except serial.SerialException as e:
        print(f"Serial error: {e}")
 
# Take Photo
def take_photo():
    camera.capture_file('Pictures/image.jpg')

def predict():
    while True:
        take_photo()
        # Run photo through Lobe TF model
        result = model.predict_from_file('Pictures/image.jpg')
        # --> Change image path
        print(result.prediction)
        if result.prediction == 'Non Recyclable':
            send_command('N')
        elif result.prediction == 'Recyclable':
            send_command('Y')


# Main Function
while True:
    t1 = threading.Thread(target=predict, args=())
    t2 = threading.Thread(target=read_line, args=())
    t1.start()
    t2.start()
   
    user_input = input("Enter a command (F: Forward, B: Backward, R: Right, L: Left, S: Stop): ")
    user_input = user_input.upper()  # Convert to uppercase for case-insensitive comparison
    if user_input == 'F':
        send_command('F')
    elif user_input == 'B':
        send_command('B')
    elif user_input == 'R':
        send_command('R')
    elif user_input == 'L':
        send_command('L')
    elif user_input == 'S':
        send_command('S')
    else:
        print("Invalid command. Please try again.")