# This needs to be the interface between the model and the machine code (.ino file)
# The layout or the values need to be changed according to the hardware interface being used and its setup.
class HardwareInterface:
    def __init__(self, hardware_params):
        # Initialize motor control pins (GPIO setup)
        self.motor_pin_in1 = hardware_params['motor_pins'][0]
        self.motor_pin_in2 = hardware_params['motor_pins'][1]

        # Initialize rotary encoder pins (GPIO setup)
        self.encoder_pin_clk = hardware_params['encoder_pins'][0]
        self.encoder_pin_dt = hardware_params['encoder_pins'][1]

        # Setup initial states and values for motor control and angle measurement
        self.angle = 0
        self.last_state_clk = self.read_encoder_pin(self.encoder_pin_clk)
        
        # Define motor control parameters (e.g., speed, direction)
        self.motor_speed = 0

    def read_encoder_pin(self, pin):
        # Implement GPIO read functionality
        return gpio_read(pin)

    def update_angle(self):
        current_state_clk = self.read_encoder_pin(self.encoder_pin_clk)

        if current_state_clk != self.last_state_clk:
            # Check rotation direction
            if self.read_encoder_pin(self.encoder_pin_dt) != current_state_clk:
                self.angle += 1  # Clockwise rotation
            else:
                self.angle -= 1  # Counter-clockwise rotation
            
            # Update last state
            self.last_state_clk = current_state_clk

    def control_motor(self, action):
        if action == 0:
            # Move motor to the left (e.g., set IN1 high, IN2 low)
            gpio_write(self.motor_pin_in1, HIGH)
            gpio_write(self.motor_pin_in2, LOW)
        elif action == 1:
            # Move motor to the right (e.g., set IN1 low, IN2 high)
            gpio_write(self.motor_pin_in1, LOW)
            gpio_write(self.motor_pin_in2, HIGH)

        # Optionally set the motor speed if using PWM
        # pwm_write(self.motor_pwm_pin, self.motor_speed)

    def reset(self):
        self.angle = 0
        return self.get_state()

    def get_state(self):
        # Calculate angular velocity if necessary
        angle_velocity = calculate_angle_velocity(self.angle)
        
        return (self.angle, angle_velocity)

    def step(self, action):
        self.control_motor(action)
        self.update_angle()
        next_state = self.get_state()

        # Define reward and done conditions based on the current state
        reward = calculate_reward(next_state)
        done = check_done(next_state)

        return next_state, reward, done

    def close(self):
        # Clean up GPIO pins
        gpio_cleanup()