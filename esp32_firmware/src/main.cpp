// ═══════════════════════════════════════════════════════════
//   SMART CAR: SWERVE DRIVE "SPINAL CORD"
//   Architecture: ESP32
// ═══════════════════════════════════════════════════════════
#include <Arduino.h>
#include "FastAccelStepper.h"
#include <ESP32Encoder.h>

// =========================================================================
//   PIN DEFINITIONS (ESP32 DevKit V1)
// =========================================================================

// 1. STEPPER MOTORS (Steering) - DRV8825
#define FL_STEP  26
#define FL_DIR   27
#define FR_STEP  14
#define FR_DIR   12
#define BL_STEP  32
#define BL_DIR   33
#define BR_STEP  25
#define BR_DIR   13

// 2. DC MOTORS (Drive) - TB6612FNG or L298N
// NOTE: PWM/EN pins on the driver must be tied to 3.3V/5V (Always ON)! 
// We are sending PWM directly to IN1/IN2 to save pins.
#define FL_IN1   15
#define FL_IN2   2
#define FR_IN1   0
#define FR_IN2   4
#define BL_IN1   16
#define BL_IN2   17
#define BR_IN1   5
#define BR_IN2   18

// 3. ENCODERS (Feedback)
#define FL_ENC_A 36 // Input Only (VP)
#define FL_ENC_B 39 // Input Only (VN)
#define FR_ENC_A 34 // Input Only
#define FR_ENC_B 35 // Input Only
#define BL_ENC_A 19
#define BL_ENC_B 21
#define BR_ENC_A 22
#define BR_ENC_B 23

// =========================================================================
//   GLOBAL OBJECTS
// =========================================================================
FastAccelStepperEngine engine = FastAccelStepperEngine();
FastAccelStepper *steerFL = NULL;
FastAccelStepper *steerFR = NULL;
FastAccelStepper *steerBL = NULL;
FastAccelStepper *steerBR = NULL;

ESP32Encoder encFL;
ESP32Encoder encFR;
ESP32Encoder encBL;
ESP32Encoder encBR;

// =========================================================================
//   PWM CHANNELS (ESP32 has 16 hardware PWM channels)
// =========================================================================
const int pwmFreq = 5000;
const int pwmRes = 8; // 8-bit resolution (0-255)

const int CHAN_FL_IN1 = 0;
const int CHAN_FL_IN2 = 1;
const int CHAN_FR_IN1 = 2;
const int CHAN_FR_IN2 = 3;
const int CHAN_BL_IN1 = 4;
const int CHAN_BL_IN2 = 5;
const int CHAN_BR_IN1 = 6;
const int CHAN_BR_IN2 = 7;

// =========================================================================
//   SETUP
// =========================================================================
void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("ESP32 Swerve Drive Firmware Initializing...");

    // --- STEPPER INIT ---
    engine.init();
    steerFL = engine.stepperConnectToPin(FL_STEP);
    steerFR = engine.stepperConnectToPin(FR_STEP);
    steerBL = engine.stepperConnectToPin(BL_STEP);
    steerBR = engine.stepperConnectToPin(BR_STEP);

    if (steerFL && steerFR && steerBL && steerBR) {
        steerFL->setDirectionPin(FL_DIR);
        steerFR->setDirectionPin(FR_DIR);
        steerBL->setDirectionPin(BL_DIR);
        steerBR->setDirectionPin(BR_DIR);

        steerFL->setSpeedInHz(800); steerFL->setAcceleration(2000);
        steerFR->setSpeedInHz(800); steerFR->setAcceleration(2000);
        steerBL->setSpeedInHz(800); steerBL->setAcceleration(2000);
        steerBR->setSpeedInHz(800); steerBR->setAcceleration(2000);
    } else {
        Serial.println("ERROR: Failed to init steppers!");
    }

    // --- DC MOTOR INIT (PWM) ---
    ledcSetup(CHAN_FL_IN1, pwmFreq, pwmRes);
    ledcSetup(CHAN_FL_IN2, pwmFreq, pwmRes);
    ledcSetup(CHAN_FR_IN1, pwmFreq, pwmRes);
    ledcSetup(CHAN_FR_IN2, pwmFreq, pwmRes);
    ledcSetup(CHAN_BL_IN1, pwmFreq, pwmRes);
    ledcSetup(CHAN_BL_IN2, pwmFreq, pwmRes);
    ledcSetup(CHAN_BR_IN1, pwmFreq, pwmRes);
    ledcSetup(CHAN_BR_IN2, pwmFreq, pwmRes);

    ledcAttachPin(FL_IN1, CHAN_FL_IN1);
    ledcAttachPin(FL_IN2, CHAN_FL_IN2);
    ledcAttachPin(FR_IN1, CHAN_FR_IN1);
    ledcAttachPin(FR_IN2, CHAN_FR_IN2);
    ledcAttachPin(BL_IN1, CHAN_BL_IN1);
    ledcAttachPin(BL_IN2, CHAN_BL_IN2);
    ledcAttachPin(BR_IN1, CHAN_BR_IN1);
    ledcAttachPin(BR_IN2, CHAN_BR_IN2);

    // --- ENCODER INIT ---
    ESP32Encoder::useInternalWeakPullResistors = puType::up;
    encFL.attachHalfQuad(FL_ENC_A, FL_ENC_B);
    encFR.attachHalfQuad(FR_ENC_A, FR_ENC_B);
    encBL.attachHalfQuad(BL_ENC_A, BL_ENC_B);
    encBR.attachHalfQuad(BR_ENC_A, BR_ENC_B);

    Serial.println("Ready to receive SWERVE: commands!");
}

// =========================================================================
//   MOTOR CONTROL HELPER
// =========================================================================
void setDCMotorSpeed(int ch_in1, int ch_in2, int speed) {
    if (speed > 255) speed = 255;
    if (speed < -255) speed = -255;

    if (speed > 0) {
        ledcWrite(ch_in1, speed);
        ledcWrite(ch_in2, 0);
    } else if (speed < 0) {
        ledcWrite(ch_in1, 0);
        ledcWrite(ch_in2, -speed);
    } else {
        ledcWrite(ch_in1, 0);
        ledcWrite(ch_in2, 0);
    }
}

// =========================================================================
//   MAIN LOOP (Serial Parsing)
// =========================================================================
void loop() {
    if (Serial.available() > 0) {
        String data = Serial.readStringUntil('\n');
        data.trim();

        // Expecting format: "SWERVE:angleFL,speedFL,angleFR,speedFR,angleBL,speedBL,angleBR,speedBR"
        if (data.startsWith("SWERVE:")) {
            data = data.substring(7);

            float values[8];
            int idx = 0;
            int start = 0;
            int commaIndex = data.indexOf(',');

            while (commaIndex != -1 && idx < 8) {
                values[idx++] = data.substring(start, commaIndex).toFloat();
                start = commaIndex + 1;
                commaIndex = data.indexOf(',', start);
            }
            if (idx < 8) {
                values[idx] = data.substring(start).toFloat();
            }

            // Convert degrees to steps (200 steps per 360 deg = 0.555 steps/deg for Full Step)
            // Note: If using 1/16 microstepping, multiply this by 16.
            float deg_to_steps = 200.0 / 360.0;

            steerFL->moveTo(values[0] * deg_to_steps);
            setDCMotorSpeed(CHAN_FL_IN1, CHAN_FL_IN2, (int)values[1]);

            steerFR->moveTo(values[2] * deg_to_steps);
            setDCMotorSpeed(CHAN_FR_IN1, CHAN_FR_IN2, (int)values[3]);

            steerBL->moveTo(values[4] * deg_to_steps);
            setDCMotorSpeed(CHAN_BL_IN1, CHAN_BL_IN2, (int)values[5]);

            steerBR->moveTo(values[6] * deg_to_steps);
            setDCMotorSpeed(CHAN_BR_IN1, CHAN_BR_IN2, (int)values[7]);
            
            Serial.println("OK");
        }
    }
}
