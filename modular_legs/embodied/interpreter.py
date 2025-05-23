

def interpret_motor_mode(mode):
        if mode == 0:
            return "Reset"
        elif mode == 1:
            return "Cali"
        elif mode == 2:
            return "Motor"
        else:
            return "?"

def interpret_motor_error(error):
    error_flag = ""
    if error & (0x01 << 0):
        error_flag = error_flag + "UNDERVOLTAGE  "
    if error & (0x01 << 1):
        error_flag = error_flag + "OVER CURRENT"
    if error & (0x01 << 2):
        error_flag = error_flag + "OVER TEMPERATURE"
    if error & (0x01 << 3):
        error_flag = error_flag + "MAGNETIC ENCODER ERROR  "
    if error & (0x01 << 4):
        error_flag = error_flag + "HALL ENCODER ERROR  "
    if error & (0x01 << 5):
        error_flag = error_flag + "UNCALIBRATED  "
    return error_flag

def interpret_reset_reason(reason):
    if reason == 1:
        return "POWERON_RESET: Vbat power on reset"         # /**<1,  Vbat power on reset*/
    elif reason == 3:
        return "SW_RESET: Software reset digital core"              # /**<3,  Software reset digital core*/
    elif reason == 4:
        return "OWDT_RESET: Legacy watch dog reset digital core"            # /**<4,  Legacy watch dog reset digital core*/
    elif reason == 5:
        return "DEEPSLEEP_RESET: Deep Sleep reset digital core"       # /**<5,  Deep Sleep reset digital core*/
    elif reason == 6:
        return "SDIO_RESET: Reset by SLC module, reset digital core"            # /**<6,  Reset by SLC module, reset digital core*/
    elif reason == 7:
        return "TG0WDT_SYS_RESET: Timer Group0 Watch dog reset digital core"      # /**<7,  Timer Group0 Watch dog reset digital core*/
    elif reason == 8:
        return "TG1WDT_SYS_RESET: Timer Group1 Watch dog reset digital core"      # /**<8,  Timer Group1 Watch dog reset digital core*/
    elif reason == 9:
        return "RTCWDT_SYS_RESET: RTC Watch dog Reset digital core"      # /**<9,  RTC Watch dog Reset digital core*/
    elif reason == 10:
        return "INTRUSION_RESET: Instrusion tested to reset CPU"       # /**<10, Instrusion tested to reset CPU*/
    elif reason == 11:
        return "TGWDT_CPU_RESET: Time Group reset CPU"       # /**<11, Time Group reset CPU*/
    elif reason == 12:
        return "SW_CPU_RESET: Software reset CPU"          # /**<12, Software reset CPU*/
    elif reason == 13:
        return "RTCWDT_CPU_RESET: RTC Watch dog Reset CPU"      # /**<13, RTC Watch dog Reset CPU*/
    elif reason == 14:
        return "EXT_CPU_RESET: for APP CPU, reseted by PRO CPU"         # /**<14, for APP CPU, reseted by PRO CPU*/
    elif reason == 15:
        return "RTCWDT_BROWN_OUT_RESET: Reset when the vdd voltage is not stable"# /**<15, Reset when the vdd voltage is not stable*/
    elif reason == 16:
        return "RTCWDT_RTC_RESET: RTC Watch dog reset digital core and rtc module"      # /**<16, RTC Watch dog reset digital core and rtc module*/
    else:
        return "NO_MEAN"
    


def interpret_motor_msg(msg):

    motor_info_dict = {
        0: "",
        100: "[WARN][Wifi] Wifi disconnected! Try to reconnect!",
        200: "[ERROR] No BNO055 detected.",
        201: "[IMU] BNO055 inited.",
        300: "[ERROR] [Motor] The motor is not calibrated.",
        301: "[Motor] Motor enabled.",
        302: "[Motor] Motor disabled.",
        303: "[WARN] [Motor] The remote swich is on. Please turn it off!!!",
        304: "[Motor] Please turn on the remote switch for motor calibration!",
        305: "[WARN] [Motor] Unsafe situations found!",
        306: "[Motor] Set to middle position",
        307: "[Motor] motor set to middle position.",
        308: "[Motor] Start auto calibration!",
        309: "[Motor] Zero position set.",
        310: "[Motor] Start manual calibration!",
        311: "[Motor] Zero position set.",
        312: "[Motor] Detect calibration command.",
        313: "[Motor] Motor initialized.",
        314: "[Motor] Swith-off request sent!",
        315: "[Motor] Restart!"
    }

    return motor_info_dict[msg]