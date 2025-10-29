from asyncua.sync import Client, ThreadLoop
from asyncua import ua
import time
import logging

# ────────────────────────────────────────────────────────────────
# 🧠 Base class for all OPC UA devices
# ────────────────────────────────────────────────────────────────

class OPCUADevice:
    def __init__(self, url, auto_start=True):
        self.url = url
        self.client = None
        self.tloop = None
        if auto_start:
            self.start_communication()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_communication()
        return False

    def start_communication(self):
        """Initialize ThreadLoop and connect to OPC UA server."""
        self.tloop = ThreadLoop()
        self.tloop.daemon = True
        self.client = Client(self.url, tloop=self.tloop)
        self.tloop.start()
        self.client.connect()
        print(f"✅ Connected to OPC UA server: {self.url}")

    def get_node(self, nodeid: str):
        """Shortcut for get_node."""
        return self.client.get_node(nodeid)

    def stop_communication(self):
        """Gracefully disconnect and stop loop."""
        if self.client:
            self.client.disconnect()
        if self.tloop:
            self.tloop.stop()
        print(f"🔌 Disconnected from {self.url}")

    def __del__(self):
        self.stop_communication()


# ────────────────────────────────────────────────────────────────
# ⚙️ PLC Client (inherits base)
# ────────────────────────────────────────────────────────────────

class PLCClient(OPCUADevice):
    def __init__(self, url, auto_start=True):
        super().__init__(url, auto_start)
        if auto_start:
            self.init_nodes()

    def init_nodes(self):
        """Initialize PLC-specific nodes."""
        self.node_x = self.get_node('ns=4;i=2')
        self.node_y = self.get_node('ns=4;i=3')
        self.node_z = self.get_node('ns=4;i=4')
        self.node_trigger = self.get_node('ns=4;i=5')
        # self.node_ack = self.get_node("ns=3;s='VisionData'.'Ack'")

    def send_coordinates(self, x, y, z):
        for node, val in zip((self.node_x, self.node_y, self.node_z), (x, y, z)):
            node.set_value(ua.Variant(val, ua.VariantType.Float))
        print(f"📤 Sent coordinates to PLC: ({x:.3f}, {y:.3f}, {z:.3f})")

    def set_trigger(self, value: bool):
        self.node_trigger.set_value(ua.Variant(value, ua.VariantType.Boolean))
        print("SENT")

    # def get_ack(self) -> bool:
    #     return self.node_ack.get_value()


# ────────────────────────────────────────────────────────────────
# 🤖 Yaskawa Robot Client (inherits base)
# ────────────────────────────────────────────────────────────────

class Yaskawa_YRC1000(OPCUADevice):
    def __init__(self, url, auto_start=True):
        super().__init__(url, auto_start)
        if auto_start:
            self.init_nodes()

    def init_nodes(self):
        """Initialize Yaskawa-specific nodes."""
        self.running_var = self.get_node(
            "ns=5;s=MotionDeviceSystem.Controllers.Controller_1.ParameterSet.IsRunning")
        self.controller_obj = self.client.nodes.root.get_child([
            "0:Objects",
            "2:DeviceSet",
            "4:MotionDeviceSystem",
            "4:Controllers",
            "4:Controller_1",
            "5:Methods"
        ])
        print("✅ Robot nodes initialized")

    def get_available_jobs(self):
        return self.controller_obj.call_method("5:GetAvailableJobs")

    def set_servo(self, enable: bool):
        return self.controller_obj.call_method("5:SetServo", enable)

    def start_job(self, job_name, block=True):
        print(f"▶️ Starting job: {job_name}")
        self.controller_obj.call_method("5:StartJob", job_name)
        time.sleep(0.1)
        if block:
            running = self.running_var.get_value()
            print("running job: ",job_name)
            while running == True:
                running = self.running_var.get_value()
            print("finished job: ", job_name)


def get_vision_coordinates():
    # Replace with actual camera or model output
    u, v, depth = 320, 240, 0.42
    fx, fy, cx, cy = 607.7, 607.8, 320.1, 258.0
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    return X, Y, Z


# ────────────────────────────────────────────────────────────────
# 🚀 Main program
# ────────────────────────────────────────────────────────────────

def main():
    robot_url = "opc.tcp://192.168.0.20:16448"
    plc_url = "opc.tcp://192.168.0.1:4840"

    #robot = Yaskawa_YRC1000(robot_url)
    plc = PLCClient(plc_url)

    try:
        print("System initialized. Waiting for PLC Ack...")
        # while not plc.get_ack():
        #     time.sleep(0.1)
        print("✅ PLC ready.")

        #coords = get_vision_coordinates()
        #plc.send_coordinates(*coords)

        plc.set_trigger(True)
        time.sleep(0.2)
        plc.set_trigger(False)

        # robot.set_servo(True)
        # robot.start_job('TICTACTOE_X0_HOME_PLAY', block=True)
        # robot.set_servo(False)

        # Notify PLC robot done
        # plc.set_trigger(True)
        # time.sleep(0.2)
        # plc.set_trigger(False)

    finally:
        plc.stop_communication()
        #robot.stop_communication()
        print("🔚 Program ended.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()