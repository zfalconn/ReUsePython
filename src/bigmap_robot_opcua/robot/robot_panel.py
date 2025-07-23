import panel as pn
from .Yaskawa_YRC1000_OPCUA_client import Yaskawa_YRC1000

class RobotPanel:
    def __init__(self,robot_object:Yaskawa_YRC1000):
        if robot_object:
            self.robot = robot_object
        else:
            self.robot = Yaskawa_YRC1000(robot_url)
        self.build_panel()

    def get_panel(self):
        return self._panel

    def start_servo_callback(self,event):
        self.robot.set_servo(True)

    def stop_servo_callback(self,event):
        self.robot.set_servo(False)

    def connect_robot_callback(self,event):
        self.robot.start_communication()
        print("robot connected")
    def disconnect_robot_callback(self,event):
        self.robot.stop_communication()
        print("robot disconnected")

    def build_panel(self):
        ### function column: a button for every job
        job_list = (job for job in self.robot.get_available_jobs() if job.startswith("!"))

        self.job_column = pn.Column(height=400)
        self.job_button_dict = {}
        for job in job_list:
            button = pn.widgets.Button(name=job, button_type="primary")
            self.job_button_dict["job_button_dict"] = button
            button.on_click(lambda event: self.robot.start_job(event.obj.name, block=False))  ## starts a robot job with exactly the name of the button
            self.job_column.append(button)

        ### basic controls: servo, disconnect
        self.start_servo_button = pn.widgets.Button(name="start servo", button_type="primary")
        self.start_servo_button.on_click(self.start_servo_callback)
        self.stop_servo_button = pn.widgets.Button(name="stop servo", button_type="primary")
        self.stop_servo_button.on_click(self.stop_servo_callback)

        self.disconnect_robot_button = pn.widgets.Button(name="disconnect", button_type="primary")
        self.disconnect_robot_button.on_click(self.disconnect_robot_callback)

        self.reconnect_robot_button = pn.widgets.Button(name="re-connect", button_type="primary")
        self.reconnect_robot_button.on_click(self.connect_robot_callback)

        self.on_off_col = pn.Column(self.start_servo_button,
                              self.stop_servo_button,
                              self.disconnect_robot_button,
                              self.reconnect_robot_button)

        self.row = pn.Row(self.on_off_col,self.job_column)

        self.header = pn.pane.HTML("""<h1>Robot</h1>""")

        self._panel = pn.Column(self.header,self.row)

if __name__=="__main__":

    robot_url = "opc.tcp://192.168.1.20:16448"
    robot = Yaskawa_YRC1000(robot_url)
    robotPanel = RobotPanel(robot)

    pn.serve(robotPanel.get_panel,port=22223,allowed_websocket_origins=["*"],threaded=True)



