from asyncua.sync import Client, ThreadLoop
import time

class Yaskawa_YRC1000:
    def __init__(self, url, auto_start = True):
        self.url = url

        if auto_start:
            self.start_communication()

    def start_communication(self):
        self.tloop = ThreadLoop()
        self.client = Client(self.url, tloop=self.tloop)
        self.tloop.start()
        self.client.connect()
        self.running_var = self.client.get_node(
            "ns=5;s=MotionDeviceSystem.Controllers.Controller_1.ParameterSet.IsRunning")
        print("running_var", self.running_var)
        print("Children of root are: %r", self.client.nodes.root.get_children())
        self.controller_obj = self.client.nodes.root.get_child(
            ["0:Objects", "2:DeviceSet", "4:MotionDeviceSystem", "4:Controllers", "4:Controller_1", "5:Methods"])

    def stop_communication(self):
        self.client.disconnect()
        self.tloop.stop()

    def get_available_jobs(self):
        res = self.controller_obj.call_method("5:GetAvailableJobs")
        return res

    def set_servo(self,set_bool):
        res = self.controller_obj.call_method("5:SetServo", set_bool)
        return res

    def start_job(self,job_name, block=True):
        print("call method", job_name)
        self.controller_obj.call_method("5:StartJob", job_name)
        time.sleep(0.1) # wait for job to start such that blocking works out, otherwise robot_running_var is not updated fast enough and the loop will not start
        if block:
            running = self.running_var.get_value()
            print("running job: ",job_name)
            while running == True:
                running = self.running_var.get_value()
            print("finished job: ", job_name)
        #return (running)

    def __del__(self):
        self.stop_communication()
        print("disconnected")



def main():
    robot_url = "opc.tcp://192.168.1.20:16448"

    robot = Yaskawa_YRC1000(robot_url)
    print("robot initialized")
    print(robot.get_available_jobs())
    print(robot.set_servo(True))
    print(robot.start_job('TICTACTOE_X0_HOME_PLAY', block=True))
 #   print(robot.start_job("BASE2TOBASE1", block=True))
    print(robot.set_servo(False))
    robot.stop_communication()
    print('Program ended.')
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main()

