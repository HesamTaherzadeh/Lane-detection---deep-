import time
import cv2
import psutil
import os

def gstreamer_pipeline(
        sensor_id=0,
        capture_width=1920,
        capture_height=1080,
        display_width=960,
        display_height=540,
        framerate=30,
        flip_method=0, 
        ):
            return (
                "nvarguscamerasrc sensor-id=%d !"
                "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
                "nvvidconv flip-method=%d ! "
                "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=(string)BGR ! appsink"
                % (
                    sensor_id,
                    capture_width,
                    capture_height,
                    framerate,
                    flip_method,
                    display_width,
                    display_height,
                )
            ) 


class Camera:
    def __init__(self, isvideo):
        self.isvideo = isvideo



    def show_camera(self, func, address=None, waitkeyno=1, *args):
            window_title = "CSI Camera"
            # Tqo flip the image, modify the flip_method parameter (0 and 2 are the most common)
            print(gstreamer_pipeline(flip_method=0))
            if self.isvideo != True:
                video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
            else:
                try :
                    video_capture = cv2.VideoCapture(address)
                except:
                    print("give the address to the video ")
            if video_capture.isOpened():
                try:
                    window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
                    while True:
                        ret_val, frame = video_capture.read()
                        prev_frame_time = 0
 
                        new_frame_time = 0
                        if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                            prev_frame_time = time.time()
                            frame = cv2.resize(frame, (960, 540))
                            frame = func(frame, *args)
                            cpu_usage = self.print_cpu_gpu_stats()
                            new_frame_time = time.time()
                            fps = "FPS : " + str(int(1/(new_frame_time-prev_frame_time)))
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            prev_frame_time = new_frame_time
                            cv2.putText(frame, fps, (7, 70), font, 3, (100, 0, 255), 3, cv2.LINE_AA)
                            cv2.putText(frame, cpu_usage, (7, 110), font, 1, (100, 0, 255), 1, cv2.LINE_AA)
                            cv2.imshow(window_title, frame)
                        else:
                            break 
                        keyCode = cv2.waitKey(waitkeyno) & 0xFF
                        # Stop the program on the ESC key or 'q'
                        if keyCode == 27 or keyCode == ord('q'):
                            break
                finally:
                    video_capture.release()
                    cv2.destroyAllWindows()
            else:
                print("Error: Unable to open camera")

    def print_cpu_gpu_stats(self):
        load1, load5, load15 = psutil.getloadavg()
        cpu_usage = (load15/os.cpu_count()) * 100
        # gpu_usage = self.get_gpu_memory()
        return "CPU usage : " + str(cpu_usage)

    # def get_gpu_memory(self):
    #     nvidia_smi.nvmlInit()
    #     deviceCount = nvidia_smi.nvmlDeviceGetCount()
    #     for i in range(deviceCount):
    #         handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
    #         info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    #     gpu_stats = "Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used)
    #     nvidia_smi.nvmlShutdown()
    #     return gpu_stats


    

