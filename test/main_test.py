from multiprocessing import Process
from print_smth import print_num
import time


def main():
    # Do something important
    print("--- Main task ---")
    start_time = time.perf_counter()
    for i in range(10):
        print("Main number:", i)
        time.sleep(0.1)
    end_time = time.perf_counter()
    print(f"Process finished in {round(end_time-start_time, 2)} second(s)")

if __name__ == "__main__":
    try:
        start_time = time.perf_counter()
        main()
        

        p1 = Process(target=print_num)
        #p2 = Process(target=main)
        print("P1 starts...")
        p1.start()
        #p2.start()

        print("Simulating main loop doing something else...")

        p1.join()

        print("P1 end...")
        #p2.join()
        
        end_time = time.perf_counter()
        print(f"Multiprocess finished in {round(end_time-start_time, 2)} second(s)")
    except KeyboardInterrupt:
        print("Ending processes...")

        p1.terminate()
        p1.join()