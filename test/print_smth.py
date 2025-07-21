import time

def print_num():
    start_time = time.perf_counter()
    for i in range(10):
        print(f"Number: {i}")
        time.sleep(0.2)

    end_time = time.perf_counter()

    print(f"Process finished in {round(end_time-start_time, 2)} second(s)")