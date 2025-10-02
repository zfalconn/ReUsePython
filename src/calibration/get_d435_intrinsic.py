import pyrealsense2 as rs
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # request same resolution
profile = pipe.start(cfg)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
# intr.fx, intr.fy, intr.ppx, intr.ppy are the values you want
print(intr.fx, intr.fy, intr.ppx, intr.ppy)
pipe.stop()
