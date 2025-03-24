import ffmpy

import os
root_dir = "wandb"

seed = ["0", "1", "2", "3", "4"]
env = ["cheetah-v4", "hopper-v4", "walker-v4", "ant-v4", "humanoid-v4"]

i = 0
j = 0
for root, dirs, files in sorted(list(os.walk(root_dir))):
    # only consider the directory structure that matches our pattern
    if ("files" in root.split(os.sep) and 
        "media" in root.split(os.sep) and
        "videos" in root.split(os.sep)):
        for filename in files:
            if filename.startswith("video_") and filename.endswith(".gif"):
                full_path = os.path.join(root, filename)
                print("Found gif:", full_path)

                if j == 5:
                    i += 1
                    j = 0
                
                ff = ffmpy.FFmpeg(
                    inputs={'/home/nas4_user/youngdolee/SimbaV2/' + full_path: None},     
                    outputs={f'simbav2-{env[j]}-{seed[i]}.mp4': '-vcodec libx264 -pix_fmt yuv420p'}
                )
                ff.run()
                j += 1
                