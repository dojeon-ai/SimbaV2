import ffmpy

import os
root_dir = "/home/nas4_user/youngdolee/SimbaV2_dev/docs/dummy"

for root, dirs, files in sorted(list(os.walk(root_dir))):
    # only consider the directory structure that matches our pattern
    
    for filename in files:
        if filename.startswith("simbav2-") and filename.endswith(".mp4"):
            full_path = os.path.join(root, filename)
        
        ff = ffmpy.FFmpeg(
            inputs={full_path: None},     
            outputs={filename.replace('mp4', 'gif'): None}
        )
        ff.run()
            