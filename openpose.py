import os

def openpose(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok = True)
    cmd = rf"openpose-master\artifacts\bin\OpenPoseDemo.exe --image_dir {input_folder} --hand --disable_blending --display 0 --write_json {output_folder}\openpose_json --write_images {output_folder}\openpose_img --model_folder openpose-master\artifacts\models"
   
    os.system(cmd)
    
openpose('qqq/image', 'www')



# import os

# import pathlib
# import shutil 

# path = pathlib.Path(__file__).parent.resolve()
# print(path)
# def openpose(input_folder, output_folder):
#     print(input_folder, output_folder)
#     os.makedirs(output_folder, exist_ok = True)
#     cmd = rf"openpose-master\artifacts\bin\OpenPoseDemo.exe --image_dir {input_folder} --hand --disable_blending --display 0 --write_json {output_folder}\openpose_json --write_images {output_folder}\openpose_img --model_folder openpose-master\artifacts\models"
   
#     os.system(cmd)
    
# openpose(os.path.join(path, 'data', 'image'), os.path.join(path, 'data'))
