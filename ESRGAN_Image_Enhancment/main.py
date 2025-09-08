import sys
from Real_ESRGAN_master.inference_realesrgan import main

sys.argv = [
    'inference_realesrgan.py',
    '-i', r'../enhanced_images',
    '-o', r'D:/MTU_Satelite_Project/Yaman_Dataset/Adnan',
    '-n', 'RealESRGAN_x4plus',
    '--suffix', '',
    '--outscale', '1.6',
    '--ext', 'png',
]
main()