from utils import *
from bodymarker import barreleye_bodymarker_visbus
import pydicom
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
import cv2
import os




########################################################################################
######################### 폴더 전체에서 읽어오는 거(sample 전체 테스트)##########################
'''
folder_path = './Sample_Files'
image_files = [f for f in os.listdir(folder_path) if f.endswith('.dcm')]
image_files = sorted(image_files, key=lambda x: int(x.split('Sample')[1].split('.dcm')[0]))

for file_name in image_files:
   print(file_name)

sample_num = 1
for file_name in image_files:
    file_path = os.path.join(folder_path, file_name)
    dcm_sample = pydicom.dcmread(file_path).pixel_array
    print(file_name)
    
    #dcm 원본 이미지 출력
    plt.imshow(dcm_sample)
    plt.title(f'Original Sample{sample_num} image')
    plt.show()

    BIRADs_lesionA = barreleye_bodymarker_visbus()
    
    #bodymarker extraction
    bodymarker_img, RorL = BIRADs_lesionA.extract_bodymarker(dcm_sample)

    if RorL == 'No image': #extraction 실패
      print(f'Sample {sample_num}: {RorL}')
    else: #extraction 성공
      #extraction 결과 출력
      plt.imshow(bodymarker_img)
      plt.title(f'Sample {sample_num} extraction Result, RorL: {RorL}')
      plt.show()
      #extraction 성공하면 classifying
      BIRADs_lesionA.classify_bodymarker(bodymarker_img)
    
    sample_num += 1
'''
########################################################################################




########################################################################################
############################## sample 하나만 테스트하는 거 ##################################
#이거 번호만 바꾸면 됨
sample_num = 13

dcm_sample = pydicom.dcmread(f'Sample_Files/Sample{sample_num}.dcm').pixel_array # Dicom 파일 읽어옴
plt.imshow(dcm_sample)
plt.title(f'Original Sample{sample_num} image')
plt.show()

BIRADs_lesionA = barreleye_bodymarker_visbus()  # BIRADs 특성 Initialize

bodymarker_img, RorL = BIRADs_lesionA.extract_bodymarker(dcm_sample)
if RorL == 'No image':
  print(f'Sample {sample_num}: {RorL}')
else:
  plt.imshow(bodymarker_img)
  plt.title(f'Sample {sample_num} extraction Result, RorL: {RorL}')
  plt.show()
  BIRADs_lesionA.classify_bodymarker(bodymarker_img)

########################################################################################
