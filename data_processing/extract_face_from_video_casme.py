#coding:utf-8
import numpy as np
import cv2
import face_landmark_detection
import os
import demo
import glob
import concurrent.futures

# preprocessing method: including affine and frontalization


def deal_with_filename(subdir):
    targetsub = '/media/ustb/Dataset2/biovid/PartA/reprocess_slstm/B_data_argu_front'

    # start_frame = os.listdir(subdir)
    # for each_frame in start_frame:
    #     classes = os.listdir(os.path.join(subdir,each_frame))
    #     for each_class in classes:
    #         samples = os.listdir(os.path.join(subdir,each_frame,each_class))
    samples = os.listdir(subdir)
    for each_sample in samples:
        imgs = os.listdir(os.path.join(subdir,each_sample))
        for filename1 in imgs:
            print(os.path.join(subdir,each_sample))
            S = os.path.join(subdir,each_sample).split('/')
            the_class = S[9]
            sample_name = S[10]
            frame = cv2.imread(os.path.join(subdir,each_sample,filename1))
            target_filename = os.path.join(targetsub,the_class,sample_name,filename1)
            print(target_filename)
            base_path = '/media/ustb/Personalfiles/Wandameng/1.jpg'
            count = 1
            # face alignment
            im1, im2, M, landmark1, landmark2 = face_landmark_detection.face_align(base_path, frame, 0)
            warped_im2 = im2
            if M == [1, 1]:
                warped_img2 = im2
                print('error')
                #### dont save the pictures which were failed to do the face alignment
                continue
            else:
                landmark2 = np.array(landmark2)
                landmark1 = np.array(landmark1)
                b = np.array([[landmark2[0], landmark2[1], landmark2[2], landmark2[3], landmark2[4], landmark2[5],
                               landmark2[6], landmark2[7], landmark2[8], landmark2[9], landmark2[10], landmark2[11],
                               landmark2[12], landmark2[13], landmark2[14], landmark2[15], landmark2[16],
                               landmark2[26],
                               landmark2[25], landmark2[24], landmark2[19], landmark2[18], landmark2[17]]],
                             dtype=np.int32)
                im = np.zeros(im2.shape[:2], dtype="uint8")
                cv2.polylines(im, b, 1, 255)
                cv2.fillPoly(im, b, 255)
                #face frontalization and crop
                mask = im
                masked = cv2.bitwise_and(im2, im2, mask=mask)

                warped_im2 = face_landmark_detection.warp_im(masked, M, im1.shape)


            front_img = demo.demo(warped_im2)
            #cv2.imshow('alignment img', front_i  mg)
            #cv2.waitKey(4)
            cv2.imwrite(target_filename,front_img)
            print(count)
            count+=1


with concurrent.futures.ProcessPoolExecutor() as executor:
    subfiles = '/media/ustb/Personalfiles/Wandameng/xuhairui/CASEME II'
    for each_start_frame in os.listdir(subfiles):
        frame_sub = os.path.join(subfiles, each_start_frame)
        for each_class in os.listdir(frame_sub):
            # class_sub = os.path.join(frame_sub, each_class)
            # for each_file in os.listdir(class_sub):
            subfiles = glob.glob(os.path.join(frame_sub, each_class))
            for image_file in zip(subfiles, executor.map(deal_with_filename, subfiles)):
                print(subfiles)