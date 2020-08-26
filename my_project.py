import os, sys
import dlib
import glob
import subprocess
import numpy as np
import scipy.io as sio
from skimage import io
from time import time
import matplotlib.pyplot as plt

sys.path.append('..')
import face3d
from face3d import mesh
from face3d.morphable_model import MorphabelModel


# load BFM model
bfm = MorphabelModel('examples/Data/BFM/Out/BFM.mat')
print('init bfm model success')

uv_coords = face3d.morphable_model.load.load_uv_coords('examples/Data/BFM/Out/BFM_UV.mat') 
t = [0, 0, 0]
s = 8e-03
c = 3

#random texture and colors
tp = bfm.get_tex_para('random')



predictor_path = "shape_predictor_68_face_landmarks.dat"
faces_folder_path = sys.argv[1]
result_folder_path = sys.argv[2]
objFilePath = sys.argv[3]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()

ii = 0
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    ii += 1
    iii = 0

    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)

    win.clear_overlay()
    #win.set_image(img)

    # find the bounding boxes of each face. Upsample the image 1 time.
    dets = detector(img, 1)

    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        iii += 1
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)

        centroid_x = (d.left() + d.right())/2
        centroid_y = (d.top() + d.bottom())/2
        h = int(0-(d.top() - d.bottom())*1.6)
        w = int(0-(d.left() - d.right())*1.6)

        x = []
        for pt in shape.parts():
            a = float(pt.x) - centroid_x
            b = float(pt.y) - centroid_y
            tmp = np.array([a, b])
            x.append(tmp)
        x = np.array(x)

        #print(h,w)
        #print(img.shape)
        X_ind = bfm.kpt_ind # index of keypoints in 3DMM. fixed.

        # fit
        fitted_sp, fitted_ep, fitted_s, fitted_angles, fitted_t = bfm.fit(x, X_ind, max_iter = 3)

        tp = bfm.get_tex_para('random')
        colors = bfm.generate_colors(tp)
        colors = np.minimum(np.maximum(colors, 0), 1)

        fitted_vertices = bfm.generate_vertices(fitted_sp, fitted_ep)
        transformed_vertices = bfm.transform(fitted_vertices, fitted_s, fitted_angles, fitted_t)
        image_vertices = mesh.transform.to_image(transformed_vertices, h, w)

        #Invert y and z axis to make rendering image normal
        z = image_vertices[:,2:]
        z = 0 - z
        image_vertices[:,2:] = z
        y = image_vertices[:,1:2]
        y = w - y
        image_vertices[:,1:2] = y
        fitted_image = mesh.render.render_colors(image_vertices, bfm.triangles, colors, h, w)
        color = fitted_image
        #print(fitted_image[0,0])
          
        ## ----get color from original image------------------------------------------------------------
        ccc = 0
        for xx in range(int(centroid_y)-int(h/2),int(centroid_y)+int(h/2)):
            if xx < 0 or xx >= img.shape[0]:
                continue
            xxx = int(xx)+int(h/2)-int(centroid_y)
            for yy in range(int(centroid_x)-int(w/2),int(centroid_x)+int(w/2)):
                if yy < 0 or yy >=img.shape[1]:
                    continue
                yyy = int(yy)+int(w/2)-int(centroid_x)
                if fitted_image[xxx, yyy][0] != 0 and fitted_image[xxx, yyy][1] != 0 and fitted_image[xxx, yyy][2] != 0:
                    color[xxx,yyy] = img[xx,yy]
                    #img[xx,yy] = (255*fitted_image[xxx, yyy]).astype(np.uint8)
        
        ## ----add colors to vertices----------------------------------------------------------------
        vvv=0
        for ver in image_vertices:
            if int(ver[1])<fitted_image.shape[0] and int(ver[0])<fitted_image.shape[1]:
                colors[vvv] = color[int(ver[1]),int(ver[0])]
            vvv+=1  
        
        fitted_image = mesh.render.render_colors(image_vertices, bfm.triangles, colors, h, w)

        # for xx in range(int(centroid_y)-int(h/2),int(centroid_y)+int(h/2)):
        #     if xx < 0 or xx >= img.shape[0]:
        #         continue
        #     xxx = int(xx)+int(h/2)-int(centroid_y)
        #     for yy in range(int(centroid_x)-int(w/2),int(centroid_x)+int(w/2)):
        #         if yy < 0 or yy >=img.shape[1]:
        #             continue
        #         yyy = int(yy)+int(w/2)-int(centroid_x)
        #         #yyy = 100
        #         if fitted_image[xxx, yyy][0] != 0 and fitted_image[xxx, yyy][1] != 0 and fitted_image[xxx, yyy][2] != 0:
        #             #color[xxx,yyy] = img[xx,yy]
        #             img[xx,yy] = (255*fitted_image[xxx, yyy]).astype(np.uint8)

        #win.set_image(img)

        print('pose, fitted: \n', fitted_s, fitted_angles[0], fitted_angles[1], fitted_angles[2], fitted_t[0], fitted_t[1])

        save_folder = result_folder_path
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        name = "{}_".format(ii) + "{}".format(iii)
        io.imsave(save_folder + '/fitted_{}.jpg'.format(name), fitted_image)

        ## ---- save colors -------------------------------------------------------------------------
        np.savetxt("colors_{}.txt".format(name), colors,fmt='%f',delimiter=',')

        ## ----make a obj file of the face model(optional)--------------------------------------------
        #mesh.io.write_obj_with_colors('{}/face_{}'.format(save_folder,name), fitted_vertices, bfm.triangles, colors)

        ## ----make a depth map(optional)-------------------------------------------------------------
        # z = image_vertices[:,2:]
        # z = z - np.min(z)
        # z = z/np.max(z)
        # attribute = z
        # depth_image = mesh.render.render_colors(image_vertices, bfm.triangles, attribute, h, w, c=1)
        # io.imsave(save_folder+'/depth_{}.jpg'.format(name), np.squeeze(depth_image))

        #rotate a proper position(can be set)
        angles = [20, 20, 20]
        
        transformed_vertices = bfm.transform(fitted_vertices, fitted_s, angles, fitted_t)
        projected_vertices = transformed_vertices.copy() # using stantard camera & orth projection
        image_vertices = mesh.transform.to_image(projected_vertices, h, w)
        image = mesh.render.render_colors(image_vertices, bfm.triangles, colors, h, w)

        io.imsave(save_folder+'/rotate_{}.jpg'.format(name), image)

        # ---- rendering Caricature face after deformation --------------------------------------------------------
        
        #objFilePath = 'deformation/deformation_7.obj'
        
        with open(objFilePath) as file:
            points = []
            while 1:
                line = file.readline()
                if not line:
                    break
                strs = line.split(" ")
                if strs[0] == "v":
                    points.append((float(strs[1]), float(strs[2]), float(strs[3])))
                if strs[0] == "vt":
                    break

        vertices = np.array(points)
        #print(vertices[0],vertices[1],vertices.shape)
        triangles = bfm.triangles

        angles = [0, 0, 0]
        
        transformed_vertices = bfm.transform(vertices, fitted_s, angles, fitted_t)
        projected_vertices = transformed_vertices.copy() # using stantard camera & orth projection
        image_vertices = mesh.transform.to_image(projected_vertices, h, w)
        image = mesh.render.render_colors(image_vertices, bfm.triangles, colors, h, w)

        # render 
        rendering =  mesh.render.render_colors(image_vertices, triangles, colors, h, w)

        ## ----make a obj file of the Caricature face model(optional)--------------------------------------------
        mesh.io.write_obj_with_colors('{}/defo_face_{}'.format(save_folder,name), vertices, bfm.triangles, colors)

        ## ----show Caricature face in original image(Very roughly)
        for xx in range(int(centroid_y)-int(h/2),int(centroid_y)+int(h/2)):
            if xx < 0 or xx >= img.shape[0]:
                continue
            xxx = int(xx)+int(h/2)-int(centroid_y)
            for yy in range(int(centroid_x)-int(w/2),int(centroid_x)+int(w/2)):
                if yy < 0 or yy >=img.shape[1]:
                    continue
                yyy = int(yy)+int(w/2)-int(centroid_x)
                if rendering[xxx, yyy][0] != 0 and rendering[xxx, yyy][1] != 0 and rendering[xxx, yyy][2] != 0:
                    img[xx-110,yy] = (rendering[xxx, yyy]).astype(np.uint8)
        win.set_image(img)

        ## ===============================save Caricature 2D face======================================================
        io.imsave('{}/Caricature_{}.jpg'.format(save_folder,name), rendering)

      
        #print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
        #                                          shape.part(1)))

        # Draw the face landmarks on the screen.
        #win.add_overlay(shape)

    #win.add_overlay(dets)
    dlib.hit_enter_to_continue()

