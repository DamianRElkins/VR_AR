from image import Image, Color
from model import Model
from shape import Point, Line, Triangle
from vector import Vector
from obj import Object
import cv2
import numpy as np
import math
import pandas as pd
from IPython.display import display
import random
import time

width = 512
height = 512
image = Image(width, height, Color(255, 255, 255, 255))

# Init z-buffer
zBuffer = [-float('inf')] * width * height

# Load the model
model = Model('data/headset.obj')
model.normalizeGeometry()

model_2 = Model('data/headset_2.obj')
model_2.normalizeGeometry()

model_4 = Model('data/headset_4.obj')
model_4.normalizeGeometry()

frame_skip = 4
num_headsets = 10

def getOrthographicProjection(x, y, z):
	# Convert vertex from world space to screen space
	# by dropping the z-coordinate (Orthographic projection)
	screenX = int((x+1.0)*width/2.0)
	screenY = int((y+1.0)*height/2.0)
	return screenX, screenY

def getPerspectiveProjection(V, x, y, z):
	p = np.array([[x],
			      [y],
			      [z],
			      [1]])
	
	
	p_camera = np.dot(V,p)

	near = 500.0 
	far = 1000.0
	top = -height/2
	right = width/2
	left = -width/2
	bottom = height/2

	Tp = np.array([[near,0,0,0],
				   [0,near,0,0],
				   [0,0,near+far,-far*near],
				   [0,0,1,0]], dtype=float)
	Tst = np.array([[2/(right-left),0,0,-(right+left)/(right-left)],
				    [0,2/(top-bottom),0,-(top+bottom)/(top-bottom)],
					[0,0,2/(near-far),-(near+far)/(near-far)],
					[0,0,0,1]])
	
	P = np.dot(Tst,Tp)

	clip = np.dot(P,p_camera)
	if clip[3] != 0:
		ndc = clip / clip[3]
	else:
		# Handle the case when clip[3] is 0
		ndc = clip


	

	return int(width/2*(ndc[0,0]+1)), int(height/2*(ndc[1,0]+1)), int(1/2*(ndc[2,0]+1)), P

def mag(v):
	return np.sqrt(np.sum(v**2))

def getViewMatrix():
	eye =np.array([0,0,-5], dtype=float)
	center = np.array([0,0,0], dtype=float)
	up = np.array([0,1,0], dtype=float)

	zc = (eye - center)/mag(eye-center)
	xc = np.cross(up,zc)/mag(np.cross(up,zc))
	yc = np.cross(zc,xc)

	R = np.array([[xc[0],xc[1],xc[2],0],
			   	  [yc[0],yc[1],yc[2],0],
			      [zc[0],zc[1],zc[2],0],
			      [0    ,0    ,0    ,1]
			   ], dtype=float)
	
	T = np.array([[1,0,0,-eye[0]],
				  [0,1,0,-eye[1]],
			      [0,0,1,-eye[2]],
			      [0,0,0,1]
				], dtype=float)
	
	return np.dot(R,T), eye, center, up


def translationMatrix(dx,dy,dz):
	trans = np.array([[1,0,0,dx],
			  [0,1,0,dy],
			  [0,0,1,dz],
			  [0,0,0,1]])
	return trans

def rotatex(degrees):
	theta = np.deg2rad(degrees)
	rotate = np.array([[1,0,0,0],
					   [0,np.cos(theta),-np.sin(theta),0],
					   [0,np.sin(theta),np.cos(theta),0],
					   [0,0,0,1]])
	return rotate
def rotatey(degrees):
	theta = np.deg2rad(degrees)
	rotate = np.array([[np.cos(theta),0,np.sin(theta),0],
					   [0,1,0,0],
					   [-np.sin(theta),0,np.cos(theta),0],
					   [0,0,0,1]])
	return rotate
def rotatez(degrees):
	theta = np.deg2rad(degrees)
	rotate = np.array([[np.cos(theta),-np.sin(theta),0,0],
					   [np.sin(theta),np.cos(theta),0,0],
					   [0,0,1,0],
					   [0,0,0,1]])
	return rotate
def scale(sx,sy,sz):
	scale = np.array([[sx,0,0,0],
					  [0,sy,0,0],
					  [0,0,sz,0],
					  [0,0,0,1]])
	return(scale)

def getVertexNormal(vertIndex, faceNormalsByVertex):
	# Compute vertex normals by averaging the normals of adjacent faces
	normal = Vector(0, 0, 0)
	for adjNormal in faceNormalsByVertex[vertIndex]:
		normal = normal + adjNormal

	return normal / len(faceNormalsByVertex[vertIndex])

def quaternion_product(a, b):
	return np.array([(a[0] * b[0]) - (a[1] * b[1]) - (a[2] * b[2]) - (a[3] * b[3]),
					(a[0] * b[1]) + (b[0]* a[1] ) - (a[2] * b[3]) + (b[2] * a[3]),
					(a[0] * b[2]) + (b[3] * a[1]) + (b[0] * a[2]) - (a[3] * b[1]),
					(a[0] * b[3]) - (b[2] * a[1]) + (a[2] * b[1]) + (b[0] * a[3])], dtype=np.float64)


def inv_quaternion(q):
	return q[0], -q[1], -q[2], -q[3]


# https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_(in_3-2-1_sequence)_conversion
def to_quaternion(x,y,z):
	# Abbreviations for the various angular functions
	cr = np.cos(x * 0.5)
	sr = np.sin(x * 0.5)
	cp = np.cos(y * 0.5)
	sp = np.sin(y * 0.5)
	cy = np.cos(z * 0.5)
	sy = np.sin(z * 0.5)

	qw = cr * cp * cy + sr * sp * sy
	qx = sr * cp * cy - cr * sp * sy
	qy = cr * sp * cy + sr * cp * sy
	qz = cr * cp * sy - sr * sp * cy

	return qw,qx,qy,qz

def to_euler(q):

	# Roll (x-axis rotation)
	sinr_cosp = 2 * (q[0] * q[1] + q[2] * q[3])
	cosr_cosp = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
	x = np.arctan2(sinr_cosp, cosr_cosp)

	# Pitch (y-axis rotation)
	sinp = np.sqrt(1 + 2 * (q[0] * q[2] - q[1] * q[3]))
	cosp = np.sqrt(1 - 2 * (q[0] * q[2] - q[1] * q[3]))
	y = 2 * np.arctan2(sinp, cosp) - np.pi / 2

	# Yaw (z-axis rotation)
	siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
	cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
	z = np.arctan2(siny_cosp, cosy_cosp)

	return (x, y, z)

def get_quaternion(axis, angle):

	w = np.cos(angle/2)
	x = axis[0] * np.sin(angle/2)    
	y = axis[1] * np.sin(angle/2)
	z = axis[2] * np.sin(angle/2)

	return [w, x, y, z]

	
imudata = pd.read_csv('../IMUData.csv')
display(imudata)
for column in imudata.columns:
	 if column in [" gyroscope.X", " gyroscope.Y", " gyroscope.Z"]:
		  imudata[column] = imudata[column] * (np.pi/180)
display(imudata)


image_list = []
startTime = time.time()
def render_model(objects,orientation, frame):
	V, eye, center, up = getViewMatrix()
	global image_list
	global startTime
	width = 512
	height = 512
	image = Image(width, height, Color(255, 255, 255, 255))

	# Init z-buffer
	zBuffer = [-float('inf')] * width * height
	for obj in objects:
		model = obj.model
		faceNormals = {}
		for face in model.faces:
			p0, p1, p2 = [model.vertices[i] for i in face]
			faceNormal = (p2 - p0).cross(p1 - p0).normalize()
			for i in face:
				if i not in faceNormals:
					faceNormals[i] = []

				faceNormals[i].append(faceNormal)

		# Calculate vertex normals
		vertexNormals = []
		for vertIndex in range(len(model.vertices)):
			vertNorm = getVertexNormal(vertIndex, faceNormals)
			vertexNormals.append(vertNorm)
		for face in model.faces:
			p0, p1, p2 = [model.vertices[i] for i in face]
			n0, n1, n2 = [vertexNormals[i] for i in face]

			# Define the light direction
			lightDir = Vector(0, 0, -1)

			# Set to true if face should be culled
			cull = False


			# Transform vertices and calculate lighting intensity per vertex
			transformedPoints = []
			if obj.static:
				for p, n in zip([p0, p1, p2], [n0, n1, n2]):
					
					
					quatp = (0,p.y,p.z,p.x)

					qp = quaternion_product(orientation,quatp)
					qw, qy, qz, qx = quaternion_product(qp,inv_quaternion(orientation))

					r_p =  obj.translationMatrix() @ np.array([[qx],[qy],[qz],[1]])
					
					screenX, screenY, screenZ,P = getPerspectiveProjection(V, r_p[0,0], r_p[1,0], r_p[2,0])

					n_p = quaternion_product(orientation,(0,n.y,n.z,n.x))
					nw,ny,nz,nx = quaternion_product(n_p,inv_quaternion(orientation))
				
					intensity = Vector(nx,ny,nz).normalize()*lightDir


					# intensity = n*lightDir

					if intensity < 0:
						intensity = 0
						# cull = True
					

					transformedPoints.append(Point(screenX, screenY, qz, Color(intensity * 255, intensity * 255, intensity * 255, 255)))
				
				if not cull:
					Triangle(transformedPoints[0], transformedPoints[1], transformedPoints[2]).draw_faster(image, zBuffer)
			else:
				for p, n in zip([p0, p1, p2], [n0, n1, n2]):
					
					t_orientation = get_quaternion((obj.axis[1],obj.axis[2],obj.axis[0]),obj.a_angle)
					quatp = (0,p.y,p.z,p.x)

					qp = quaternion_product(t_orientation,quatp)
					qw, qy, qz, qx = quaternion_product(qp,inv_quaternion(t_orientation))
					
					r_p =  obj.translationMatrix() @ np.array([[qx],[qy],[qz],[1]])

					x,y,z = r_p[0,0],r_p[1,0],r_p[2,0]

					screenX, screenY, screenZ,P = getPerspectiveProjection(V, x, y, z)

					n_p = quaternion_product(t_orientation,(0,n.y,n.z,n.x))
					nw,ny,nz,nx = quaternion_product(n_p,inv_quaternion(t_orientation))

					r_n = np.array([[nx],[ny],[nz],[1]]) 

					nx,ny,nz = r_n[0,0],r_n[1,0],r_n[2,0]

					intensity = Vector(nx,ny,nz).normalize()*lightDir

					# intensity = n*lightDir

					if intensity < 0:
						intensity = 0
						# cull = True

					if qz < -5 or screenX < 20 or screenY < -20 or screenX >532 or screenY > 532: 
						cull = True

					transformedPoints.append(Point(screenX, screenY, screenZ, Color(intensity * 255, intensity * 255, intensity * 255, 255)))
				
				if not cull:
					Triangle(transformedPoints[0], transformedPoints[1], transformedPoints[2]).draw_faster(image, zBuffer)
				

	ocv_img = image.show()
	font = cv2.FONT_HERSHEY_SIMPLEX
	font_scale = 1
	font_color = (0, 0, 0)  # White color
	thickness = 2  # Thickness of the text
	text_size = cv2.getTextSize(str(frame), font, font_scale, thickness)[0]
	text_x = 10  # X coordinate of the text
	text_y = 30  # Y coordinate of the text (adjust as needed)

	cv2.putText(ocv_img, f"Frame {frame} FPS: {frame/(time.time()-startTime)/frame_skip:.2f}", (text_x, text_y), font, font_scale, font_color, thickness)
	
	cv2.imshow("render",ocv_img)
	cv2.waitKey(1)
	# Append the current frame to the list
	image_list.append(ocv_img)

def getGlobalAccel(a,g):
	return quaternion_product(g,quaternion_product(a,inv_quaternion(g)))

def complimentary_filter(objects,imudata, useAccel = False, useMag = False, LevelOD = True):
	orientations = [[1,0,0,0]]
	global image_list
	image_list = []
	global frame_skip
	half_dist = 15
	quarter_dist = 25
	alpha = 0.01
	alpha2 = 0.1
	for frame in range(len(imudata)):
		prev = orientations[-1]

		current_imu = imudata.iloc[frame]

		gx = current_imu[" gyroscope.X"]
		gy = current_imu[" gyroscope.Y"]
		gz = current_imu[" gyroscope.Z"]

		ax = current_imu[" accelerometer.X"]
		ay = current_imu[" accelerometer.Y"]
		az = current_imu[" accelerometer.Z"] 

		magnitude = np.sqrt((gx**2) + (gy**2) +(gz**2))
		
		# compute instantaneous axis of rotation
		rotation_axis = np.array([gy,gz,gx], dtype=np.float64) / magnitude	

		# compute amount of rotation
		rotation_angle = magnitude/256

		# compute estimated rotation axis
		orientation_change = get_quaternion(rotation_axis, rotation_angle)
		
		# compute quaternion change
		orientation = quaternion_product(orientation_change, prev)

		orientations.append(orientation)

		if useAccel:
			a = np.array([0,ay,az,ax],dtype=float)
			a_mag = np.linalg.norm(a) 
			a = a/a_mag if a_mag != 0 and a_mag != np.nan else [0,0,1,0]
			worldAccel = quaternion_product(quaternion_product(orientation,a),inv_quaternion(orientation))
			worldAccel = worldAccel[1:]
			worldAccel = worldAccel/np.linalg.norm(worldAccel)
			wa_x,wa_y,wa_z = worldAccel
			tiltAxis = np.array([wa_z,0,-wa_x])
			tiltAxis = tiltAxis/np.linalg.norm(tiltAxis)
			phi = np.arccos(wa_y)

			
			a_q = get_quaternion(tiltAxis,-alpha * phi)


			orientation = quaternion_product(orientation,a_q)

		if useMag:
			if frame == 0:
				mag_0 = np.array([0,current_imu[' magnetometer.Y'],current_imu[' magnetometer.Z '],current_imu[' magnetometer.X']],dtype=float)
				refx,refy,refz = quaternion_product(inv_quaternion(orientation),quaternion_product(mag_0,orientation))[1:]
			
			mag_i = np.array([0,current_imu[' magnetometer.Y'],current_imu[' magnetometer.Z '],current_imu[' magnetometer.X']],dtype=float)
			mx,my,mz = quaternion_product(inv_quaternion(orientation),quaternion_product(mag_i,orientation))[1:]
			
			
			theta = np.arctan2(mx,mz)
			theta_r = np.arctan2(refx,refz)

			yaw_q = get_quaternion(np.array([0,1,0]),-alpha2 * (theta-theta_r))
			orientation = quaternion_product(yaw_q,orientation)


	
		for obj in objects:
			obj.update(1/256)
			if obj.distance(np.array([0,0,-5])) > quarter_dist and LevelOD:
				obj.model = model_4
			elif obj.distance(np.array([0,0,-5])) > half_dist and LevelOD:
				obj.model = model_2
			else:
				obj.model = model


		if frame%frame_skip == 0:
			# Render the image iterating through facses
			render_model(objects,orientation,frame)

	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter(f'comp_{useAccel}{alpha}_{useMag}{alpha2}_LOD_{LevelOD}.avi', fourcc, 256/frame_skip, (image_list[0].shape[1], image_list[0].shape[0]))

	# Write each frame to the video file
	for frame in image_list:
		out.write(frame)

	# Release the VideoWriter object
	out.release()

objects = []
random.seed(42)
for n in range(num_headsets):
	z = random.random()*40+10
	loc = np.array([random.random()*z-(z/2),random.random()*z*2,z])
	speed = 30 * random.random()
	v_loc = np.array([-loc[0],-loc[1]+10,-loc[2]])
	velocity = v_loc / np.linalg.norm(v_loc) * speed
	angle = np.array([ random.random()*2*np.pi, random.random()*2*np.pi, random.random()*2*np.pi])
	objects.append(Object(model,loc,velocity,angle,False))
objects.append(Object(model,[0,0,0],[0,0,0],[0,0,0],True))
# dead_reckoning_filter(model,imudata)
complimentary_filter(objects,imudata,True, False, True)
complimentary_filter(objects,imudata,True, True, True)
cv2.destroyAllWindows()  # Close the window after a key is pressed