import numpy as np
class Object(object):
    def __init__(self,
                 model,
                 location,
                 velocity,
                 angle,
                 static = False
                 ):
        self.model = model
        self.location = location
        self.velocity = velocity
        self.acceleration = np.array([0,0,0],dtype=float)
        self.static = static
        self.angle = angle
        self.angularVel = np.array([0,0,0],dtype=float)
        self.angularAccel = np.array([0,0,0],dtype=float)
        self.axis = [0,0,0]
        self.a_angle = 0
        self.mass = 1.0
        self.gravity = np.array([0,-9.8,0])
        self.weight = self.mass * self.gravity
        self.cd = 0.8
        self.density = 1.3
        self.area = 0.2
        self.drag = 0 

    def updateVel(self,dt):
        self.velocity = self.acceleration*dt + self.velocity
        self.angularVel = self.angularAccel*dt + self.angularVel 
    
    def updateLoc(self,dt):
        if not self.static:
            self.location = self.location + self.velocity * dt
            self.angle = self.angle + self.angularVel * dt
            if self.location[1] < -25:
                self.location[1] = 25
                self.velocity = np.array([self.velocity[0],0,self.velocity[2]])
            if self.location[2] < -10:
                self.location[2] = 30
                self.velocity = np.array([self.velocity[0],self.velocity[1],self.velocity**2])


    def updateAccel(self, dt):
        vsq = np.square(np.linalg.norm(self.velocity))
        D = 1/2 * self.cd * self.density * vsq * self.area 
        self.acceleration = (self.weight - D * (self.velocity/np.linalg.norm(self.velocity))) / self.mass 

    
    def axisAngle(self):
        mag = np.linalg.norm(self.angle)
        if mag != 0 and mag != np.nan:
            self.axis = self.angle/mag
            self.a_angle = mag
        else:
            self.axis = self.angle
            self.a_angle = 0

    def update(self,dt):
        if not self.static:
            self.updateAccel(dt)
            self.updateVel(dt)
            self.updateLoc(dt)
            self.axisAngle()

    def distance(self,eye):
        dists = self.location - eye
        return np.sqrt(dists[0]**2 + dists[1]**2 + dists[2]**2)
    
    def translationMatrix(self):
        trans = np.array([[1,0,0,self.location[0]],
                [0,1,0,self.location[1]],
                [0,0,1,self.location[2]],
                [0,0,0,1]])
        return trans
