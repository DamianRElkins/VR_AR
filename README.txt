In order to run a sequence the complimentary_filter function can be used. It is passed a list of objects, the IMU data, and boolean values indicating whether it should use accelerometer readings, magnetometer readings, and LOD.
In its current state, running render.py will first render a sequence using gyroscope and accelerometer fused readings. After it will render all fused readings.
This is done using pandas, numpy, and opencv as new packages. 

The image class has been granted a new function to return an opencv compatible image for real time rendering.

A new object class has been created to handle instances of models including gravity and air resistance. 
