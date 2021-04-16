import raisimpy as raisim
import os


raisim.World.setLicenseFile("/home/ami/raisim/activation.raisim")
world = raisim.World()
ground = world.addGround()

# launch raisim server
server = raisim.RaisimServer(world)
server.launchServer(8080)



server.killServer()