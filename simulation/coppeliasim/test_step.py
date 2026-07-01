from coppeliasim_zmqremoteapi_client import RemoteAPIClient

client = RemoteAPIClient()
sim = client.require('sim')

print("Connected")

# Start simulation
sim.startSimulation()
sim.setStepping(True)

# Step a bit
for _ in range(10):
    sim.step()

print("Stepping active")

# Get script handle
scripts = sim.getObjectsInTree(sim.handle_scene, sim.object_script_type)

if not scripts:
    raise RuntimeError("No script found!")

script_handle = scripts[0]

# 🔥 Single call (no request/response anymore)
res = sim.callScriptFunction(
    'createBox',
    script_handle,
    [],
    [0.1, 0.1, 0.1],
    [],
    ''
)

handle = res[0][0]
print("Handle:", handle)

# Step so object appears
for _ in range(10):
    sim.step()

sim.stopSimulation()