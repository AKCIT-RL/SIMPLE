import mujoco
spec = mujoco.MjSpec()
body = spec.worldbody.add_body(name="test")
body.add_camera(name="cam1")
cams = spec.worldbody.find_all("camera")
print([c.name for c in cams])
