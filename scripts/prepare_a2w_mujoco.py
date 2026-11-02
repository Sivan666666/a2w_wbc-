"""Build MuJoCo MJCF from the same private URDF used by Isaac Lab."""
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import trimesh

p=argparse.ArgumentParser()
p.add_argument("--asset-dir",type=Path,required=True)
p.add_argument("--output",type=Path,required=True)
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
tree=ET.parse(a.asset_dir/"a2wz1_wbc.urdf")
root=tree.getroot()
compiler=root.find("mujoco/compiler")
compiler.set("meshdir",str((a.output/"meshes").resolve()))
compiler.set("fusestatic","false")
compiler.set("discardvisual","false")
meshdir=a.output/"meshes"
meshdir.mkdir(exist_ok=True)
converted={}
for e in root.iter("mesh"):
    src=Path(e.get("filename"))
    out=meshdir/(src.stem+".obj")
    if str(src) not in converted:
        mesh=trimesh.load(str(src),force="mesh",process=False)
        mesh.export(str(out))
        converted[str(src)]=str(out.resolve())
    e.set("filename",str(out.resolve()))
temp=a.output/"converted.urdf"
tree.write(temp,encoding="utf-8",xml_declaration=True)
model=mujoco.MjModel.from_xml_path(str(temp))
compiled=a.output/"robot.xml"
mujoco.mj_saveLastXML(str(compiled),model)
tree=ET.parse(compiled)
root=tree.getroot()
world=root.find("worldbody")
base=world.find("body")
assert base.get("name")=="base",base.get("name")
base.set("pos","0 0 0.55")
ET.SubElement(base,"freejoint",name="floating_base_joint")
# Isaac configuration disables robot self-collision. Keep only robot-ground contact here.
for geom in base.iter("geom"):
    if geom.get("contype") != "0":
        geom.set("contype","2")
        geom.set("conaffinity","1")
    geom.set("friction","1 0.005 0.0001")
ET.SubElement(world,"geom",name="floor",type="plane",size="100 100 .1",rgba=".7 .72 .75 1",contype="1",conaffinity="2",friction="1 .005 .0001")
ET.SubElement(world,"light",pos="0 0 4",dir="0 0 -1",diffuse=".8 .8 .8")
option=root.find("option")
if option is None:
    option=ET.SubElement(root,"option")
option.set("timestep","0.005")
option.set("gravity","0 0 -9.81")
act=root.find("actuator")
if act is None:
    act=ET.SubElement(root,"actuator")
for j in root.findall(".//body/joint"):
    name=j.get("name")
    if not name:
        continue
    limit=24.5 if "wheel" in name else (180. if "calf" in name else (60. if name=="joint2" else (30. if name.startswith("joint") else 120.)))
    j.set("armature","0.01")
    j.set("damping","0")
    j.set("frictionloss","0")
    ET.SubElement(act,"motor",name=name+"_motor",joint=name,ctrllimited="true",ctrlrange=f"{-limit} {limit}")
ET.indent(tree)
scene=a.output/"scene.xml"
tree.write(scene,encoding="utf-8",xml_declaration=True)
m=mujoco.MjModel.from_xml_path(str(scene))
assert m.nu==22,(m.nu,m.nq,m.nv)
report={"urdf_sha256":hashlib.sha256((a.asset_dir/"a2wz1_wbc.urdf").read_bytes()).hexdigest(),
        "nq":m.nq,"nv":m.nv,"nu":m.nu,"mass":float(m.body_mass.sum()),"converted_meshes":converted}
(a.output/"manifest.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k!="converted_meshes"},indent=2))
