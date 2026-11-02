"""Prepare a private dynamic A2W+Z1 URDF; original door assets stay untouched."""
import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--destination", type=Path, required=True)
a = p.parse_args()
src = a.source / "urdf/a2wz1.urdf"
dst = a.destination
dst.mkdir(parents=True, exist_ok=True)
shutil.copytree(a.source / "meshes", dst / "meshes", dirs_exist_ok=True)
tree = ET.parse(src)
root = tree.getroot()
for e in list(root):
    if (e.tag == "link" and e.get("name") == "world") or (e.tag == "joint" and e.get("name") == "floating_base_joint"):
        root.remove(e)
rename = {"base_link": "base", "base": "z1_mount", "link00": "link0", "ee_gripper_link": "end_effector"}
for e in root.iter():
    if e.tag == "link":
        e.set("name", rename.get(e.get("name"), e.get("name")))
    if e.tag in ("parent", "child"):
        e.set("link", rename.get(e.get("link"), e.get("link")))
    if e.tag == "mesh":
        e.set("filename", str((dst / "meshes" / Path(e.get("filename")).name).resolve()))
for j in root.findall("joint"):
    if j.get("name", "").endswith("_wheel_joint"):
        j.set("type", "continuous")
        for k in ("lower", "upper"):
            j.find("limit").attrib.pop(k, None)
    if j.get("name") == "jointGripper":
        j.set("type", "fixed")
        for tag in ("limit", "dynamics", "axis"):
            for e in j.findall(tag):
                j.remove(e)
added_inertia = []
for link in root.findall("link"):
    if link.find("inertial") is None:
        i = ET.SubElement(link, "inertial")
        ET.SubElement(i, "mass", value="0.000001")
        ET.SubElement(i, "inertia", ixx="1e-9", iyy="1e-9", izz="1e-9", ixy="0", ixz="0", iyz="0")
        added_inertia.append(link.get("name"))
ET.indent(tree)
out = dst / "a2wz1_wbc.urdf"
tree.write(out, encoding="utf-8", xml_declaration=True)
manifest = {"source": str(src), "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "output_sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "renamed_frames": rename,
            "notes": ["free root supplied by simulator, not world URDF link", "four continuous wheel joints retain torque/speed limits", "gripper fixed at source zero pose: WBC does not control grasp", "no mesh/shape or existing inertia changes"],
            "massless_frames_given_1mg": added_inertia,
            "active_joints": [j.get("name") for j in root.findall("joint") if j.get("type") != "fixed"]}
(dst / "asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))
