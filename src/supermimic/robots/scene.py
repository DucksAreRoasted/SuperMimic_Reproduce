"""Turn an official MJCF that references meshes into a mesh-free scene MuJoCo can compile.

Inertial parameters, joint axes, and joint limits are kept. Collision is foot-to-floor
plus primitive geoms that were already in the file. Self-collision is off: robot geoms
use contype 1 and conaffinity 0, the floor uses both bits.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from supermimic.robots.actuators import ActuatorSpec


def _parent_map(root: ET.Element) -> dict[ET.Element, ET.Element]:
    return {child: parent for parent in root.iter() for child in parent}


def prepare_scene(
    raw_xml: Path,
    out_xml: Path,
    actuators: dict[str, ActuatorSpec],
    foot_bodies: tuple[str, str],
) -> Path:
    tree = ET.parse(raw_xml)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
    for asset in list(root.findall("asset")):
        for mesh in list(asset.findall("mesh")):
            asset.remove(mesh)
        for tex in list(asset.findall("texture")):
            asset.remove(tex)
        for mat in list(asset.findall("material")):
            asset.remove(mat)

    parents = _parent_map(root)
    for geom in list(root.iter("geom")):
        if geom.get("type") == "mesh" or geom.get("mesh") is not None:
            parent = parents[geom]
            parent.remove(geom)
            continue
        if geom.get("type") == "plane":
            geom.set("name", "floor")
            geom.set("friction", "1 0.005 0.0001")
            geom.set("contype", "1")
            geom.set("conaffinity", "1")
            geom.attrib.pop("material", None)
            continue
        geom.set("contype", "1")
        geom.set("conaffinity", "0")
        geom.set("density", "0")

    body_by_name = {b.get("name"): b for b in root.iter("body") if b.get("name")}
    for side, body_name in zip(("left", "right"), foot_bodies):
        body = body_by_name[body_name]
        name = f"{side}_foot_collision"
        existing = None
        for geom in body.findall("geom"):
            if geom.get("type") == "box":
                existing = geom
                break
        if existing is None:
            existing = ET.SubElement(body, "geom")
            existing.set("type", "box")
            existing.set("pos", "0.04 0 -0.02")
            existing.set("size", "0.09 0.035 0.015")
        existing.set("name", name)
        existing.set("friction", "1 0.005 0.0001")
        existing.set("contype", "1")
        existing.set("conaffinity", "0")
        existing.set("density", "0")

    if root.find("worldbody/geom[@name='floor']") is None and root.find(".//geom[@name='floor']") is None:
        world = root.find("worldbody")
        assert world is not None
        floor = ET.Element("geom")
        floor.set("name", "floor")
        floor.set("type", "plane")
        floor.set("pos", "0 0 0")
        floor.set("size", "0 0 0.05")
        floor.set("friction", "1 0.005 0.0001")
        floor.set("contype", "1")
        floor.set("conaffinity", "1")
        world.insert(0, floor)

    hinge_names: list[str] = []
    world = root.find("worldbody")
    assert world is not None
    for joint in world.iter("joint"):
        if joint.get("type", "hinge") == "free":
            continue
        name = joint.get("name")
        assert name is not None
        hinge_names.append(name)
        spec = actuators[name]
        joint.set("armature", f"{spec.armature:.8g}")
        joint.set("damping", "0")
        joint.set("frictionloss", "0")

    for child in list(root):
        if child.tag in ("actuator", "contact"):
            root.remove(child)
    actuator = ET.SubElement(root, "actuator")
    for name in hinge_names:
        spec = actuators[name]
        pos = ET.SubElement(actuator, "position")
        pos.set("name", name)
        pos.set("joint", name)
        pos.set("kp", f"{spec.kp:.8g}")
        pos.set("kv", f"{spec.kd:.8g}")
        pos.set("forcerange", f"{-spec.effort:.8g} {spec.effort:.8g}")
        pos.set("ctrllimited", "false")

    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(0, option)
    option.set("timestep", "0.005")
    option.set("gravity", "0 0 -9.81")
    option.set("integrator", "implicitfast")

    out_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_xml, encoding="unicode", xml_declaration=False)
    return out_xml
