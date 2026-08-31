from pathlib import Path

import imageio.v3 as iio
import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = PROJECT_ROOT / "embodied_vla/assets/so_arm100/pick_place.xml"
OUTPUT_PATH = PROJECT_ROOT / "outputs/mujoco_lesson00.png"


def main() -> None:
    # MjModel stores the fixed model; MjData stores the changing simulation state.
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)

    grip_site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "grip_site",
    )

    mujoco.mj_forward(model, data)
    initial_grip_position = data.site_xpos[grip_site_id].copy()

    for _ in range(100):
        mujoco.mj_step(model, data)

    # The scene XML uses MuJoCo's default 256-pixel offscreen framebuffer.
    # For a larger image, add <visual><global offwidth="640"/></visual> to the XML.
    renderer = mujoco.Renderer(model, height=256, width=256)
    renderer.update_scene(data, camera="front")
    image = renderer.render()
    renderer.close()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(OUTPUT_PATH, image)

    print(f"MuJoCo version: {mujoco.__version__}")
    print(f"XML: {XML_PATH}")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(f"physics timestep={model.opt.timestep:.4f} s")
    print(f"initial grip_site position={initial_grip_position.round(4)}")
    print(f"simulated time={data.time:.3f} s")
    print(f"rendered image={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
