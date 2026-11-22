"""Shared 3D target markers and a two-view recorded presentation."""
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def markers(scene, metric):
    for name, color, length, width in (("target", [0., .85, 1., 1.], .17, .0035),
                                       ("actual", [1., .48, .12, 1.], .11, .006)):
        p, r = metric[name + "_p"], metric[name + "_r"]
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.ones(3)*.014,
                          p, np.eye(3).ravel(), np.array(color, dtype=np.float32))
        scene.ngeom += 1
        for i, rgb in enumerate(([1., .2, .25, 1.], [.25, 1., .4, 1.], [.25, .55, 1., 1.])):
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                              np.eye(3).ravel(), np.array(rgb, dtype=np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, p, p+length*r[:, i])
            scene.ngeom += 1


class Video:
    def __init__(self, backend, path):
        self.backend = backend
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.writer = imageio.get_writer(path, fps=25, codec="libx264", quality=8,
                                        macro_block_size=1, ffmpeg_params=["-movflags", "+faststart"])
        self.main = mujoco.Renderer(backend.model, width=896, height=560)
        self.detail = mujoco.Renderer(backend.model, width=384, height=360)
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.15, 135., -22.
        self.close_cam = mujoco.MjvCamera()
        self.close_cam.distance, self.close_cam.azimuth, self.close_cam.elevation = 1.15, 125., -15.
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[3] = 0
        font = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
        self.font = ImageFont.truetype(str(font), 17) if font.exists() else ImageFont.load_default()
        self.title = ImageFont.truetype(str(font), 25) if font.exists() else self.font
        self.history = []

    def frame(self, metric, label, source):
        b = self.backend
        self.cam.lookat[:] = b.data.xpos[b.base] + [.15, 0, -.03]
        self.close_cam.lookat[:] = .5*(metric['actual_p']+metric['target_p'])
        canvas = Image.new('RGB', (1280, 720), '#101b29')
        for renderer, camera, origin in ((self.main, self.cam, (0, 80)),
                                          (self.detail, self.close_cam, (896, 80))):
            renderer.update_scene(b.data, camera=camera, scene_option=self.opt)
            markers(renderer.scene, metric)
            canvas.paste(Image.fromarray(renderer.render()), origin)
        draw = ImageDraw.Draw(canvas)
        draw.text((22, 12), 'A2W + Z1  |  Precision PPO10K', font=self.title, fill='#f2f7ff')
        draw.text((22, 48), f'{label}   |   {source}   |   t = {metric["time"]:05.2f} s',
                  font=self.font, fill='#9fb5cc')
        draw.text((914, 91), 'END-EFFECTOR DETAIL', font=self.font, fill='white', stroke_width=1, stroke_fill='#172638')
        values = [('EE position', f'{metric["ee_cm"]:.2f} cm'),
                  ('EE rotation', f'{metric["orientation_deg"]:.2f} deg'),
                  ('Base height / tilt', f'{metric["base_height"]:.2f} m / {metric["tilt_deg"]:.1f} deg')]
        for i, (name, value) in enumerate(values):
            draw.text((914, 456+28*i), name, font=self.font, fill='#9fb5cc')
            draw.text((1100 if i<2 else 1080, 456+28*i), value, font=self.font, fill='#ffffff')
        v, a = metric['velocity_command'], metric['actual_velocity']
        draw.text((914, 548), 'vx / vy / yaw rate', font=self.font, fill='#9fb5cc')
        draw.text((914, 574), f'CMD  {v[0]:+.2f}  {v[1]:+.2f}  {v[2]:+.2f}', font=self.font, fill='#3bd8f5')
        draw.text((914, 600), f'ACT   {a[0]:+.2f}  {a[1]:+.2f}  {a[2]:+.2f}', font=self.font, fill='#ffad54')
        draw.text((22, 650), 'Cyan origin + long axes: TARGET     Orange origin + short axes: ACTUAL', font=self.font, fill='#d2e2f2')
        draw.text((22, 680), 'RGB axes = X/Y/Z  |  Floating base + wheel contact  |  50 Hz PPO / 200 Hz physics  |  No IK',
                  font=self.font, fill='#9fb5cc')
        self.writer.append_data(np.asarray(canvas))
        return canvas

    def close(self):
        self.writer.close()
        self.main.close()
        self.detail.close()
