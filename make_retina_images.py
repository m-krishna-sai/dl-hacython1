"""
Generate synthetic retina fundus-style test images that work with the APTOS 2019-trained model.
The APTOS model was trained on fundus photographs (circular orange disc images).
We generate:
  - healthy_normal.jpg : clean orange circular fundus (no lesions)
  - diabetic_retinopathy.jpg : orange circular fundus WITH bright yellow exudates and dark hemorrhages
"""
import numpy as np
from PIL import Image
import os

os.makedirs('test_images/4_retina', exist_ok=True)

def make_fundus(width=512, with_lesions=False):
    """Create a synthetic circular fundus image."""
    img = np.zeros((width, width, 3), dtype=np.uint8)
    cx, cy, r = width // 2, width // 2, int(width * 0.46)

    # Build pixel grid
    Y, X = np.ogrid[:width, :width]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask = dist < r  # circular disc mask

    # Fill the circular fundus with dark orange-red background (like a real fundus photo)
    # Gradient: slightly brighter near center
    for y in range(width):
        for x in range(width):
            if mask[y, x]:
                d = dist[y, x] / r  # 0 = center, 1 = edge
                # Fundus base color: dark orange-red background
                r_val = int(120 - d * 40)
                g_val = int(50 - d * 20)
                b_val = int(20 - d * 10)
                img[y, x] = [max(0, r_val), max(0, g_val), max(0, b_val)]

    # Add optic disc (bright yellow-white circle at center-right)
    disc_x, disc_y = int(cx + r * 0.25), cy
    disc_r = int(r * 0.12)
    disc_mask = np.sqrt((X - disc_x) ** 2 + (Y - disc_y) ** 2) < disc_r
    img[disc_mask & mask] = [255, 240, 180]  # bright yellowish-white

    # Add blood vessels (thin dark red lines radiating from disc)
    rng = np.random.RandomState(42)
    for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        for t in np.linspace(0, 1, 200):
            vx = int(disc_x + t * r * 0.7 * np.cos(angle + 0.3 * t))
            vy = int(disc_y + t * r * 0.7 * np.sin(angle + 0.3 * t))
            if 0 <= vx < width and 0 <= vy < width and mask[vy, vx]:
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = vy + dy, vx + dx
                        if 0 <= ny < width and 0 <= nx < width and mask[ny, nx]:
                            img[ny, nx] = [80, 20, 10]  # dark red vessel

    if with_lesions:
        # Add bright yellow exudates (small bright dots scattered in retina)
        rng2 = np.random.RandomState(7)
        for _ in range(20):
            ex = int(cx + rng2.uniform(-0.4, 0.4) * r)
            ey = int(cy + rng2.uniform(-0.4, 0.4) * r)
            er = rng2.randint(5, 15)
            exudate_mask = np.sqrt((X - ex) ** 2 + (Y - ey) ** 2) < er
            img[exudate_mask & mask] = [255, 230, 50]  # bright yellow exudate

        # Add dark hemorrhages (dark red/brown blotches)
        rng3 = np.random.RandomState(13)
        for _ in range(12):
            hx = int(cx + rng3.uniform(-0.35, 0.35) * r)
            hy = int(cy + rng3.uniform(-0.35, 0.35) * r)
            hr = rng3.randint(8, 20)
            hemo_mask = np.sqrt((X - hx) ** 2 + (Y - hy) ** 2) < hr
            img[hemo_mask & mask] = [40, 5, 5]  # dark hemorrhage

        # Add micro-aneurysms (tiny dark dots)
        rng4 = np.random.RandomState(99)
        for _ in range(30):
            mx = int(cx + rng4.uniform(-0.5, 0.5) * r)
            my = int(cy + rng4.uniform(-0.5, 0.5) * r)
            ma_mask = np.sqrt((X - mx) ** 2 + (Y - my) ** 2) < 4
            img[ma_mask & mask] = [30, 0, 0]

    return Image.fromarray(img, 'RGB')

# Normal fundus
normal_img = make_fundus(width=512, with_lesions=False)
normal_img.save('test_images/4_retina/healthy_normal.jpg', quality=95)
print(f"Saved healthy_normal.jpg — size: {os.path.getsize('test_images/4_retina/healthy_normal.jpg')} bytes")

# DR fundus
dr_img = make_fundus(width=512, with_lesions=True)
dr_img.save('test_images/4_retina/diabetic_retinopathy.jpg', quality=95)
print(f"Saved diabetic_retinopathy.jpg — size: {os.path.getsize('test_images/4_retina/diabetic_retinopathy.jpg')} bytes")

print("Done. Now run test_inference.py to verify.")
