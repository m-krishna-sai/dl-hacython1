import sys
sys.path.insert(0, '.')
from PIL import Image
from backend.api.inference import process_single_image

tests = [
    ('test_images/1_lung/normal.jpg', 'lung'),
    ('test_images/1_lung/pneumonia.jpg', 'lung'),
    ('test_images/2_brain/normal.jpg', 'brain'),
    ('test_images/2_brain/tumor.jpg', 'brain'),
    ('test_images/3_skin/normal_benign_nevus.jpg', 'skin'),
    ('test_images/3_skin/malignant_melanoma.jpg', 'skin'),
    ('test_images/4_retina/healthy_normal.jpg', 'retina'),
    ('test_images/4_retina/diabetic_retinopathy.jpg', 'retina'),
]

print('--- TrustMed Inference Test ---')
for path, spec in tests:
    try:
        img = Image.open(path).convert('RGB')
        res = process_single_image(img, filename=path, requested_specialty=spec)
        fname = path.split('/')[-1]
        pred = res.get('predicted_class', res.get('primary_finding', 'N/A'))
        conf = round(res.get('confidence', 0) * 100, 1)
        normal = res.get('is_normal', '?')
        print(f"[{spec.upper():6s}] {fname:35s} => {pred:35s} ({conf}%)  normal={normal}")
    except Exception as e:
        print(f"[{spec.upper()}] {path} => ERROR: {e}")
        import traceback; traceback.print_exc()
