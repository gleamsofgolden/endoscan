import os
import numpy as np
import nibabel as nib
from PIL import Image
from pathlib import Path

for split in ['train', 'val']:
    for label in ['positive', 'negative']:
        folder = Path(f'data/{split}/{label}')
        for f in folder.glob('*.nii.gz'):
            try:
                img = nib.load(str(f))
                data = img.get_fdata()
                data = np.rot90(data)
                if data.ndim == 3:
                    mid = data.shape[2] // 2
                    slice_ = data[:, :, mid]
                else:
                    slice_ = data
                slice_ = (slice_ - slice_.min()) / (slice_.max() - slice_.min() + 1e-8) * 255
                png = Image.fromarray(slice_.astype('uint8')).convert('RGB')
                out = f.with_suffix('').with_suffix('.png')
                png.save(out)
                os.remove(f)
                print(f'Converted {f.name}')
            except Exception as e:
                print(f'Skipped {f.name}: {e}')

print('Done!')
