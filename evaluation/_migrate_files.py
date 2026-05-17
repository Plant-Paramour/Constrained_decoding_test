import os
import shutil

base = r'C:\code\Constrained_decoding_test_bio\evaluation\evaluate_input'
songci_dir = os.path.join(base, 'Songci')

os.makedirs(songci_dir, exist_ok=True)

for f in os.listdir(base):
    src = os.path.join(base, f)
    if os.path.isfile(src):
        dst = os.path.join(songci_dir, f)
        print(f'Moving: {f}')
        shutil.move(src, dst)
        print(f'  -> {dst}')

print('Migration complete.')
print('Songci dir contents:', os.listdir(songci_dir))
