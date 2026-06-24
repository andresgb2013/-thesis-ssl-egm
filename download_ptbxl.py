import urllib.request
import os

base     = "https://physionet.org/files/ptb-xl/1.0.3/records100/"
save_dir = "data/raw/ptb-xl/records100"
folders  = [f"{i:05d}" for i in range(0, 22000, 1000)]

total   = 0
skipped = 0

for folder in folders:
    folder_dir = os.path.join(save_dir, folder)
    os.makedirs(folder_dir, exist_ok=True)

    for i in range(1, 1001):
        rec_num = int(folder) + i
        if rec_num > 21837:
            break

        fname    = f"{rec_num:05d}_lr"
        dest_hea = os.path.join(folder_dir, fname + '.hea')

        # skip already downloaded
        if os.path.exists(dest_hea):
            skipped += 1
            total   += 1
            continue

        try:
            for ext in ['.hea', '.dat']:
                url  = f"{base}{folder}/{fname}{ext}"
                dest = os.path.join(folder_dir, fname + ext)
                urllib.request.urlretrieve(url, dest)

            total += 1
            if total % 100 == 0:
                print(f"✓ {total}/21837 downloaded")

        except Exception as e:
            pass

print(f"\n✓ Complete — {total} records ready ({skipped} already existed)")