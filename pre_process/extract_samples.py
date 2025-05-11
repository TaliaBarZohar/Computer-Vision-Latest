import argparse
import os
import numpy as np
import joblib
from datasets.dataset import get_dataset, img_batch_tensor2numpy


def samples_extraction(dataset_root, dataset_name, mode, all_bboxes, save_dir):
    num_predicted_frame = 1

    if dataset_name == "ped2":
        num_samples_each_chunk = 100000
    elif dataset_name == "avenue":
        num_samples_each_chunk = 200000 if mode == "test" else 20000
    elif dataset_name == "shanghaitech":
        num_samples_each_chunk = 300000 if mode == "test" else 100000
    else:
        raise NotImplementedError("dataset name should be one of ped2, avenue, or shanghaitech!")

    dataset = get_dataset(
        dataset_name=dataset_name,
        dir=os.path.join(dataset_root, dataset_name),
        context_frame_num=4,
        mode=mode,
        border_mode="predict",
        all_bboxes=all_bboxes,
        patch_size=32,
        of_dataset=False
    )

    flow_dataset = get_dataset(
        dataset_name=dataset_name,
        dir=os.path.join(dataset_root, dataset_name),
        context_frame_num=4,
        mode=mode,
        border_mode="predict",
        all_bboxes=all_bboxes,
        patch_size=32,
        of_dataset=True
    )

    os.makedirs(save_dir, exist_ok=True)

    global_sample_id = 0
    cnt = 0
    chunk_id = 0
    chunked_samples = dict(sample_id=[], appearance=[], motion=[], bbox=[], pred_frame=[])

    for idx in range(len(dataset)):
        if idx % 1000 == 0:
            print(f'Extracting foreground in {idx + 1}-th frame, {len(dataset)} in total')

        frameRange = dataset._context_range(idx)
        batch, _ = dataset.__getitem__(idx)
        flow_batch, _ = flow_dataset.__getitem__(idx)
        cur_bboxes = all_bboxes[idx]

        if len(cur_bboxes) > 0:
            batch = img_batch_tensor2numpy(batch)
            flow_batch = img_batch_tensor2numpy(flow_batch)

            for idx_box in range(cur_bboxes.shape[0]):
                chunked_samples["sample_id"].append(global_sample_id)
                chunked_samples["appearance"].append(batch[idx_box])
                chunked_samples["motion"].append(flow_batch[idx_box])
                chunked_samples["bbox"].append(cur_bboxes[idx_box])
                chunked_samples["pred_frame"].append(frameRange[-num_predicted_frame:])
                global_sample_id += 1
                cnt += 1

                if cnt == num_samples_each_chunk:
                    for key in chunked_samples:
                        chunked_samples[key] = np.array(chunked_samples[key])
                    joblib.dump(chunked_samples, os.path.join(save_dir, f"chunked_samples_{chunk_id:02d}.pkl"))
                    print(f"Chunk {chunk_id} file saved!")

                    chunk_id += 1
                    cnt = 0
                    chunked_samples = dict(sample_id=[], appearance=[], motion=[], bbox=[], pred_frame=[])

    if len(chunked_samples["sample_id"]) != 0:
        for key in chunked_samples:
            chunked_samples[key] = np.array(chunked_samples[key])
        joblib.dump(chunked_samples, os.path.join(save_dir, f"chunked_samples_{chunk_id:02d}.pkl"))
        print(f"Chunk {chunk_id} file saved!")

    print('All samples have been saved!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj_root", type=str, default="/mnt/d/ADA-VAD", help='project root path')
    parser.add_argument("--dataset_name", type=str, default="ped2", help='dataset name')
    parser.add_argument("--mode", type=str, default="train", help='train or test data')
    args = parser.parse_args()

    dataset_root = os.path.join(args.proj_root, "data")
    all_bboxes_path = os.path.join(dataset_root, args.dataset_name, f"{args.dataset_name}_bboxes_{args.mode}.npy")
    all_bboxes = np.load(all_bboxes_path, allow_pickle=True)

    save_dir = os.path.join(dataset_root, args.dataset_name, f"{args.mode}ing", "chunked_samples")

    samples_extraction(
        dataset_root=dataset_root,
        dataset_name=args.dataset_name,
        mode=args.mode,
        all_bboxes=all_bboxes,
        save_dir=save_dir
    )
