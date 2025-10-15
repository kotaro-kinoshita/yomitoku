import argparse
import os

from huggingface_hub import snapshot_download
from yomitoku.configs import DEFAULT_CONFIGS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_hub_repo", type=str, default=None)
    parser.add_argument("--local", type=str, default="KotaroKinoshita")
    args = parser.parse_args()

    if args.hf_hub_repo is None:
        for config in DEFAULT_CONFIGS:
            model_name = config.hf_hub_repo.split("/")[-1]
            local_repo = os.path.join(args.local, model_name)
            snapshot_download(
                repo_id=config.hf_hub_repo,
                revision="main",
                local_dir=local_repo,
            )
    else:
        model_name = args.hf_hub_repo.split("/")[-1]
        local_repo = os.path.join(args.local, model_name)

        snapshot_download(
            repo_id=args.hf_hub_repo,
            revision="main",
            local_dir=local_repo,
        )


if __name__ == "__main__":
    main()
