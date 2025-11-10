"""
Script to download dataset from AWS S3.
"""

import argparse
import subprocess
from pathlib import Path
import sys


def download_dataset(output_dir: str = "data/raw", bucket: str = "phishing-detection-homework-public-bucket"):
    """
    Download dataset from S3.

    Args:
        output_dir: Directory to save data
        bucket: S3 bucket name
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("DOWNLOADING PHISHING DETECTION DATASET")
    print("="*80)
    print(f"\nBucket: s3://{bucket}")
    print(f"Output directory: {output_path.absolute()}\n")

    # AWS S3 command
    cmd = [
        "aws", "s3", "cp",
        f"s3://{bucket}",
        str(output_path),
        "--recursive",
        "--no-sign-request"
    ]

    try:
        print("Downloading...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)

        print("\n" + "="*80)
        print("DOWNLOAD COMPLETED")
        print("="*80)

        # Count downloaded files
        total_files = sum(1 for _ in output_path.rglob('*') if _.is_file())
        print(f"\nTotal files downloaded: {total_files}")

        # Show directory structure
        print("\nDataset structure:")
        for item in sorted(output_path.iterdir()):
            if item.is_dir():
                num_files = sum(1 for _ in item.rglob('*') if _.is_file())
                print(f"  {item.name}/: {num_files} files")

    except subprocess.CalledProcessError as e:
        print(f"\nError downloading dataset: {e}", file=sys.stderr)
        print(f"Error output: {e.stderr}", file=sys.stderr)
        print("\nPlease ensure:")
        print("1. AWS CLI is installed (pip install awscli)")
        print("2. You have internet connection")
        print("3. The S3 bucket is accessible")
        sys.exit(1)
    except FileNotFoundError:
        print("\nError: AWS CLI not found", file=sys.stderr)
        print("Please install it: pip install awscli")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Download phishing detection dataset from S3')
    parser.add_argument('--output_dir', type=str, default='data/raw',
                        help='Output directory for dataset')
    parser.add_argument('--bucket', type=str,
                        default='phishing-detection-homework-public-bucket',
                        help='S3 bucket name')

    args = parser.parse_args()

    download_dataset(args.output_dir, args.bucket)


if __name__ == '__main__':
    main()
