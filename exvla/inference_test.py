"""
CLI batch evaluation (clinical metrics Excel).
For Gradio UI see: python inference_ui.py
"""
import argparse

from inference_core import EndoVLAInferenceEngine

DEFAULT_MODEL = "/home/rennc1/Documents/Yidong_code/exvla/models/grpo_5090_10percent_image_newDora_Ultimate_Merged"
DEFAULT_JSONL = "/home/rennc1/Documents/Yidong_code/exvla/output_dir/gastrohun_llm_en_images_train_sft_10.jsonl"
DEFAULT_EXCEL = "/home/rennc1/Documents/Yidong_code/exvla/evaluation_report_clinical_metrics.xlsx"


def main():
    parser = argparse.ArgumentParser(description="EndoVLA batch test → Excel")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--jsonl", type=str, default=DEFAULT_JSONL)
    parser.add_argument("--image_dir", type=str, default="")
    parser.add_argument("--output", type=str, default=DEFAULT_EXCEL)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    print("=" * 50)
    print("Loading model...")
    engine = EndoVLAInferenceEngine(args.model_path)
    print(engine.load())

    max_n = args.max_samples if args.max_samples > 0 else None
    df, summary, path = engine.run_test_evaluation(
        args.jsonl,
        image_dir=args.image_dir,
        output_excel=args.output,
        max_samples=max_n,
        progress=lambda p, m: print(f"\r{m}", end="", flush=True),
    )
    print()

    if summary:
        print("=" * 50)
        print("Clinical metrics:")
        print(f"  Exact match:     {summary['exact_match_rate']*100:.2f}%")
        print(f"  Unique label:    {summary['unique_label_rate']*100:.2f}%")
        print(f"  Depth:           {summary['depth_acc']*100:.2f}%")
        print(f"  Depth ±1:        {summary['depth_tol_acc']*100:.2f}%")
        print(f"  Wall:            {summary['wall_acc']*100:.2f}%")
        print(f"  Feature hit:     {summary['feature_acc']*100:.2f}%")
        print(f"  Avg latency:     {summary['avg_latency']:.4f}s")
        print(f"Saved: {path}")
        print("=" * 50)
    else:
        print("No valid samples evaluated.")


if __name__ == "__main__":
    main()
