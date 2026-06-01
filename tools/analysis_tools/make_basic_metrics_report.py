import argparse
import ast
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create a markdown/JSON report from eval logs and profile output.')
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--stdout-log', required=True)
    parser.add_argument('--profile-json', required=True)
    parser.add_argument('--output-md', default=None)
    parser.add_argument('--output-json', default=None)
    return parser.parse_args()


def load_accuracy(stdout_log):
    text = Path(stdout_log).read_text(errors='replace')
    candidates = []
    for match in re.finditer(r'\{[^\n]*\}', text):
        raw = match.group(0)
        try:
            value = ast.literal_eval(raw)
        except Exception:
            continue
        if isinstance(value, dict) and value:
            numeric = {k: v for k, v in value.items()
                       if isinstance(v, (int, float))}
            if numeric:
                candidates.append(numeric)
    return candidates[-1] if candidates else {}


def fmt_float(value):
    if value is None:
        return 'n/a'
    if abs(value) >= 1000:
        return f'{value:,.2f}'
    return f'{value:.6g}'


def make_markdown(report):
    lines = [
        '# RTX 3090 Basic Metrics Report',
        '',
        f"- Config: `{report['config']}`",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Work dir: `{report['work_dir']}`",
        f"- Dataset size: `{report['latency'].get('dataset_size', 'n/a')}`",
        f"- Batch size: `{report['latency'].get('batch_size', 'n/a')}`",
        f"- Profile samples: `{report['latency'].get('samples', 'n/a')}`",
        f"- Requested profile samples: `{report['latency'].get('requested_samples', 'n/a')}`",
        f"- Warmup inferences: `{report['latency'].get('warmup', 'n/a')}`",
        f"- Requested warmup inferences: `{report['latency'].get('requested_warmup', 'n/a')}`",
        '',
        '## Accuracy',
        '',
    ]
    if report['accuracy']:
        lines += ['| Metric | Value |', '| :--- | ---: |']
        for key in sorted(report['accuracy']):
            lines.append(f"| `{key}` | {fmt_float(report['accuracy'][key])} |")
    else:
        lines.append('No numeric accuracy metrics were parsed from the eval log.')

    latency = report['latency']
    lines += [
        '',
        '## Latency And FLOPs',
        '',
        '| Metric | Value |',
        '| :--- | ---: |',
        f"| Single mini-batch latency | {fmt_float(latency.get('single_batch_latency_s'))} s |",
        f"| Mean latency | {fmt_float(latency.get('mean_latency_s'))} s |",
        f"| Median latency | {fmt_float(latency.get('median_latency_s'))} s |",
        f"| FPS | {fmt_float(latency.get('fps'))} |",
        f"| Parameters | {fmt_float(latency.get('params'))} |",
        f"| Profiler FLOPs | {fmt_float(latency.get('profiler_flops'))} |",
        '',
        'Latency was measured with `model.eval()` and `torch.inference_mode()` on one RTX 3090.',
    ]
    return '\n'.join(lines) + '\n'


def main():
    args = parse_args()
    work_dir = Path(args.work_dir)
    profile = json.loads(Path(args.profile_json).read_text())
    report = {
        'work_dir': str(work_dir),
        'config': profile.get('config'),
        'checkpoint': profile.get('checkpoint'),
        'accuracy': load_accuracy(args.stdout_log),
        'latency': profile,
    }

    output_json = Path(args.output_json) if args.output_json else work_dir / 'basic_metrics_report.json'
    output_md = Path(args.output_md) if args.output_md else work_dir / 'basic_metrics_report.md'
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    output_md.write_text(make_markdown(report))
    print(f'Wrote {output_md}')
    print(f'Wrote {output_json}')


if __name__ == '__main__':
    main()
