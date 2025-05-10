#!/bin/bash
echo "🧹 Cleaning workspace..."

rm -rf plots/
rm -rf wandb/
rm -rf ~/.cache/huggingface/datasets/ms_marco/

echo "✅ Cleanup complete."