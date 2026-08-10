# ARHGNN

## 安装

```powershell
pip install -r requirements.txt
```

## 运行

```powershell
python -m arhgnn.train --config configs/arhgnn.yaml
python -m arhgnn.evaluate --pred results_fixed/predictions_external.csv
python -m arhgnn.ablation --config configs/ablation.yaml
```

快速烟雾测试可缩短训练和 bootstrap：

```powershell
python -m arhgnn.train --config configs/arhgnn.yaml --epochs 2 --bootstrap 10
```

## 关键输出

- `results_fixed/predictions_primary_test.csv`
- `results_fixed/predictions_external.csv`
- `results_fixed/metrics_primary_test.csv`
- `results_fixed/metrics_external.csv`
- `results_fixed/per_label_metrics_external.csv`
- `results_fixed/dataset_info.json`
- `results_fixed/best_arhgnn.pt`

## 测试

```powershell
python tests\run_tests.py
```

当前测试覆盖数据列隔离、患者数/类别计数、划分互斥、超图构建不含标签、概率阈值指标和硬编码路径静态检查。

