# Raw Entropy Analysis
This directory works for calculating raw image entropy of datasets.

For debugging the entropy calculator, run `python debug.py`.

To integrate the concept-wise API into your datasets, see `concept_entropy.py`.
```
python concept_entropy.py --path [DATADIR] --dataset [DATASET] --save [SAVEPATH] 
```

To visualize concept-wise entropy across datasets, see `vis_all_entropy.py`. Run `python run_vis.py` directly to visualize all datasets.
```
python run_vis.py --path [DATADIR] --dataset [DATASET] --save [false/SAVE.png] 
```

To visualize the regression target of entropy across datasets, see `vis_regression.py`.

To reshape datasets, see `sel_data.py`.
