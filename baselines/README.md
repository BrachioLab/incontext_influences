# Example Selection Baselines

This directory provides code for running different baselines discussed in the paper, including:
* One-shot influence: Rank examples based on their one-shot performance on the Dev set
* (To add) Best set seen: Select the set with the highest performance seen in influence training runs
* (To add) Distance to the Dev set: Rank examples based on a distance measure from the Dev set

As an example, run the following command from the base directory.
```markdown
python baselines/oneshot_influence.py --task=hellaswag \ 
                                      --model_name_or_path=EleutherAI/gpt-j-6b \
                                      --data_dir=data-train400-dev200 \
                                      --out_dir=out_oneshot
```