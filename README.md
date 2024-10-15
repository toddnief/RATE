
## Config Files

We use a single config file for all of the settings to run experiments. The config file is broken up into the three "phases" of the experiment and the settings for each are separate.

### Smoke Test

## Creating Datasets

```
make create_dataset
```

### Dataset Templates

## Scoring Datasets

Note: You can pass a config file to the Make command (which is useful for scheduling multiple jobs with different configurations via SLURM). If you don't pass a value, this defaults to ```config.yaml```.:
```
make score_dataset CONFIG=custom_config.yaml
```

### Scoring Templates

## Calculating Treatment Effects

