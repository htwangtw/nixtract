# nii-masker
Command-line tool for extracting out ROI data

This is a simple command-line wrapper for `nilearn`'s (https://nilearn.github.io/manipulating_images/masker_objects.html)[Masker object], which lets you easily extract out region-of-interest (ROI) timeseries from functional MRI data while giving you several options to apply post-processing to your MRI data (e.g., spatial smoothing, temporal filtering, confound regression, etc). This tool ultimately aims to extend `nilearn`'s powerful and convenient masking features to non-Python users who wish to analyze fMRI data.


## Installation

First, download this repository to a directory. Then, run `pip install .` to install `niimasker`.

## Running `niimasker`

In order to run `niimasker`, you will need to specify an input directory, and output directory, and a configuration file (see `config_template.json`). This can be run into the command-line as so:

`niimasker /path/to/input /path/to/output config.json`

## Configuring `niimasker`

Coming soon.

## To do

A future goal for `niimasker` is to generate a visual report of your timeseries data once you've extracted out your data. This way, you can get a birds-eye view into your data in order to check for quality issues (e.g., amplitude spikes from motion) and see a log of what post-processing was performed.

In addition to adding command-line arguments (in addition to a configuration file), `niimasker` will have an option to include an "events" file (similar to what SPM or FSL require for first-level analyses) so you can get an index of what event is associated with each time point. This, of course, is only meangingful in task-based fMRI.