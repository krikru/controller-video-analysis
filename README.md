[//]: # (Atom: Turn on preview using ctrl+shift+m)

# Controller video analysis

_For a list of required libraries, please see `environment.yml`._

## Conda environment

To create the Conda environment:

```
conda env create --file environment.yml
```

To update an already existing environment:

```
conda env update --file environment.yml
```

To update `environment.yml` and share your environment:

[//]: # (```)
[//]: # (. export-environment.sh)
[//]: # (```)

1. Run `conda env export --no-builds`
2. Hand-pick the environments that need to be installed manually (i.e. hasn't been installed for you automatically upon installing any other package) and add those to `environment.yml` (while maintaining the alphabetic order of packages in the file).
3. Run `conda env update --file environment.yml` and verify that Conda doesn't want to install, update or uninstall any packages.
  * If Conda does want to install/update/remove packages, choose no, modify your `environment.yml` file and try again

To activate the environment:

```
conda activate controller-videos
```

To deactivate the environment:

```
conda deactivate
```

## Git submodules

To get all Git submodules, which are needed for the project, run

```
git submodule update --init --recursive
```

## Usage

Create and activate the Conda environment, then run

```
python <script>.py -h
```

to display help about how to use the various scripts.
