# Datasets

Extract the Zenodo datasets so that all the files are contained in this `data` folder.

## Official Dataset

The official Same Side Stance Classification Shared Task dataset is available via: [webis-de/argmining19-same-side-classification #data](https://github.com/webis-de/argmining19-same-side-classification#data). Please refer to the instructions on how to retrieve the data in the linked repo.

## Resampled Dataset

Official publication: https://zenodo.org/record/5380989

### Citation

If you use this dataset, please use the following citation:

```bib
@dataset{gregor_wiedemann_2021_5380989,
  author       = {Gregor Wiedemann and
                  Erik Körner and
                  Ahmad Dawar Hakimi and
                  Gerhard Heyer and
                  Martin Potthast},
  title        = {Same Side Stance Classification Resampled Datasets},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.5380989},
  url          = {https://doi.org/10.5281/zenodo.5380989}
}
```


## Artificial Test Set

Official publication: https://zenodo.org/record/5282635

This dataset contains 175 cases. It is manually created to reveal the ability of our models to solve different types of **adversarial** cases for same side stance prediction more systematically. It uses samples from the official Same Side Stance Classification shared task [dataset](https://zenodo.org/record/4382353).

We have selected **25** distinct arguments from the _"gay marriage"_ topic that are short and express their stance clearly.
For each selected argument, we construct new arguments of four distinct types to obtain two pairs, one with same stance, and one with opposing stance:

- **Negation**: a simple negation of the argument.
- **Paraphrase**: alters important words from the argument to synonymous expressions with the same stance. 
- **Argument**: uses an argument from the same topic and stance, but semantically completely different regarding the first one.
- **Citation**: repeats or summarizes the first argument and then expresses agreement or rejection (a case frequently occurring in the dataset).

The types _Paraphrase_, _Argument_ and _Citation_ are also formulated in a negated version to create additional test instances for the opposite stance.

For results on our best model, please refer to our paper.

### Citation

If you use this dataset, please use the following citation:

```bib
@dataset{ahmad_dawar_hakimi_2021_5282635,
  author       = {Ahmad Dawar Hakimi and
                  Erik Körner and
                  Gregor Wiedemann and
                  Gerhard Heyer and
                  Martin Potthast},
  title        = {{Same Side Stance Classification Adversarial Test 
                   Cases}},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.5282635},
  url          = {https://doi.org/10.5281/zenodo.5282635}
}
```
