# Characterization of Stigmatizing Language in Medical Records

This is the official repository for the ACL 2023 paper, ["Characterization of Stigmatizing Language in Medical Records."](notebooks/resources/ACL2023.pdf) If you publish any research which uses the code, data, and/or models within this repo, we kindly ask you to cite us:

```bibtex
@inproceedings{harrigian2023characterizing,
  title={Characterization of Stigmatizing Language in Medical Records},
  author={Harrigian, Keith and 
          Zirikly, Ayah and 
          Chee, Brant and 
          Ahmad, Alya and 
          Links, {Anne R.} and 
          Saha, Somnath and 
          Beach, {Mary Catherine} and 
          Dredze, Mark},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2023}
}
```

If you encounter issues with the code in this repository, we encourage you to open a new GitHub issue or contact us directly via [email](mailto:kharrigian@jhu.edu). We will be more than happy to help you get up and running with our code, models, and data.

## Resources

This repository only provides an API for interacting with our data and models. The actual data (including annotations) and models are *not* hosted natively within this repository. To access these resources, you must first go through the appropriate credentialing process on [PhysioNet](https://physionet.org). Once you have signed our usage agreement on PhysioNet, you can make full use of our toolkit.

#### Data (MIMIC-IV)

To replicate our experiments or train new models, you will need access to the [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/) and [MIMIC-IV-Notes](https://physionet.org/content/mimic-iv-note/2.2/) datasets (v2.2). Both of these resources are hosted on PhysioNet and require completion of IRB-related training.

Once you have completed the credentialing process, you can easily acquire the minimally necessary data resources using our utility script `./scripts/acquire/get_mimic.sh`. You will be asked for your PhysioNet username and password. Files will be downloaded to `data/resources/mimic-iv/`.

#### Labels and Models

We have opted to keep our labels and models behind a gate for a few reasons. First, although we do not expect our training procedure to encode sensitive information regarding the MIMIC dataset, the risk is nonzero and worth respecting. Furthermore, if we release models in the future which do allow end-users to extract sensitive information, existing end-uers will be able to acquire them seamlessly. Finally, by requiring end-users to complete IRB training prior to accessing our models, we can limit the risk of malevolent use.

Our models can be acquired from PhysioNet after completing the same requirements necessary to access MIMIC data. If you already have access to MIMIC, downloading our models should only require you sign our [data usage agreement](TBD).

Once you have completed this credentialing process, you can use our utility script `./scripts/acquire/get_models_and_labels.sh` to download the pretrained models and annotations. Models will be downloaded to `data/resources/models/`, while annotations will be downloaded to `data/resources/annotations/`.

If you have downloaded the MIMIC-IV dataset, you can create an augmented annotated dataset for training new models using `scripts/acquire/build_mimic.py`.

```bash
python scripts/acquire/build_mimic.py \
    --annotations_dir data/resources/annotations/
    --keywords data/resources/keywords/keywords.json \
    --load_chunksize 1000 \
    --load_window_size 10
```

#### Expected Resource Structure

If the utility scripts above worked appropriately, you should see the directory structure below.

```bash
data/
    resources/
            keywords/
                keywords.json
            annotations/
                annotations.csv
            datasets/
                mimic-iv/
                    admissions.csv.gz
                    diagnoses_icd.csv.gz
                    discharge.csv.gz
                    patients.csv.gz
                    services.csv.gz
                    transfers.csv.gz
            models/
                mimic-iv-discharge_clinical-bert/
                        adamant_fold-0/
                        compliance_fold-0/
                        other_fold-0/
                ...
```

## Installation

We recommend interacting with the resources above using our `stigma` API (Python package). Installing this package should be relatively straightforward. However, please feel free to reach out to us on GitHub or via [email](mailto:kharrigian@jhu.edu) if you encounter issues.

To install the package, you should run the command below from the root of this repository. This command will install all external dependencies, as well as the `stigma` package itself. It is *extremely important* to keep the `-e` environment flag, as it will ensure default data and model paths are preserved.

```bash
pip install -e .
```

We **strongly** recommend using a virtual environment manager (e.g., `conda`) when working with this codebase. This will help limit unintended consequences that arise due to e.g., dependency upgrades. The `conda` [documentation](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) provides all the information you need to set up your first environment.

*Note:* Our toolkit was developed and tested using Python 3.10. We cannot guarantee that other versions of Python will support the entirety of our codebase. That said, we expect the majority of functionality to be preserved as long as you are using Python >= 3.7.

## API Usage

For a quick introduction to our API, we recommend exploring our [quickstart notebook](notebooks/api.demo.ipynb). We have abstracted most of the codebase into a few modules to make interacting with the pretrained models easy.

```python
## Import API Modules
from stigma import StigmaSearch
from stigma import StigmaBaselineModel, StigmaBertModel

## Examples of Clinical Notes
examples = [
    """
    Despite my best advice, the patient remains adamant about leaving the hospital today. 
    Social services is aware of the situation.
    """,
    """
    The patient claims they have remained sober since their last visit, though I smelled
    alcohol on their clothing.
    """
]

## Initialize Keyword Search Wrapper
search_tool = StigmaSearch(context_size=10)

## Run Keyword Search
search_results = search_tool.search(examples)

## Prepare Inputs for the Model
example_ids, example_keywords, example_text = search_tool.format_for_model(search_results=search_results,
                                                                           keyword_category="adamant")

## Initialize Model Wrapper
model = StigmaBertModel(model="mimic-iv-discharge_clinical-bert",
                        keyword_category="adamant",
                        batch_size=8,
                        device="cpu")

## Run Prediction Procedure
predictions = model.predict(text=example_text,
                            keywords=example_keywords)
```

#### A Note on Phrasing

Throughout the repository, you may notice certain naming conventions which do not align with what was presented in the ACL paper. The main differences to be aware of are as follows:

1. `keyword` is what we use to denote the anchors referenced in the paper.
2. `keyword_category` is what we use to refer to the 3 stigma classification tasks.
3. `adamant`, `compliance`, and `other` are shorthand keyword categories which refer to the Credibility and Obstinance, Compliance, and Other Descriptors tasks, respectively.

## Other Functionalities

Although the API shown above should be sufficient for most purposes, this repository contains a substantial amount of additional code which some users may find helpful. This includes scripts which may be used to reproduce our published results. The bash files contained in `jobs/` showcase most of the functionalities. Please see the [README](jobs/README.md) file for more information about each set of commands.